#!/usr/bin/env python3
"""DeepVQE AEC full training — self-contained single file for HF Jobs.

Inlines all model, dataset, loss, and training code so no project files are
needed on the remote machine. Downloads DNS5 data from HF at startup.

Usage (local test, single GPU):
    python scripts/hf_train_all.py --epochs 5

Usage (HF Jobs, 4x A100):
    bash scripts/hf_train.sh
"""

import argparse
import functools
import math
import os
import random
import subprocess
import sys
from datetime import datetime
from math import gcd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from einops import rearrange
from scipy.signal import fftconvolve, resample_poly
from torch.utils.data import DataLoader, Dataset

# ═══════════════════════════════════════════════════════════════════════════════
# Multi-GPU bootstrap: re-exec under accelerate launch if not already in DDP
# ═══════════════════════════════════════════════════════════════════════════════

def _maybe_launch_distributed():
    """If not already running under accelerate/torchrun, re-launch with DDP."""
    if "LOCAL_RANK" in os.environ:
        return  # already distributed
    n_gpus = torch.cuda.device_count()
    if n_gpus <= 1:
        return  # single GPU, no need for DDP
    print(f"Detected {n_gpus} GPUs, re-launching with accelerate launch", flush=True)
    sys.exit(subprocess.call([
        sys.executable, "-m", "accelerate.commands.launch",
        "--num_processes", str(n_gpus),
        "--num_machines", "1",
        "--dynamo_backend", "no",
        "--mixed_precision", "bf16",
        *sys.argv,
    ]))

# ═══════════════════════════════════════════════════════════════════════════════
# STFT helpers
# ═══════════════════════════════════════════════════════════════════════════════

@functools.lru_cache(maxsize=4)
def make_window(n_fft, device=None):
    return torch.sqrt(torch.hann_window(n_fft, device=device) + 1e-12)

def stft(x, n_fft=512, hop_length=256):
    window = make_window(n_fft, device=x.device)
    X = torch.stft(x, n_fft, hop_length, window=window, return_complex=True)
    return torch.view_as_real(X)

def istft(X, n_fft=512, hop_length=256, length=None):
    window = make_window(n_fft, device=X.device)
    X_complex = torch.complex(X[..., 0], X[..., 1])
    return torch.istft(X_complex, n_fft, hop_length, window=window, length=length)

# ═══════════════════════════════════════════════════════════════════════════════
# Model blocks (identical to hf_overfit_all.py)
# ═══════════════════════════════════════════════════════════════════════════════

class FE(nn.Module):
    def __init__(self, c=0.3):
        super().__init__()
        self.c = c

    def forward(self, x):
        x_mag = torch.sqrt(x[..., [0]] ** 2 + x[..., [1]] ** 2 + 1e-12)
        x_c = torch.div(x, x_mag.pow(1 - self.c) + 1e-12)
        return x_c.permute(0, 3, 2, 1).contiguous()


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pad = nn.ZeroPad2d([1, 1, 3, 0])
        self.conv = nn.Conv2d(channels, channels, kernel_size=(4, 3))
        self.bn = nn.BatchNorm2d(channels)
        self.elu = nn.ELU()

    def forward(self, x):
        return self.elu(self.bn(self.conv(self.pad(x)))) + x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(4, 3), stride=(1, 2)):
        super().__init__()
        self.pad = nn.ZeroPad2d([1, 1, 3, 0])
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU()
        self.resblock = ResidualBlock(out_channels)

    def forward(self, x):
        return self.resblock(self.elu(self.bn(self.conv(self.pad(x)))))


class Bottleneck(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        y = rearrange(x, "b c t f -> b t (c f)")
        y = self.gru(y)[0]
        y = self.fc(y)
        y = rearrange(y, "b t (c f) -> b c t f", c=x.shape[1])
        return y


class SubpixelConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(4, 3)):
        super().__init__()
        self.pad = nn.ZeroPad2d([1, 1, 3, 0])
        self.conv = nn.Conv2d(in_channels, out_channels * 2, kernel_size)

    def forward(self, x):
        y = self.conv(self.pad(x))
        y = rearrange(y, "b (r c) t f -> b c t (r f)", r=2)
        return y


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(4, 3), is_last=False):
        super().__init__()
        self.skip_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.resblock = ResidualBlock(in_channels)
        self.deconv = SubpixelConv2d(in_channels, out_channels, kernel_size)
        self.is_last = is_last
        if not is_last:
            self.bn = nn.BatchNorm2d(out_channels)
            self.elu = nn.ELU()

    def forward(self, x, x_en):
        y = x + self.skip_conv(x_en)
        y = self.deconv(self.resblock(y))
        if not self.is_last:
            y = self.elu(self.bn(y))
        return y

# ═══════════════════════════════════════════════════════════════════════════════
# AlignBlock
# ═══════════════════════════════════════════════════════════════════════════════

class AlignBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, dmax=32, temperature=1.0):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.dmax = dmax
        self.temperature = temperature
        self.pconv_mic = nn.Conv2d(in_channels, hidden_channels, 1)
        self.pconv_ref = nn.Conv2d(in_channels, hidden_channels, 1)
        self.unfold_k = nn.Sequential(
            nn.ZeroPad2d([0, 0, dmax - 1, 0]),
            nn.Unfold((dmax, 1)),
        )
        self.conv = nn.Sequential(
            nn.ZeroPad2d([1, 1, 4, 0]),
            nn.Conv2d(hidden_channels, 1, (5, 3)),
        )

    def forward(self, x_mic, x_ref, return_delay=False):
        B, C, T, F = x_ref.shape
        Q = self.pconv_mic(x_mic)
        K = self.pconv_ref(x_ref)
        Ku = self.unfold_k(K)
        Ku = Ku.view(B, self.hidden_channels, self.dmax, T, F)
        Ku = Ku.permute(0, 1, 3, 2, 4).contiguous()
        V = torch.sum(Q.unsqueeze(-2) * Ku, dim=-1)
        V = V / math.sqrt(F)
        V = self.conv(V)
        A = torch.softmax(V / self.temperature, dim=-1)
        unfold_ref = nn.functional.pad(x_ref, [0, 0, self.dmax - 1, 0])
        ref_unfolded = unfold_ref.unfold(2, self.dmax, 1)
        ref_unfolded = ref_unfolded.permute(0, 1, 2, 4, 3).contiguous()
        A_expanded = A[:, :, :, :, None]
        aligned = torch.sum(ref_unfolded * A_expanded, dim=-2)
        if return_delay:
            return aligned, A.squeeze(1)
        return aligned

# ═══════════════════════════════════════════════════════════════════════════════
# CCM
# ═══════════════════════════════════════════════════════════════════════════════

class CCM(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("v_real", torch.tensor([1, -1/2, -1/2], dtype=torch.float32))
        self.register_buffer("v_imag", torch.tensor([0, np.sqrt(3)/2, -np.sqrt(3)/2], dtype=torch.float32))
        self.unfold = nn.Sequential(
            nn.ZeroPad2d([1, 1, 2, 0]),
            nn.Unfold(kernel_size=(3, 3)),
        )

    def forward(self, m, x):
        m = rearrange(m, "b (r c) t f -> b r c t f", r=3)
        H_real = torch.sum(self.v_real[None, :, None, None, None] * m, dim=1)
        H_imag = torch.sum(self.v_imag[None, :, None, None, None] * m, dim=1)
        M_real = rearrange(H_real, "b (m n) t f -> b m n t f", m=3)
        M_imag = rearrange(H_imag, "b (m n) t f -> b m n t f", m=3)
        x_perm = x.permute(0, 3, 2, 1).contiguous()
        x_unfold = self.unfold(x_perm)
        x_unfold = rearrange(x_unfold, "b (c m n) (t f) -> b c m n t f", m=3, n=3, f=x.shape[1])
        x_enh_real = torch.sum(M_real * x_unfold[:, 0] - M_imag * x_unfold[:, 1], dim=(1, 2))
        x_enh_imag = torch.sum(M_real * x_unfold[:, 1] + M_imag * x_unfold[:, 0], dim=(1, 2))
        x_enh = torch.stack([x_enh_real, x_enh_imag], dim=3).transpose(1, 2).contiguous()
        return x_enh

# ═══════════════════════════════════════════════════════════════════════════════
# DeepVQEAEC model
# ═══════════════════════════════════════════════════════════════════════════════

class DeepVQEAEC(nn.Module):
    def __init__(self, mic_channels=None, far_channels=None,
                 align_hidden=32, dmax=32, power_law_c=0.3):
        super().__init__()
        if mic_channels is None:
            mic_channels = [2, 64, 128, 128, 128, 128]
        if far_channels is None:
            far_channels = [2, 32, 128]

        self.fe_mic = FE(c=power_law_c)
        self.fe_ref = FE(c=power_law_c)
        self.mic_enc1 = EncoderBlock(mic_channels[0], mic_channels[1])
        self.mic_enc2 = EncoderBlock(mic_channels[1], mic_channels[2])
        self.far_enc1 = EncoderBlock(far_channels[0], far_channels[1])
        self.far_enc2 = EncoderBlock(far_channels[1], far_channels[2])
        self.align = AlignBlock(in_channels=mic_channels[2], hidden_channels=align_hidden, dmax=dmax)
        self.mic_enc3 = EncoderBlock(mic_channels[2] * 2, mic_channels[3])
        self.mic_enc4 = EncoderBlock(mic_channels[3], mic_channels[4])
        self.mic_enc5 = EncoderBlock(mic_channels[4], mic_channels[5])
        self.bottleneck = Bottleneck(mic_channels[5] * 9, mic_channels[5] * 9 // 2)
        self.dec5 = DecoderBlock(mic_channels[5], mic_channels[4])
        self.dec4 = DecoderBlock(mic_channels[4], mic_channels[3])
        self.dec3 = DecoderBlock(mic_channels[3], mic_channels[2])
        self.dec2 = DecoderBlock(mic_channels[2], mic_channels[1])
        self.dec1 = DecoderBlock(mic_channels[1], 27, is_last=True)
        self.ccm = CCM()
        self._init_ccm_identity()

    def _init_ccm_identity(self):
        conv = self.dec1.deconv.conv
        with torch.no_grad():
            conv.bias.zero_()
            conv.bias[7] = 1.0
            conv.bias[34] = 1.0

    def forward(self, mic_stft, ref_stft, return_delay=False):
        mic_fe = self.fe_mic(mic_stft)
        ref_fe = self.fe_ref(ref_stft)
        mic_e1 = self.mic_enc1(mic_fe)
        mic_e2 = self.mic_enc2(mic_e1)
        far_e1 = self.far_enc1(ref_fe)
        far_e2 = self.far_enc2(far_e1)
        align_result = self.align(mic_e2, far_e2, return_delay=return_delay)
        if return_delay:
            aligned_far, delay_dist = align_result
        else:
            aligned_far = align_result
        concat = torch.cat([mic_e2, aligned_far], dim=1)
        mic_e3 = self.mic_enc3(concat)
        mic_e4 = self.mic_enc4(mic_e3)
        mic_e5 = self.mic_enc5(mic_e4)
        bn = self.bottleneck(mic_e5)
        d5 = self.dec5(bn, mic_e5)[..., :mic_e4.shape[-1]]
        d4 = self.dec4(d5, mic_e4)[..., :mic_e3.shape[-1]]
        d3 = self.dec3(d4, mic_e3)[..., :mic_e2.shape[-1]]
        d2 = self.dec2(d3, mic_e2)[..., :mic_e1.shape[-1]]
        d1 = self.dec1(d2, mic_e1)[..., :mic_fe.shape[-1]]
        enhanced = self.ccm(d1, mic_stft)
        if return_delay:
            return enhanced, delay_dist, d1
        return enhanced

# ═══════════════════════════════════════════════════════════════════════════════
# Loss (full multi-component)
# ═══════════════════════════════════════════════════════════════════════════════

class DeepVQELoss(nn.Module):
    def __init__(self, plcmse_weight=1.0, mag_l1_weight=0.5, time_l1_weight=0.5,
                 delay_weight=1.0, entropy_weight=0.01,
                 n_fft=512, hop_length=256, power_law_c=0.3):
        super().__init__()
        self.plcmse_weight = plcmse_weight
        self.mag_l1_weight = mag_l1_weight
        self.time_l1_weight = time_l1_weight
        self.delay_weight = delay_weight
        self.entropy_weight = entropy_weight
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.c = power_law_c

    def forward(self, pred_stft, target_stft, target_wav=None,
                delay_dist=None, delay_samples=None, dmax=32):
        components = {}

        pred_mag = torch.sqrt(pred_stft[..., 0] ** 2 + pred_stft[..., 1] ** 2 + 1e-12)
        target_mag = torch.sqrt(target_stft[..., 0] ** 2 + target_stft[..., 1] ** 2 + 1e-12)

        # Power-law compressed MSE
        plcmse = torch.tensor(0.0, device=pred_stft.device)
        if self.plcmse_weight > 0:
            pred_c = pred_stft * pred_mag.unsqueeze(-1).pow(self.c - 1)
            target_c = target_stft * target_mag.unsqueeze(-1).pow(self.c - 1)
            plcmse = torch.mean((pred_c - target_c) ** 2)
        components["plcmse"] = plcmse

        # Magnitude L1
        mag_l1 = torch.mean(torch.abs(pred_mag - target_mag))
        components["mag_l1"] = mag_l1

        # Time-domain L1
        time_l1 = torch.tensor(0.0, device=pred_stft.device)
        if target_wav is not None and self.time_l1_weight > 0:
            pred_wav = istft(pred_stft, self.n_fft, self.hop_length, length=target_wav.shape[-1])
            time_l1 = torch.mean(torch.abs(pred_wav - target_wav))
        components["time_l1"] = time_l1

        # Delay cross-entropy
        delay_loss = torch.tensor(0.0, device=pred_stft.device)
        delay_acc = torch.tensor(0.0, device=pred_stft.device)
        if delay_dist is not None and delay_samples is not None and self.delay_weight > 0:
            delay_loss, delay_acc = compute_delay_loss(
                delay_dist, delay_samples, self.hop_length, dmax)
        components["delay"] = delay_loss
        components["delay_acc"] = delay_acc

        # Entropy regularization (sharpen attention)
        entropy = torch.tensor(0.0, device=pred_stft.device)
        if delay_dist is not None and self.entropy_weight > 0:
            entropy = -(delay_dist * torch.log(delay_dist + 1e-10)).sum(dim=-1).mean()
        components["entropy"] = entropy

        total = (self.plcmse_weight * plcmse
                 + self.mag_l1_weight * mag_l1
                 + self.time_l1_weight * time_l1
                 + self.delay_weight * delay_loss
                 + self.entropy_weight * entropy)
        components["total"] = total
        return total, components

# ═══════════════════════════════════════════════════════════════════════════════
# Audio synthesis helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _load_audio(path, target_len, sr=16000):
    info = sf.info(path)
    file_sr = info.samplerate
    total_frames = info.frames

    if file_sr != sr:
        audio, _ = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
        g = gcd(sr, file_sr)
        audio = resample_poly(audio, sr // g, file_sr // g).astype(np.float32)
    elif total_frames <= target_len:
        audio, _ = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
    else:
        start = random.randint(0, total_frames - target_len)
        audio, _ = sf.read(path, dtype="float32", start=start, stop=start + target_len)
        if audio.ndim > 1:
            audio = audio[:, 0]

    if len(audio) < target_len:
        pad_total = target_len - len(audio)
        pad_left = random.randint(0, pad_total)
        audio = np.pad(audio, (pad_left, pad_total - pad_left))
    elif len(audio) > target_len:
        start = random.randint(0, len(audio) - target_len)
        audio = audio[start:start + target_len]
    return audio.astype(np.float32)


def _load_random_rir(rir_list, max_length_samples=None):
    path = random.choice(rir_list)
    rir, _ = sf.read(path, dtype="float32")
    if rir.ndim > 1:
        rir = rir[:, 0]
    rir = rir / (np.abs(rir).max() + 1e-12)
    if max_length_samples is not None and len(rir) > max_length_samples:
        rir = rir[:max_length_samples]
        fade_len = min(160, max_length_samples)
        fade = np.cos(np.linspace(0, np.pi / 2, fade_len)).astype(np.float32) ** 2
        rir[-fade_len:] *= fade
    return rir


def _rms(x):
    return np.sqrt(np.mean(x ** 2) + 1e-12)


def _scale_to_snr(signal, noise, snr_db):
    sig_rms = _rms(signal)
    noise_rms = _rms(noise)
    target_noise_rms = sig_rms / (10 ** (snr_db / 20))
    return noise * (target_noise_rms / (noise_rms + 1e-12))


def _scale_to_ser(nearend, echo, ser_db):
    ne_rms = _rms(nearend)
    echo_rms = _rms(echo)
    target_echo_rms = ne_rms / (10 ** (ser_db / 20))
    return echo * (target_echo_rms / (echo_rms + 1e-12))

# ═══════════════════════════════════════════════════════════════════════════════
# Datasets
# ═══════════════════════════════════════════════════════════════════════════════

def _collect_audio_files(directory):
    if not directory:
        return []
    d = Path(directory)
    if not d.exists():
        return []
    files = []
    for ext in ("*.wav", "*.flac"):
        files.extend(str(f) for f in d.rglob(ext))
    files.sort()
    return files


class OnlineSynthDataset(Dataset):
    """Online synthesis AEC dataset — generates fresh examples on each access."""

    def __init__(self, clean_files, noise_files, farend_files, rir_files,
                 sr=16000, target_len=48000, n_fft=512, hop_length=256,
                 snr_range=(5, 40), ser_range=(-10, 10), delay_range=(0, 5120),
                 single_talk_prob=0.2, max_rir_length_ms=500, drr_range=(3, 30)):
        self.clean_files = clean_files
        self.noise_files = noise_files
        self.farend_files = farend_files or clean_files
        self.rir_files = rir_files
        self.sr = sr
        self.target_len = target_len
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.snr_range = snr_range
        self.ser_range = ser_range
        self.delay_range = delay_range
        self.single_talk_prob = single_talk_prob
        self.max_rir_length_ms = max_rir_length_ms
        self.drr_range = drr_range

        assert clean_files, "No clean files found"
        assert noise_files, "No noise files found"
        print(f"OnlineSynthDataset: {len(clean_files)} clean, {len(noise_files)} noise, "
              f"{len(self.farend_files)} farend, {len(rir_files) if rir_files else 0} RIR")

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        nearend = _load_audio(self.clean_files[idx], self.target_len, self.sr)
        farend = _load_audio(random.choice(self.farend_files), self.target_len, self.sr)
        noise = _load_audio(random.choice(self.noise_files), self.target_len, self.sr)

        is_single_talk = random.random() < self.single_talk_prob
        max_rir_samples = int(self.max_rir_length_ms * self.sr / 1000) if self.max_rir_length_ms else None

        # Apply RIR
        if self.rir_files:
            rir = _load_random_rir(self.rir_files, max_rir_samples)
            nearend_reverbed = fftconvolve(nearend, rir)[:self.target_len].astype(np.float32)
            farend_rir = _load_random_rir(self.rir_files, max_rir_samples)
            echo = fftconvolve(farend, farend_rir)[:self.target_len].astype(np.float32)
        else:
            nearend_reverbed = nearend.copy()
            echo = farend.copy()

        # DRR mixing
        drr_db = None
        if self.drr_range is not None and self.rir_files:
            drr_db = random.uniform(*self.drr_range)
            drr_linear = 10 ** (drr_db / 10)
            alpha = drr_linear / (1 + drr_linear)
            nearend_in_mic = (alpha * nearend + (1 - alpha) * nearend_reverbed).astype(np.float32)
        else:
            nearend_in_mic = nearend_reverbed

        # Delay
        delay_ms = random.uniform(*self.delay_range)
        delay_samples = int(delay_ms * self.sr / 1000)
        if delay_samples > 0:
            echo = np.pad(echo, (delay_samples, 0))[:self.target_len]

        # SNR/SER
        snr_db = random.uniform(*self.snr_range)
        ser_db = random.uniform(*self.ser_range)

        if is_single_talk:
            nearend_in_mic = np.zeros_like(nearend)
            clean = np.zeros_like(nearend)
            scenario = "single_talk_farend"
        else:
            clean = nearend.copy()
            scenario = "double_talk"

        if not is_single_talk and _rms(nearend_in_mic) > 1e-6:
            echo = _scale_to_ser(nearend_in_mic, echo, ser_db)
            noise = _scale_to_snr(nearend_in_mic, noise, snr_db)
        else:
            noise = _scale_to_snr(echo, noise, snr_db)

        mic = nearend_in_mic + echo + noise

        # Normalize
        peak = max(np.abs(mic).max(), np.abs(farend).max(), 1e-6)
        if peak > 0.95:
            scale = 0.9 / peak
            mic *= scale
            farend_out = farend * scale
            clean *= scale
        else:
            farend_out = farend

        mic_t = torch.from_numpy(mic).unsqueeze(0)
        ref_t = torch.from_numpy(farend_out).unsqueeze(0)
        clean_t = torch.from_numpy(clean).unsqueeze(0)

        return {
            "mic_stft": stft(mic_t, self.n_fft, self.hop_length).squeeze(0),
            "ref_stft": stft(ref_t, self.n_fft, self.hop_length).squeeze(0),
            "clean_stft": stft(clean_t, self.n_fft, self.hop_length).squeeze(0),
            "mic_wav": mic_t.squeeze(0),
            "clean_wav": clean_t.squeeze(0),
            "delay_samples": delay_samples,
            "metadata": {
                "delay_ms": delay_ms, "delay_samples": delay_samples,
                "snr_db": snr_db, "ser_db": ser_db, "drr_db": drr_db,
                "scenario": scenario,
            },
        }

# ═══════════════════════════════════════════════════════════════════════════════
# Training helpers
# ═══════════════════════════════════════════════════════════════════════════════

def collate_fn(batch):
    return {
        "mic_stft": torch.stack([b["mic_stft"] for b in batch]),
        "ref_stft": torch.stack([b["ref_stft"] for b in batch]),
        "clean_stft": torch.stack([b["clean_stft"] for b in batch]),
        "mic_wav": torch.stack([b["mic_wav"] for b in batch]),
        "clean_wav": torch.stack([b["clean_wav"] for b in batch]),
        "delay_samples": torch.tensor([b["delay_samples"] for b in batch], dtype=torch.long),
        "metadata": [b["metadata"] for b in batch],
    }


def delay_samples_to_frame(delay_samples, hop_length, dmax):
    delay_frames = torch.round(delay_samples.float() / hop_length).long()
    return ((dmax - 1) - delay_frames).clamp(0, dmax - 1)


def compute_delay_loss(delay_dist, delay_samples, hop_length, dmax):
    B, T, D = delay_dist.shape
    target_frames = delay_samples_to_frame(delay_samples, hop_length, dmax)
    log_probs = torch.log(delay_dist + 1e-10)
    target_expanded = target_frames[:, None, None].expand(B, T, 1)
    nll = -log_probs.gather(dim=-1, index=target_expanded).squeeze(-1)
    loss = nll.mean()
    avg_dist = delay_dist.mean(dim=1)
    peak_frames = avg_dist.argmax(dim=-1)
    accuracy = ((peak_frames - target_frames).abs() <= 1).float().mean()
    return loss, accuracy


def compute_erle(mic_wav, enhanced_wav, clean_wav):
    echo_plus_noise = mic_wav - clean_wav
    residual = enhanced_wav - clean_wav
    echo_power = (echo_plus_noise ** 2).sum(dim=-1)
    residual_power = (residual ** 2).sum(dim=-1)
    return (10 * torch.log10(echo_power / (residual_power + 1e-10))).mean()


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def plot_spectrogram_comparison(mic_stft, enh_stft, clean_stft, sr, hop_length):
    """2x2 grid: mic | enhanced | clean | residual."""
    def _mag_db(x):
        if x.dim() == 4:
            x = x[0]
        mag = torch.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2 + 1e-12)
        return 20 * torch.log10(mag + 1e-12).cpu().numpy()

    mic_db = _mag_db(mic_stft)
    enh_db = _mag_db(enh_stft)
    clean_db = _mag_db(clean_stft)
    res_db = enh_db - clean_db

    vmin = max(min(mic_db.min(), enh_db.min(), clean_db.min()), -80.0)
    vmax = max(mic_db.max(), enh_db.max(), clean_db.max())

    F, T = mic_db.shape
    t_axis = np.arange(T) * hop_length / sr
    f_axis = np.arange(F) * sr / ((F - 1) * 2)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    titles = ["Mic", "Enhanced", "Clean", "Residual (enh - clean)"]
    data = [mic_db, enh_db, clean_db, res_db]

    for ax, title, d in zip(axes.flat, titles, data):
        if title == "Residual (enh - clean)":
            im = ax.pcolormesh(t_axis, f_axis / 1000, d, vmin=-80, vmax=80,
                               cmap="RdBu_r", shading="auto")
        else:
            im = ax.pcolormesh(t_axis, f_axis / 1000, d, vmin=vmin, vmax=vmax,
                               cmap="magma", shading="auto")
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Freq (kHz)")
        fig.colorbar(im, ax=ax, label="dB")

    fig.tight_layout()
    return fig


def plot_delay_with_gt(delay_dist, gt_delay_samples, hop_length, dmax):
    """Delay heatmap (T x dmax) with ground-truth overlay."""
    if delay_dist.dim() == 3:
        delay_dist = delay_dist[0]
    dd = delay_dist.detach().cpu().numpy()
    T, D = dd.shape

    if isinstance(gt_delay_samples, torch.Tensor):
        gt_delay_samples = gt_delay_samples.item()

    gt_frame = dmax - 1 - round(gt_delay_samples / hop_length)
    gt_frame = max(0, min(dmax - 1, gt_frame))

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    im = ax.imshow(dd.T, aspect="auto", origin="lower", cmap="viridis",
                   interpolation="nearest")
    fig.colorbar(im, ax=ax, label="Probability")

    ax.axhline(gt_frame, color="red", linestyle="--", linewidth=1.5,
               label=f"GT delay={gt_delay_samples} samp (frame {gt_frame})")

    peak_frames = dd.argmax(axis=1)
    for t in range(T):
        correct = abs(int(peak_frames[t]) - gt_frame) <= 1
        color = "lime" if correct else "red"
        ax.plot(t, peak_frames[t], "o", color=color, markersize=2)

    ax.set_xlabel("Frame")
    ax.set_ylabel("Delay index")
    ax.set_title("Delay Distribution")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def _unwrap(model, accelerator=None):
    if accelerator:
        model = accelerator.unwrap_model(model)
    return getattr(model, "_orig_mod", model)


def save_checkpoint(model, optimizer, scheduler, epoch, loss, path, accelerator=None):
    torch.save({
        "epoch": epoch,
        "model_state_dict": _unwrap(model, accelerator).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss,
    }, path)

# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════

class Config:
    """Full DNS5 training config matching configs/hf_train.yaml."""
    # Model
    mic_channels = [2, 64, 128, 128, 128, 128]
    far_channels = [2, 32, 128]
    align_hidden = 32
    dmax = 32
    power_law_c = 0.3
    align_temp_start = 1.0
    align_temp_end = 0.1
    align_temp_epochs = 50
    # Audio
    sample_rate = 16000
    n_fft = 512
    hop_length = 256
    # Training
    batch_size = 32
    grad_accum_steps = 2
    num_workers = 8
    lr = 1.2e-3
    weight_decay = 5e-7
    epochs = 250
    clip_length_sec = 3.0
    grad_clip = 5.0
    warmup_epochs = 3
    checkpoint_every = 5
    keep_checkpoints = 5
    early_stop_patience = 20
    # Loss
    plcmse_weight = 1.0
    mag_l1_weight = 0.5
    time_l1_weight = 0.5
    delay_weight = 1.0
    entropy_weight = 0.01
    # Data
    clean_dir = "/data/dns5/clean"
    noise_dir = "/data/dns5/noise"
    rir_dir = "/data/dns5/impulse_responses"
    farend_dir = ""
    num_val = 500
    snr_range = (5, 40)
    ser_range = (-10, 10)
    delay_range = (0, 5120)
    single_talk_prob = 0.2
    max_rir_length_ms = 500
    drr_range = (3, 30)
    # Hub
    push_to_hub = True
    hub_model_id = "richiejp/deepvqe-aec"
    hf_dataset_id = "richiejp/dns5-16k"
    push_logs_every = 5

# ═══════════════════════════════════════════════════════════════════════════════
# Data download
# ═══════════════════════════════════════════════════════════════════════════════

def download_data(cfg):
    """Download DNS5 data from HF dataset."""
    if Path(cfg.clean_dir).exists():
        n = len(list(Path(cfg.clean_dir).rglob("*.wav"))[:10])
        if n > 0:
            print(f"Data already exists at {cfg.clean_dir}, skipping download")
            return
    print(f"=== Downloading DNS5 from {cfg.hf_dataset_id} ===")
    from huggingface_hub import snapshot_download
    snapshot_download(
        cfg.hf_dataset_id,
        local_dir="/data/dns5",
        repo_type="dataset",
    )
    for subdir in ["clean", "noise", "impulse_responses"]:
        p = Path(f"/data/dns5/{subdir}")
        if p.exists():
            n = sum(1 for _ in p.rglob("*") if _.is_file())
            print(f"  {subdir}: {n} files")
        else:
            print(f"  WARNING: {subdir} not found!")

# ═══════════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════════

def train(cfg):
    from accelerate import Accelerator
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.grad_accum_steps,
        mixed_precision="bf16",
    )
    device = accelerator.device
    is_main = accelerator.is_main_process

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    if is_main:
        print(f"Device: {device}, num_processes: {accelerator.num_processes}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = Path("checkpoints/hf_train") / run_id
    log_dir = Path("logs/hf_train") / run_id
    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir) if is_main else None

    # Hub
    hub_api, hub_repo_id = None, None
    if is_main and cfg.push_to_hub and cfg.hub_model_id:
        from huggingface_hub import HfApi
        hub_api = HfApi()
        hub_repo_id = cfg.hub_model_id
        hub_api.create_repo(hub_repo_id, exist_ok=True)

    # Model
    raw_model = DeepVQEAEC(
        mic_channels=cfg.mic_channels, far_channels=cfg.far_channels,
        align_hidden=cfg.align_hidden, dmax=cfg.dmax, power_law_c=cfg.power_law_c,
    )
    n_params = sum(p.numel() for p in raw_model.parameters())
    if is_main:
        print(f"Parameters: {n_params:,}")

    criterion = DeepVQELoss(
        plcmse_weight=cfg.plcmse_weight, mag_l1_weight=cfg.mag_l1_weight,
        time_l1_weight=cfg.time_l1_weight, delay_weight=cfg.delay_weight,
        entropy_weight=cfg.entropy_weight, n_fft=cfg.n_fft,
        hop_length=cfg.hop_length, power_law_c=cfg.power_law_c,
    )
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Dataset
    target_len = int(cfg.clip_length_sec * cfg.sample_rate)
    all_clean = _collect_audio_files(cfg.clean_dir)
    noise_files = _collect_audio_files(cfg.noise_dir)
    farend_files = _collect_audio_files(cfg.farend_dir) or all_clean
    rir_files = _collect_audio_files(cfg.rir_dir)

    # Train/val split
    val_clean = all_clean[:cfg.num_val]
    train_clean = all_clean[cfg.num_val:]

    if is_main:
        print(f"Train: {len(train_clean)} clean, Val: {len(val_clean)} clean, "
              f"Noise: {len(noise_files)}, RIR: {len(rir_files)}")

    train_ds = OnlineSynthDataset(
        clean_files=train_clean, noise_files=noise_files,
        farend_files=farend_files, rir_files=rir_files,
        sr=cfg.sample_rate, target_len=target_len,
        n_fft=cfg.n_fft, hop_length=cfg.hop_length,
        snr_range=cfg.snr_range, ser_range=cfg.ser_range,
        delay_range=cfg.delay_range, single_talk_prob=cfg.single_talk_prob,
        max_rir_length_ms=cfg.max_rir_length_ms, drr_range=cfg.drr_range,
    )
    val_ds = OnlineSynthDataset(
        clean_files=val_clean, noise_files=noise_files,
        farend_files=farend_files, rir_files=rir_files,
        sr=cfg.sample_rate, target_len=target_len,
        n_fft=cfg.n_fft, hop_length=cfg.hop_length,
        snr_range=cfg.snr_range, ser_range=cfg.ser_range,
        delay_range=cfg.delay_range, single_talk_prob=cfg.single_talk_prob,
        max_rir_length_ms=cfg.max_rir_length_ms, drr_range=cfg.drr_range,
    )

    pin = device.type == "cuda"
    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        random.seed(worker_seed + worker_id)
        np.random.seed(worker_seed + worker_id)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=collate_fn,
        drop_last=True, pin_memory=pin,
        persistent_workers=cfg.num_workers > 0, worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate_fn,
        pin_memory=pin, persistent_workers=cfg.num_workers > 0,
        worker_init_fn=worker_init_fn,
    )

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        raw_model, optimizer, train_loader, val_loader,
    )
    criterion = criterion.to(device)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6,
    )

    best_val_loss = float("inf")
    no_improve_count = 0
    global_step = 0

    for epoch in range(cfg.epochs):
        model.train()

        # Anneal AlignBlock temperature
        t_start, t_end, t_epochs = cfg.align_temp_start, cfg.align_temp_end, cfg.align_temp_epochs
        temperature = t_start + (t_end - t_start) * min(epoch, t_epochs) / max(t_epochs, 1)
        _unwrap(model, accelerator).align.temperature = temperature

        # Linear warmup
        if epoch < cfg.warmup_epochs and cfg.warmup_epochs > 0:
            warmup_factor = (epoch + 1) / cfg.warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = cfg.lr * warmup_factor

        epoch_loss = 0
        epoch_delay_acc = 0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}", disable=not is_main)
        for batch in pbar:
            mic_stft = batch["mic_stft"].to(device, non_blocking=True)
            ref_stft = batch["ref_stft"].to(device, non_blocking=True)
            clean_stft = batch["clean_stft"].to(device, non_blocking=True)
            clean_wav = batch["clean_wav"].to(device, non_blocking=True)
            delay_samp = batch["delay_samples"].to(device, non_blocking=True)

            with accelerator.accumulate(model):
                enhanced, delay_dist, _ = model(mic_stft, ref_stft, return_delay=True)
                loss, components = criterion(
                    enhanced, clean_stft, clean_wav,
                    delay_dist=delay_dist, delay_samples=delay_samp, dmax=cfg.dmax,
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    global_step += 1
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += components["total"].item()
            epoch_delay_acc += components["delay_acc"].item()
            n_batches += 1

            if is_main:
                pbar.set_postfix(
                    loss=f"{components['total'].item():.4f}",
                    dacc=f"{components['delay_acc'].item():.0%}",
                )

            if is_main and accelerator.sync_gradients:
                writer.add_scalar("train/loss", components["total"].item(), global_step)
                writer.add_scalar("train/plcmse", components["plcmse"].item(), global_step)
                writer.add_scalar("train/mag_l1", components["mag_l1"].item(), global_step)
                writer.add_scalar("train/time_l1", components["time_l1"].item(), global_step)
                writer.add_scalar("train/delay_loss", components["delay"].item(), global_step)
                writer.add_scalar("train/delay_acc", components["delay_acc"].item(), global_step)
                writer.add_scalar("train/entropy", components["entropy"].item(), global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("train/temperature", temperature, global_step)

        if epoch >= cfg.warmup_epochs:
            scheduler.step()

        # Epoch summary
        avg_train_loss = epoch_loss / max(n_batches, 1)
        avg_delay_acc = epoch_delay_acc / max(n_batches, 1)

        # Validation
        model.eval()
        raw_model_ref = _unwrap(model, accelerator)
        val_loss_sum = torch.tensor(0.0, device=device)
        val_erle_sum = torch.tensor(0.0, device=device)
        val_n = torch.tensor(0, device=device)
        tb_samples = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                mic_stft = batch["mic_stft"].to(device, non_blocking=True)
                ref_stft = batch["ref_stft"].to(device, non_blocking=True)
                clean_stft = batch["clean_stft"].to(device, non_blocking=True)
                clean_wav = batch["clean_wav"].to(device, non_blocking=True)
                delay_samp = batch["delay_samples"].to(device, non_blocking=True)

                enhanced, delay_dist, _ = raw_model_ref(mic_stft, ref_stft, return_delay=True)
                loss, _ = criterion(
                    enhanced, clean_stft, clean_wav,
                    delay_dist=delay_dist, delay_samples=delay_samp, dmax=cfg.dmax,
                )

                length = clean_wav.shape[-1]
                mic_wav = batch["mic_wav"].to(device, non_blocking=True)
                enh_wav = istft(enhanced, cfg.n_fft, cfg.hop_length, length=length)
                erle = compute_erle(mic_wav, enh_wav, clean_wav)

                val_loss_sum += loss.detach()
                val_erle_sum += erle.detach()
                val_n += 1

                if is_main and len(tb_samples) < 3:
                    si = 0
                    tb_samples.append({
                        "mic_stft": mic_stft[si:si+1],
                        "ref_stft": ref_stft[si:si+1],
                        "clean_stft": clean_stft[si:si+1],
                        "enhanced": enhanced[si:si+1],
                        "delay_dist": delay_dist[si:si+1] if delay_dist is not None else None,
                        "mic_wav": mic_wav[si:si+1],
                        "enh_wav": enh_wav[si:si+1],
                        "clean_wav": clean_wav[si:si+1],
                        "delay_samples": batch["delay_samples"][si],
                    })

        val_loss_sum = accelerator.gather(val_loss_sum.unsqueeze(0)).sum()
        val_erle_sum = accelerator.gather(val_erle_sum.unsqueeze(0)).sum()
        val_n_total = accelerator.gather(val_n.unsqueeze(0)).sum()
        avg_val_loss = (val_loss_sum / val_n_total).item()
        avg_val_erle = (val_erle_sum / val_n_total).item()

        if is_main:
            writer.add_scalar("val/loss", avg_val_loss, epoch)
            writer.add_scalar("val/erle_db", avg_val_erle, epoch)
            writer.add_scalar("train_epoch/loss", avg_train_loss, epoch)
            writer.add_scalar("train_epoch/delay_acc", avg_delay_acc, epoch)

            # Log audio samples, spectrograms, and delay heatmaps
            sr = cfg.sample_rate
            for i, s in enumerate(tb_samples):
                tag = f"audio_{i}" if i > 0 else "audio"
                writer.add_audio(f"{tag}/mic", s["mic_wav"][0].cpu().float(), epoch, sample_rate=sr)
                writer.add_audio(f"{tag}/enhanced", s["enh_wav"][0].cpu().float(), epoch, sample_rate=sr)
                writer.add_audio(f"{tag}/clean", s["clean_wav"][0].cpu().float(), epoch, sample_rate=sr)

                fig = plot_spectrogram_comparison(
                    s["mic_stft"], s["enhanced"], s["clean_stft"], sr, cfg.hop_length)
                writer.add_figure(f"spectrograms/comparison_{i}", fig, epoch)
                plt.close(fig)

                if s["delay_dist"] is not None:
                    fig = plot_delay_with_gt(
                        s["delay_dist"], s["delay_samples"], cfg.hop_length, cfg.dmax)
                    writer.add_figure(f"delay/distribution_{i}", fig, epoch)
                    plt.close(fig)

            # VRAM reporting
            if device.type == "cuda":
                alloc = torch.cuda.max_memory_allocated() / 1e9
                writer.add_scalar("system/vram_peak_gb", alloc, epoch)
                print(f"  Peak VRAM: {alloc:.1f} GB")

            print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, "
                  f"val_loss={avg_val_loss:.4f}, erle={avg_val_erle:.1f} dB, "
                  f"delay_acc={avg_delay_acc:.1%}, lr={optimizer.param_groups[0]['lr']:.2e}")

            # Checkpoint
            if (epoch + 1) % cfg.checkpoint_every == 0:
                ckpt_path = ckpt_dir / f"epoch_{epoch+1:04d}.pt"
                save_checkpoint(model, optimizer, scheduler, epoch + 1, avg_val_loss, ckpt_path, accelerator)
                ckpts = sorted(ckpt_dir.glob("epoch_*.pt"), key=lambda p: p.stat().st_mtime)
                while len(ckpts) > cfg.keep_checkpoints:
                    ckpts.pop(0).unlink()
                if hub_api:
                    try:
                        hub_api.upload_file(path_or_fileobj=str(ckpt_path),
                                            path_in_repo=f"checkpoints/{ckpt_path.name}",
                                            repo_id=hub_repo_id, run_as_future=True)
                    except Exception as e:
                        print(f"  Hub checkpoint upload failed: {e}")

            # Best model + early stopping
            if avg_val_loss < best_val_loss - 0.001:
                best_val_loss = avg_val_loss
                no_improve_count = 0
                best_path = ckpt_dir / "best.pt"
                save_checkpoint(model, optimizer, scheduler, epoch + 1, avg_val_loss, best_path, accelerator)
                print(f"  New best val loss: {best_val_loss:.4f}")
                if hub_api:
                    try:
                        hub_api.upload_file(path_or_fileobj=str(best_path),
                                            path_in_repo="checkpoints/best.pt",
                                            repo_id=hub_repo_id, run_as_future=True)
                    except Exception as e:
                        print(f"  Hub best upload failed: {e}")
            else:
                no_improve_count += 1

            # Hub logs
            if hub_api and (epoch + 1) % cfg.push_logs_every == 0:
                try:
                    hub_api.upload_folder(folder_path=str(log_dir), path_in_repo="logs",
                                          repo_id=hub_repo_id, run_as_future=True)
                except Exception as e:
                    print(f"  Hub log upload failed: {e}")

            # Early stopping
            if cfg.early_stop_patience > 0 and no_improve_count >= cfg.early_stop_patience:
                print(f"Early stopping: no improvement for {no_improve_count} epochs")
                break

    # Final uploads
    if is_main:
        if hub_api:
            try:
                hub_api.upload_folder(folder_path=str(log_dir), path_in_repo="logs",
                                      repo_id=hub_repo_id, run_as_future=True)
            except Exception as e:
                print(f"  Final hub log upload failed: {e}")
        if writer:
            writer.close()
        print("Training complete.")

    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    _maybe_launch_distributed()

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--no-hub", action="store_true")
    parser.add_argument("--hf-dataset", type=str, default=None,
                        help="HF dataset ID to download DNS5 from")
    args = parser.parse_args()

    cfg = Config()
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.epochs:
        cfg.epochs = args.epochs
    if args.lr:
        cfg.lr = args.lr
    if args.no_hub:
        cfg.push_to_hub = False
    if args.hf_dataset:
        cfg.hf_dataset_id = args.hf_dataset

    # Only rank 0 downloads data
    if int(os.environ.get("RANK", "0")) == 0:
        download_data(cfg)

    # Barrier: wait for rank 0 to finish downloading
    if "RANK" in os.environ:
        import torch.distributed as dist
        if not dist.is_initialized():
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            dist.init_process_group(device_id=torch.device("cuda", local_rank))
        dist.barrier()

    train(cfg)
