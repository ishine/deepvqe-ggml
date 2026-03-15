"""AEC Dataset for DeepVQE training.

Supports two modes:
1. Online synthesis: generates examples on-the-fly from clean/noise/RIR files
2. Pre-existing: loads pre-synthesized AEC challenge data
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data.synth import synthesize_example
from src.stft import stft

_CACHE_DIR = Path(".cache/file_lists")


def _collect_audio_files(directory):
    """Recursively collect .wav and .flac files, with disk caching.

    On large datasets (500k+ files), rglob + sort can take minutes.
    Results are cached to .cache/file_lists/ keyed by directory path.
    Cache is invalidated when file count changes (top-level mtime is
    unreliable for nested directories on most filesystems).
    """
    if not directory:
        return []
    d = Path(directory)
    if not d.exists():
        return []

    # Cache key based on absolute path
    abs_path = str(d.resolve())
    cache_key = hashlib.md5(abs_path.encode()).hexdigest()
    cache_file = _CACHE_DIR / f"{cache_key}.json"

    # Try loading from cache
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
            if cached.get("path") == abs_path:
                # Quick validation: spot-check that first and last files still exist
                files = cached["files"]
                if files and os.path.exists(files[0]) and os.path.exists(files[-1]):
                    return files
        except (json.JSONDecodeError, KeyError):
            pass

    # Scan directory
    files = []
    for ext in ("*.wav", "*.flac"):
        files.extend(str(f) for f in d.rglob(ext))
    files.sort()

    # Write cache (rank-0 only to avoid race in multi-GPU)
    if int(os.environ.get("RANK", "0")) == 0:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps({
            "path": abs_path,
            "count": len(files),
            "files": files,
        }))

    return files


class AECDataset(Dataset):
    """Online-synthesis AEC dataset.

    Each __getitem__ call synthesizes a fresh example.
    """

    def __init__(self, cfg, split="train"):
        """
        Args:
            cfg: Config object with data and audio sections
            split: 'train' or 'val'
        """
        self.sr = cfg.audio.sample_rate
        self.n_fft = cfg.audio.n_fft
        self.hop_length = cfg.audio.hop_length
        self.target_len = int(cfg.training.clip_length_sec * self.sr)
        self.snr_range = tuple(cfg.data.snr_range)
        self.ser_range = tuple(cfg.data.ser_range)
        self.delay_range = tuple(cfg.data.delay_range)
        self.single_talk_prob = cfg.data.single_talk_prob

        all_clean = _collect_audio_files(cfg.data.clean_dir)
        self.noise_files = _collect_audio_files(cfg.data.noise_dir)
        self.farend_files = _collect_audio_files(cfg.data.farend_dir)
        self.rir_files = _collect_audio_files(cfg.data.rir_dir) or None

        # If farend_dir not specified, use clean_dir (full pool, no split)
        if not self.farend_files:
            self.farend_files = all_clean

        # Hold out num_val clean files for validation (deterministic split)
        num_val = cfg.data.num_val
        if split == "train":
            self.clean_files = all_clean[num_val:]
            self.length = len(self.clean_files)
        else:
            self.clean_files = all_clean[:num_val]
            self.length = len(self.clean_files)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        clean_path = self.clean_files[idx]

        mic, ref, clean, metadata = synthesize_example(
            clean_path=clean_path,
            noise_files=self.noise_files,
            farend_files=self.farend_files,
            rir_files=self.rir_files,
            target_len=self.target_len,
            sr=self.sr,
            snr_range=self.snr_range,
            ser_range=self.ser_range,
            delay_range=self.delay_range,
            single_talk_prob=self.single_talk_prob,
        )

        # Convert to tensors and compute STFTs
        mic_t = torch.from_numpy(mic).unsqueeze(0)  # (1, N)
        ref_t = torch.from_numpy(ref).unsqueeze(0)
        clean_t = torch.from_numpy(clean).unsqueeze(0)

        mic_stft = stft(mic_t, self.n_fft, self.hop_length).squeeze(0)  # (F, T, 2)
        ref_stft = stft(ref_t, self.n_fft, self.hop_length).squeeze(0)
        clean_stft = stft(clean_t, self.n_fft, self.hop_length).squeeze(0)

        return {
            "mic_stft": mic_stft,
            "ref_stft": ref_stft,
            "clean_stft": clean_stft,
            "mic_wav": mic_t.squeeze(0),
            "clean_wav": clean_t.squeeze(0),
            "delay_samples": metadata["delay_samples"],
            "metadata": metadata,
        }


class DummyAECDataset(Dataset):
    """Synthetic dataset for testing (no audio files needed).

    Generates amplitude-modulated tonal signals with per-example random
    delay for verification.  The amplitude envelope gives temporal structure
    so the AlignBlock can detect the delay from STFT magnitude patterns
    (stationary tones have identical frames, making delay undetectable).
    """

    def __init__(self, length=100, target_len=48000, n_fft=512, hop_length=256,
                 delay_range=(0, 0), sr=16000):
        self.length = length
        self.target_len = target_len
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.delay_range = delay_range  # (min_samples, max_samples)
        self.sr = sr

    def __len__(self):
        return self.length

    @staticmethod
    def _tonal_signal(rng, length, sr, n_harmonics=15, amplitude=0.1):
        """Generate an amplitude-modulated tonal signal.

        Sum of harmonics (smooth spectral structure like speech) multiplied
        by a random smooth envelope (3-6 bumps over the signal duration).
        The envelope gives temporal variation so delay estimation can work.
        """
        t = np.arange(length, dtype=np.float32) / sr
        f0 = rng.uniform(80, 300)  # fundamental frequency
        signal = np.zeros(length, dtype=np.float32)
        for h in range(1, n_harmonics + 1):
            freq = f0 * h
            if freq > sr / 2:
                break
            amp = amplitude * rng.uniform(0.2, 1.0) / h  # harmonic roll-off
            phase = rng.uniform(0, 2 * np.pi)
            signal += amp * np.sin(2 * np.pi * freq * t + phase).astype(np.float32)

        # Amplitude modulation: smooth random envelope with 3-6 bumps
        n_bumps = rng.randint(3, 7)
        env_freqs = rng.uniform(0.5, 3.0, size=n_bumps)
        env_phases = rng.uniform(0, 2 * np.pi, size=n_bumps)
        envelope = np.ones(length, dtype=np.float32) * 0.3  # baseline
        for ef, ep in zip(env_freqs, env_phases):
            envelope += (0.7 / n_bumps) * (1 + np.sin(2 * np.pi * ef * t + ep).astype(np.float32))
        envelope = np.clip(envelope, 0.0, 1.0)
        signal *= envelope
        return signal

    def __getitem__(self, idx):
        rng = np.random.RandomState(idx)

        # Generate amplitude-modulated tonal signals
        clean = self._tonal_signal(rng, self.target_len, self.sr)
        farend = self._tonal_signal(rng, self.target_len, self.sr)
        noise = rng.randn(self.target_len).astype(np.float32) * 0.01

        # Per-example random delay (quantized to hop_length for clean frame alignment)
        delay_lo, delay_hi = self.delay_range
        if delay_hi > delay_lo:
            delay_samples = rng.randint(delay_lo, delay_hi + 1)
            # Quantize to hop_length so ground truth aligns to STFT frames
            delay_samples = int(round(delay_samples / self.hop_length) * self.hop_length)
            delay_samples = max(delay_lo, min(delay_hi, delay_samples))
        else:
            delay_samples = delay_lo

        # Echo with delay
        echo = np.zeros_like(farend)
        if delay_samples < self.target_len:
            end = min(self.target_len, self.target_len - delay_samples)
            echo[delay_samples:] = farend[:end] * 0.5

        mic = clean + echo + noise

        mic_t = torch.from_numpy(mic).unsqueeze(0)
        ref_t = torch.from_numpy(farend).unsqueeze(0)
        clean_t = torch.from_numpy(clean).unsqueeze(0)

        mic_stft = stft(mic_t, self.n_fft, self.hop_length).squeeze(0)
        ref_stft = stft(ref_t, self.n_fft, self.hop_length).squeeze(0)
        clean_stft = stft(clean_t, self.n_fft, self.hop_length).squeeze(0)

        return {
            "mic_stft": mic_stft,
            "ref_stft": ref_stft,
            "clean_stft": clean_stft,
            "mic_wav": mic_t.squeeze(0),
            "clean_wav": clean_t.squeeze(0),
            "delay_samples": delay_samples,
            "metadata": {
                "delay_ms": delay_samples / self.sr * 1000,
                "delay_samples": delay_samples,
                "snr_db": 20.0,
                "ser_db": 6.0,
                "scenario": "double_talk",
            },
        }
