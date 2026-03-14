"""Loss functions for DeepVQE training.

Available components (enable via config weights):
1. Power-law compressed MSE  (plcmse_weight, default 1.0)
2. Magnitude L1              (mag_l1_weight, default 0.0)
3. Time-domain L1            (time_l1_weight, default 0.0)
4. SI-SDR                    (sisdr_weight, default 0.0)
5. Mask magnitude regularizer (mask_reg_weight, default 0.0)
6. SmoothL1 on waveform      (smooth_l1_weight, default 0.0)

Delay supervision and entropy penalty are in train.py (delay_weight, entropy_weight).
"""

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from src.stft import istft


def mask_mag_from_raw(d1):
    """Compute per-element mask magnitude from raw 27-channel decoder output.

    The 27 channels encode 3 cube-root-of-unity basis × 9 kernel elements.
    This decomposes into H_real and H_imag (each 9 elements), then computes
    the magnitude per kernel tap: sqrt(H_real^2 + H_imag^2).

    Args:
        d1: (B, 27, T, F) raw mask channels from decoder

    Returns:
        mask_mag: (B, 9, T, F) magnitude per kernel element
    """
    v_real = torch.tensor([1, -0.5, -0.5], device=d1.device, dtype=d1.dtype)
    v_imag = torch.tensor(
        [0, np.sqrt(3) / 2, -np.sqrt(3) / 2], device=d1.device, dtype=d1.dtype
    )
    m = rearrange(d1, "b (r c) t f -> b r c t f", r=3)
    H_real = torch.sum(v_real[None, :, None, None, None] * m, dim=1)  # (B,9,T,F)
    H_imag = torch.sum(v_imag[None, :, None, None, None] * m, dim=1)  # (B,9,T,F)
    return torch.sqrt(H_real ** 2 + H_imag ** 2 + 1e-12)


def si_sdr(pred, target):
    """Scale-invariant signal-to-distortion ratio (higher is better).

    Returns negative SI-SDR so it can be minimized.
    """
    # Remove mean
    pred = pred - pred.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    # s_target projection
    dot = torch.sum(pred * target, dim=-1, keepdim=True)
    s_target = dot * target / (torch.sum(target**2, dim=-1, keepdim=True) + 1e-8)
    e_noise = pred - s_target
    si_sdr_val = 10 * torch.log10(
        torch.sum(s_target**2, dim=-1) / (torch.sum(e_noise**2, dim=-1) + 1e-8) + 1e-8
    )
    return -si_sdr_val.mean()  # negative so we minimize


def mask_magnitude_regularizer(d1):
    """Penalize CCM mask magnitude deviating from 3×3 identity kernel.

    CCM uses causal ZeroPad2d([1,1,2,0]) with 3×3 kernel:
      m=0: t-2, m=1: t-1, m=2: t (current frame)
      n=0: f-1, n=1: f (current freq), n=2: f+1
    Current (t,f) = kernel index 2*3+1 = 7.  For identity convolution,
    tap 7 should have magnitude 1 and all other taps magnitude 0.

    Args:
        d1: (B, 27, T, F) raw 27-channel decoder output (before CCM)

    Returns:
        Scalar regularization loss.
    """
    mag = mask_mag_from_raw(d1)  # (B, 9, T, F)
    center = mag[:, 7]  # (B, T, F) — current (t, f)
    off_center = torch.cat([mag[:, :7], mag[:, 8:]], dim=1)  # (B, 8, T, F)
    return torch.mean((center - 1.0) ** 2) + torch.mean(off_center ** 2)


class DeepVQELoss(nn.Module):
    """Combined loss for DeepVQE AEC training."""

    def __init__(
        self,
        plcmse_weight=1.0,
        mag_l1_weight=0.5,
        time_l1_weight=0.5,
        sisdr_weight=0.5,
        smooth_l1_weight=0.0,
        smooth_l1_beta=1.0,
        power_law_c=0.5,
        n_fft=512,
        hop_length=256,
    ):
        super().__init__()
        self.plcmse_weight = plcmse_weight
        self.mag_l1_weight = mag_l1_weight
        self.time_l1_weight = time_l1_weight
        self.sisdr_weight = sisdr_weight
        self.smooth_l1_weight = smooth_l1_weight
        self.smooth_l1_beta = smooth_l1_beta
        self.c = power_law_c
        self.n_fft = n_fft
        self.hop_length = hop_length

    def _compress(self, x):
        """Power-law compression: |X|^c * exp(j*angle(X)), as real-valued."""
        mag = torch.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2 + 1e-12)
        mag_c = mag.pow(self.c)
        # Scale real/imag by |X|^(c-1)
        scale = mag_c / (mag + 1e-12)
        return x * scale.unsqueeze(-1)

    def _magnitude(self, x):
        """Compute magnitude from (B,F,T,2)."""
        return torch.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2 + 1e-12)

    def forward(self, pred_stft, target_stft, target_wav=None):
        """
        pred_stft: (B, F, T, 2) — predicted enhanced STFT
        target_stft: (B, F, T, 2) — target clean STFT
        target_wav: (B, N) — target clean waveform (optional, for time-domain loss)

        Returns:
            total_loss, dict of component losses
        """
        components = {}

        # 1. Power-law compressed MSE
        pred_c = self._compress(pred_stft)
        target_c = self._compress(target_stft)
        plcmse = torch.mean((pred_c - target_c) ** 2)
        components["plcmse"] = plcmse

        # 2. Magnitude L1
        pred_mag = self._magnitude(pred_stft)
        target_mag = self._magnitude(target_stft)
        mag_l1 = torch.mean(torch.abs(pred_mag - target_mag))
        components["mag_l1"] = mag_l1

        # 3. Time-domain losses (L1, SI-SDR, SmoothL1)
        _need_wav = (self.time_l1_weight > 0 or self.sisdr_weight > 0
                     or self.smooth_l1_weight > 0)
        zero = torch.tensor(0.0, device=pred_stft.device)
        if target_wav is not None and _need_wav:
            pred_wav = istft(
                pred_stft, self.n_fft, self.hop_length, length=target_wav.shape[-1]
            )
            time_l1 = torch.mean(torch.abs(pred_wav - target_wav)) if self.time_l1_weight > 0 else zero
            sisdr_loss = si_sdr(pred_wav, target_wav) if self.sisdr_weight > 0 else zero
            smooth_l1 = nn.functional.smooth_l1_loss(
                pred_wav, target_wav, beta=self.smooth_l1_beta
            ) if self.smooth_l1_weight > 0 else zero
        else:
            time_l1 = zero
            sisdr_loss = zero
            smooth_l1 = zero
        components["time_l1"] = time_l1
        components["sisdr"] = sisdr_loss
        components["smooth_l1"] = smooth_l1

        total = (
            self.plcmse_weight * plcmse
            + self.mag_l1_weight * mag_l1
            + self.time_l1_weight * time_l1
            + self.sisdr_weight * sisdr_loss
            + self.smooth_l1_weight * smooth_l1
        )
        components["total"] = total
        return total, components

    @classmethod
    def from_config(cls, cfg):
        return cls(
            plcmse_weight=cfg.loss.plcmse_weight,
            mag_l1_weight=cfg.loss.mag_l1_weight,
            time_l1_weight=cfg.loss.time_l1_weight,
            sisdr_weight=cfg.loss.sisdr_weight,
            smooth_l1_weight=cfg.loss.smooth_l1_weight,
            smooth_l1_beta=cfg.loss.smooth_l1_beta,
            power_law_c=cfg.loss.power_law_c,
            n_fft=cfg.audio.n_fft,
            hop_length=cfg.audio.hop_length,
        )
