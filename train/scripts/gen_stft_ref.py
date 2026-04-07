#!/usr/bin/env python3
"""Generate STFT reference data for C++ test_stft verification.

Saves a test signal and its torch.stft output as .npy files.

Usage:
    python scripts/gen_stft_ref.py --output-dir ../ggml/test_data
"""
import argparse
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    n_fft = 512
    sr = 16000

    # 1 second of 440Hz sine
    t = np.arange(sr, dtype=np.float32)
    x = (0.5 * np.sin(2 * np.pi * 440 * t / sr)).astype(np.float32)

    # sqrt-Hann window matching C++ (with 1e-12 epsilon)
    hann = 0.5 * (1.0 - np.cos(2 * np.pi * np.arange(n_fft) / n_fft))
    window = torch.from_numpy(np.sqrt(hann + 1e-12).astype(np.float32))

    # torch.stft: center=True (default), pad_mode="reflect" (default)
    stft = torch.stft(torch.from_numpy(x), n_fft, hop_length=256,
                       window=window, return_complex=False)
    # stft shape: (F, T, 2) where F = n_fft/2 + 1

    np.save(out / "stft_input.npy", x)
    np.save(out / "stft_ref.npy", stft.numpy())

    print(f"Saved to {out}/")
    print(f"  stft_input.npy: {x.shape}")
    print(f"  stft_ref.npy:   {stft.shape}")


if __name__ == "__main__":
    main()
