#!/usr/bin/env python3
"""Generate STFT and model reference data for C++ verification.

Saves:
  - Synthetic sine STFT reference (stft_input.npy, stft_ref.npy)
  - Real audio model reference (val_mic.npy, val_ref.npy, val_enhanced_py.npy)
    from running PyTorch model on val_audio samples

Usage:
    python scripts/gen_stft_ref.py --output-dir ../ggml/test_data
    python scripts/gen_stft_ref.py --output-dir ../ggml/test_data \
        --checkpoint <path> --val-dir ../eval_output/val_audio
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def gen_stft_ref(out):
    """Generate synthetic STFT reference."""
    n_fft = 512

    # 1 second of 440Hz sine
    sr = 16000
    t = np.arange(sr, dtype=np.float32)
    x = (0.5 * np.sin(2 * np.pi * 440 * t / sr)).astype(np.float32)

    # sqrt-Hann window matching C++ (with 1e-12 epsilon)
    hann = 0.5 * (1.0 - np.cos(2 * np.pi * np.arange(n_fft) / n_fft))
    window = torch.from_numpy(np.sqrt(hann + 1e-12).astype(np.float32))

    stft = torch.stft(torch.from_numpy(x), n_fft, hop_length=256,
                       window=window, return_complex=False)

    np.save(out / "stft_input.npy", x)
    np.save(out / "stft_ref.npy", stft.numpy())
    print(f"  stft_input.npy: {x.shape}")
    print(f"  stft_ref.npy:   {stft.shape}")


def gen_model_ref(out, checkpoint_path, val_dir, n_samples=3):
    """Generate model enhanced PCM reference from real val_audio."""
    from src.config import load_config
    from src.model import DeepVQEAEC
    from src.stft import stft, istft
    from utils import load_checkpoint

    cfg = load_config("configs/default.yaml")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = DeepVQEAEC.from_config(cfg).to(device)
    epoch, _ = load_checkpoint(checkpoint_path, model)
    model.eval()
    # Run on CPU for exact f32 match with GGML (CUDA float32 has different
    # accumulation order in cuBLAS/cuDNN, causing ~0.05 divergence in STFT domain)
    model = model.cpu()
    device = torch.device("cpu")
    print(f"  Loaded epoch {epoch}, running on CPU for GGML comparison")

    val_dir = Path(val_dir)
    sr = cfg.audio.sample_rate
    target_len = int(cfg.training.clip_length_sec * sr)

    with torch.no_grad():
        for i in range(n_samples):
            mic_path = val_dir / f"mic_{i:04d}.npy"
            ref_path = val_dir / f"ref_{i:04d}.npy"
            if not mic_path.exists():
                break

            mic_pcm = np.load(mic_path)
            ref_pcm = np.load(ref_path)

            # STFT (same as training pipeline)
            mic_t = torch.from_numpy(mic_pcm).unsqueeze(0).to(device)
            ref_t = torch.from_numpy(ref_pcm).unsqueeze(0).to(device)
            mic_stft_t = stft(mic_t, cfg.audio.n_fft, cfg.audio.hop_length)
            ref_stft_t = stft(ref_t, cfg.audio.n_fft, cfg.audio.hop_length)

            # Save Python STFT for C++ comparison (bypass C++ STFT)
            # Shape: (1, F, T, 2) matching forward_graph() input format
            np.save(out / f"val_mic_stft_{i:04d}.npy",
                    mic_stft_t.cpu().numpy())
            np.save(out / f"val_ref_stft_{i:04d}.npy",
                    ref_stft_t.cpu().numpy())

            # Model forward (f32 — no AMP, matches GGML precision)
            enhanced_stft, _, _ = model(mic_stft_t, ref_stft_t, return_delay=True)

            # Save enhanced STFT for layer comparison
            np.save(out / f"val_enhanced_stft_py_{i:04d}.npy",
                    enhanced_stft.cpu().numpy())

            # iSTFT
            enh_pcm = istft(enhanced_stft.float(), cfg.audio.n_fft,
                            cfg.audio.hop_length, length=target_len)
            enh_np = enh_pcm[0].cpu().numpy()

            L = min(len(mic_pcm), len(enh_np))
            np.save(out / f"val_enhanced_py_{i:04d}.npy", enh_np[:L])
            print(f"  [{i}] pcm_rms=%.4f stft_shape=%s" %
                  (np.sqrt(np.mean(enh_np[:L]**2)),
                   list(enhanced_stft.shape)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint for model reference (optional)")
    parser.add_argument("--val-dir", default=None,
                        help="val_audio dir with mic/ref .npy (optional)")
    parser.add_argument("--n-samples", type=int, default=3)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Generating STFT reference...")
    gen_stft_ref(out)

    if args.checkpoint and args.val_dir:
        print("Generating model reference...")
        gen_model_ref(out, args.checkpoint, args.val_dir, args.n_samples)

    print(f"Done. Saved to {out}/")


if __name__ == "__main__":
    main()
