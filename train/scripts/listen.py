"""Generate listenable WAV audio from a DeepVQE checkpoint.

Runs inference on validation samples and writes mic/enhanced/clean WAV files.

Usage:
    python scripts/listen.py --checkpoint checkpoints/best.pt
    python scripts/listen.py --checkpoint checkpoints/best.pt --dummy --num-samples 5
    python scripts/listen.py --checkpoint checkpoints/best.pt --sample 42
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from data.dataset import AECDataset, DummyAECDataset
from src.config import load_config
from src.metrics import evaluate_sample
from src.model import DeepVQEAEC
from src.stft import istft
from utils import load_checkpoint


def run(cfg, checkpoint_path, output_dir, dummy=False, num_samples=5, sample_idx=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = DeepVQEAEC.from_config(cfg).to(device)
    epoch, _ = load_checkpoint(checkpoint_path, model)
    model.eval()
    print(f"Loaded checkpoint from epoch {epoch}")

    # Dataset
    sr = cfg.audio.sample_rate
    target_len = int(cfg.training.clip_length_sec * sr)

    if dummy:
        ds = DummyAECDataset(
            length=max(num_samples, (sample_idx or 0) + 1),
            target_len=target_len,
            n_fft=cfg.audio.n_fft,
            hop_length=cfg.audio.hop_length,
            delay_range=tuple(int(x) for x in cfg.data.delay_range),
        )
    else:
        ds = AECDataset(cfg, split="val")

    # Determine which samples to process
    if sample_idx is not None:
        indices = [sample_idx]
    else:
        indices = list(range(min(num_samples, len(ds))))

    print(f"Processing {len(indices)} samples...")

    with torch.no_grad():
        for i in indices:
            sample = ds[i]
            mic_stft = sample["mic_stft"].unsqueeze(0).to(device)
            ref_stft = sample["ref_stft"].unsqueeze(0).to(device)

            enhanced = model(mic_stft, ref_stft)
            enh_wav = istft(
                enhanced.float(), cfg.audio.n_fft, cfg.audio.hop_length, length=target_len
            )

            mic_np = sample["mic_wav"].numpy()
            enh_np = enh_wav[0].cpu().numpy()
            clean_np = sample["clean_wav"].numpy()

            # Normalize to [-1, 1] for WAV output
            def _norm(x):
                peak = np.abs(x).max()
                return x / peak if peak > 0 else x

            sf.write(out_dir / f"mic_{i:04d}.wav", _norm(mic_np), sr, subtype="PCM_16")
            sf.write(out_dir / f"enhanced_{i:04d}.wav", _norm(enh_np), sr, subtype="PCM_16")
            sf.write(out_dir / f"clean_{i:04d}.wav", _norm(clean_np), sr, subtype="PCM_16")

            # Compute and print metrics
            metrics = evaluate_sample(mic_np, enh_np, clean_np, sr)
            parts = [f"Sample {i}: ERLE={metrics['erle_db']:.1f} dB"]
            parts.append(f"segSNR={metrics['seg_snr']:.1f} dB")
            if "pesq" in metrics:
                parts.append(f"PESQ={metrics['pesq']:.2f}")
            if "stoi" in metrics:
                parts.append(f"STOI={metrics['stoi']:.3f}")
            if "delay_samples" in sample.get("metadata", {}):
                parts.append(f"delay={sample['metadata']['delay_samples']}smp")
            print("  " + ", ".join(parts))

    print(f"\nWAV files saved to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate listenable WAV audio from checkpoint")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="../eval_output/audio")
    parser.add_argument("--dummy", action="store_true", help="Use DummyAECDataset")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--sample", type=int, default=None, help="Specific sample index")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run(cfg, args.checkpoint, args.output_dir,
        dummy=args.dummy, num_samples=args.num_samples, sample_idx=args.sample)
