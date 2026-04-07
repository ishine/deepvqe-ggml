#!/usr/bin/env python3
"""Dump validation audio (mic, ref, clean) as .npy for C++ testing.

Also computes Python ERLE/segSNR on model-enhanced output for comparison.

Usage:
    python scripts/dump_val_audio.py --checkpoint <ckpt> --output eval_output/val_audio
    python scripts/dump_val_audio.py --checkpoint <ckpt> --output eval_output/val_audio --max-samples 100
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.model import DeepVQEAEC
from src.stft import istft
from src.metrics import erle, segmental_snr
from data.dataset import AECDataset
from utils import load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="eval_output/val_audio")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for reproducible sample generation")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepVQEAEC.from_config(cfg).to(device)
    epoch, _ = load_checkpoint(args.checkpoint, model)
    if device.type == "cuda":
        model = torch.compile(model)
    model.eval()
    print(f"Loaded epoch {epoch}, device={device}")

    val_ds = AECDataset(cfg, split="val")
    n = len(val_ds) if args.max_samples is None else min(len(val_ds), args.max_samples)
    sr = cfg.audio.sample_rate
    target_len = int(cfg.training.clip_length_sec * sr)
    print(f"Dumping {n} validation samples to {out_dir}/")

    all_erle, all_snr, all_meta = [], [], []

    with torch.no_grad():
        for i in range(n):
            # Seed per-sample so each index always produces the same mix
            random.seed(args.seed + i)
            np.random.seed(args.seed + i)
            sample = val_ds[i]

            # Save raw audio + metadata
            mic_np = sample["mic_wav"].numpy()
            ref_np = sample["ref_wav"].numpy()
            clean_np = sample["clean_wav"].numpy()
            metadata = sample["metadata"]
            np.save(out_dir / f"mic_{i:04d}.npy", mic_np)
            np.save(out_dir / f"ref_{i:04d}.npy", ref_np)
            np.save(out_dir / f"clean_{i:04d}.npy", clean_np)
            all_meta.append({
                "scenario": metadata["scenario"],
                "delay_samples": int(sample["delay_samples"]),
                "snr_db": float(metadata.get("snr_db", 0)),
                "ser_db": float(metadata.get("ser_db", 0)),
            })

            # Run model for Python baseline metrics
            mic_stft = sample["mic_stft"].unsqueeze(0).to(device)
            ref_stft = sample["ref_stft"].unsqueeze(0).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                enhanced, _, _ = model(mic_stft, ref_stft, return_delay=True)
            enh_wav = istft(enhanced.float(), cfg.audio.n_fft,
                            cfg.audio.hop_length, length=target_len)
            enh_np = enh_wav[0].cpu().numpy()

            L = min(len(mic_np), len(enh_np), len(clean_np))
            e, _ = erle(mic_np[:L], enh_np[:L], clean_np[:L])
            s = segmental_snr(clean_np[:L], enh_np[:L])
            all_erle.append(e)
            all_snr.append(s)

            if (i + 1) % 100 == 0 or i == n - 1:
                # Per-scenario breakdown
                st = [all_erle[j] for j in range(len(all_erle))
                      if all_meta[j]["scenario"] == "single_talk"]
                dt = [all_erle[j] for j in range(len(all_erle))
                      if all_meta[j]["scenario"] == "double_talk"]
                st_str = f"ST={np.mean(st):.1f}" if st else "ST=N/A"
                dt_str = f"DT={np.mean(dt):.1f}" if dt else "DT=N/A"
                print(f"  [{i+1}/{n}] ERLE={np.mean(all_erle):.2f} dB  "
                      f"segSNR={np.mean(all_snr):.2f} dB  ({st_str} {dt_str})")

    # Per-scenario breakdown
    scenarios = {}
    for i, m in enumerate(all_meta):
        sc = m["scenario"]
        if sc not in scenarios:
            scenarios[sc] = {"erle": [], "snr": []}
        scenarios[sc]["erle"].append(all_erle[i])
        scenarios[sc]["snr"].append(all_snr[i])

    # Save summary
    summary = {
        "n_samples": n,
        "epoch": epoch,
        "seed": args.seed,
        "erle_mean": float(np.mean(all_erle)),
        "erle_std": float(np.std(all_erle)),
        "seg_snr_mean": float(np.mean(all_snr)),
        "seg_snr_std": float(np.std(all_snr)),
    }
    for sc, vals in scenarios.items():
        summary[f"erle_{sc}"] = float(np.mean(vals["erle"]))
        summary[f"snr_{sc}"] = float(np.mean(vals["snr"]))
        summary[f"n_{sc}"] = len(vals["erle"])

    np.save(out_dir / "python_baseline.npy", summary)
    np.save(out_dir / "metadata.npy", all_meta)
    print(f"\nPython baseline ({n} samples):")
    print(f"  ERLE:   {summary['erle_mean']:.2f} +/- {summary['erle_std']:.2f} dB")
    print(f"  segSNR: {summary['seg_snr_mean']:.2f} +/- {summary['seg_snr_std']:.2f} dB")
    for sc, vals in scenarios.items():
        print(f"  {sc}: n={len(vals['erle'])}  "
              f"ERLE={np.mean(vals['erle']):.2f}  segSNR={np.mean(vals['snr']):.2f}")


if __name__ == "__main__":
    main()
