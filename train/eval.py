"""DeepVQE AEC evaluation script.

Features:
- Compute ERLE, PESQ, STOI, segmental SNR on validation set
- Generate spectrogram comparisons (mic vs enhanced vs clean)
- Delay distribution heatmaps
- Audio sample output
- TensorBoard and/or console output
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

from data.dataset import AECDataset, DummyAECDataset
from src.config import load_config
from src.metrics import evaluate_sample
from src.model import DeepVQEAEC
from src.stft import istft
from utils import collate_fn, load_checkpoint


def plot_spectrograms(mic_wav, enh_wav, clean_wav, sr, title="", save_path=None):
    """Plot side-by-side spectrograms of mic, enhanced, and clean."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    for ax, wav, label in zip(
        axes,
        [mic_wav, enh_wav, clean_wav],
        ["Microphone", "Enhanced", "Clean"],
    ):
        ax.specgram(wav, NFFT=512, Fs=sr, noverlap=256, cmap="magma")
        ax.set_ylabel(f"{label}\nFreq (Hz)")
        ax.set_ylim(0, sr // 2)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


def plot_delay_heatmap(delay_dist, save_path=None):
    """Plot delay distribution heatmap.

    delay_dist: (T, dmax) numpy array
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.imshow(delay_dist.T, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Delay (frames)")
    ax.set_title("Delay Distribution")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


def evaluate(cfg, checkpoint_path, dummy=False, output_dir="eval_output",
             max_samples=None, save_val_audio=False, seed=42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "spectrograms").mkdir(exist_ok=True)
    (out_dir / "audio").mkdir(exist_ok=True)
    (out_dir / "delays").mkdir(exist_ok=True)

    if save_val_audio:
        (out_dir / "val_audio").mkdir(exist_ok=True)
        (out_dir / "pt_enhanced").mkdir(exist_ok=True)
        (out_dir / "scores").mkdir(exist_ok=True)

    # Model
    model = DeepVQEAEC.from_config(cfg).to(device)
    epoch, _ = load_checkpoint(checkpoint_path, model)
    if device.type == "cuda":
        model = torch.compile(model)
    model.eval()
    print(f"Loaded checkpoint from epoch {epoch}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Dataset
    if dummy:
        val_ds = DummyAECDataset(
            length=cfg.data.num_val,
            target_len=int(cfg.training.clip_length_sec * cfg.audio.sample_rate),
            n_fft=cfg.audio.n_fft,
            hop_length=cfg.audio.hop_length,
            delay_range=tuple(int(x) for x in cfg.data.delay_range),
        )
    else:
        val_ds = AECDataset(cfg, split="val")

    n_samples = len(val_ds)
    if max_samples:
        n_samples = min(n_samples, max_samples)

    sr = cfg.audio.sample_rate
    target_len = int(cfg.training.clip_length_sec * sr)

    all_metrics = []
    all_metadata = []
    use_amp = device.type == "cuda"

    with torch.no_grad():
        for i in range(n_samples):
            if save_val_audio:
                random.seed(seed + i)
                np.random.seed(seed + i)
            sample = val_ds[i]
            mic_stft = sample["mic_stft"].unsqueeze(0).to(device)
            ref_stft = sample["ref_stft"].unsqueeze(0).to(device)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                enhanced, delay_dist, mask_raw = model(mic_stft, ref_stft, return_delay=True)

            # Convert to waveforms (float32 for metrics)
            mic_wav = sample["mic_wav"].unsqueeze(0).to(device)
            enh_wav = istft(enhanced.float(), cfg.audio.n_fft, cfg.audio.hop_length, length=target_len)
            clean_wav = sample["clean_wav"].unsqueeze(0).to(device)

            mic_np = mic_wav[0].cpu().numpy()
            enh_np = enh_wav[0].cpu().numpy()
            clean_np = clean_wav[0].cpu().numpy()
            ref_np = sample["ref_wav"].numpy() if "ref_wav" in sample else None

            # Save val_audio + pt_enhanced for GGML eval pipeline
            if save_val_audio:
                L = min(len(mic_np), len(enh_np), len(clean_np))
                np.save(out_dir / "val_audio" / f"mic_{i:04d}.npy", mic_np[:L])
                np.save(out_dir / "val_audio" / f"ref_{i:04d}.npy", ref_np[:L] if ref_np is not None else mic_np[:L])
                np.save(out_dir / "val_audio" / f"clean_{i:04d}.npy", clean_np[:L])
                np.save(out_dir / "pt_enhanced" / f"enhanced_{i:04d}.npy", enh_np[:L])
                meta = sample.get("metadata", {})
                all_metadata.append({
                    "scenario": meta.get("scenario", "unknown"),
                    "delay_samples": int(sample.get("delay_samples", 0)),
                    "snr_db": float(meta.get("snr_db", 0)),
                    "ser_db": float(meta.get("ser_db", 0)),
                })

            # Compute metrics
            metrics = evaluate_sample(mic_np, enh_np, clean_np, sr, ref_wav=ref_np)
            metrics["sample_idx"] = i
            if "delay_samples" in sample.get("metadata", {}):
                metrics["true_delay"] = sample["metadata"]["delay_samples"]
            all_metrics.append(metrics)

            # Save visualizations for first N samples
            if i < cfg.eval.audio_samples:
                # Spectrograms
                plot_spectrograms(
                    mic_np, enh_np, clean_np, sr,
                    title=f"Sample {i} (ERLE={metrics['erle_db']:.1f} dB)",
                    save_path=out_dir / "spectrograms" / f"sample_{i:04d}.png",
                )

                # Delay heatmap
                if delay_dist is not None:
                    plot_delay_heatmap(
                        delay_dist[0].cpu().numpy(),
                        save_path=out_dir / "delays" / f"delay_{i:04d}.png",
                    )

                # Save audio as .npy (for programmatic access) and .wav (for listening)
                np.save(out_dir / "audio" / f"mic_{i:04d}.npy", mic_np)
                np.save(out_dir / "audio" / f"enhanced_{i:04d}.npy", enh_np)
                np.save(out_dir / "audio" / f"clean_{i:04d}.npy", clean_np)
                sf.write(out_dir / "audio" / f"mic_{i:04d}.wav", mic_np, sr, subtype="PCM_16")
                sf.write(out_dir / "audio" / f"enhanced_{i:04d}.wav", enh_np, sr, subtype="PCM_16")
                sf.write(out_dir / "audio" / f"clean_{i:04d}.wav", clean_np, sr, subtype="PCM_16")

            if (i + 1) % 10 == 0 or i == n_samples - 1:
                print(f"  Evaluated {i+1}/{n_samples}")

    # Aggregate metrics
    erle_values = [m["erle_db"] for m in all_metrics]
    seg_snr_values = [m["seg_snr"] for m in all_metrics]

    print(f"\n{'='*50}")
    print(f"Evaluation Results ({n_samples} samples)")
    print(f"{'='*50}")
    print(f"ERLE:        {np.mean(erle_values):.2f} dB (std={np.std(erle_values):.2f})")
    print(f"Seg SNR:     {np.mean(seg_snr_values):.2f} dB (std={np.std(seg_snr_values):.2f})")

    pesq_values = [m["pesq"] for m in all_metrics if "pesq" in m]
    if pesq_values:
        print(f"PESQ:        {np.mean(pesq_values):.3f} (std={np.std(pesq_values):.3f})")

    stoi_values = [m["stoi"] for m in all_metrics if "stoi" in m]
    if stoi_values:
        print(f"STOI:        {np.mean(stoi_values):.3f} (std={np.std(stoi_values):.3f})")

    dnsmos_ovrl = [m["dnsmos_ovrl"] for m in all_metrics if "dnsmos_ovrl" in m]
    if dnsmos_ovrl:
        print(f"DNSMOS OVRL: {np.mean(dnsmos_ovrl):.3f} (std={np.std(dnsmos_ovrl):.3f})")
        dnsmos_sig = [m["dnsmos_sig"] for m in all_metrics if "dnsmos_sig" in m]
        dnsmos_bak = [m["dnsmos_bak"] for m in all_metrics if "dnsmos_bak" in m]
        print(f"DNSMOS SIG:  {np.mean(dnsmos_sig):.3f}  BAK: {np.mean(dnsmos_bak):.3f}")

    echo_mos = [m["echo_mos"] for m in all_metrics if "echo_mos" in m]
    deg_mos = [m["deg_mos"] for m in all_metrics if "deg_mos" in m]
    if echo_mos:
        print(f"AECMOS echo: {np.mean(echo_mos):.3f} (std={np.std(echo_mos):.3f})")
        print(f"AECMOS deg:  {np.mean(deg_mos):.3f} (std={np.std(deg_mos):.3f})")

    print(f"\nOutput saved to: {out_dir}")

    # Save summary
    summary = {
        "epoch": epoch,
        "n_samples": n_samples,
        "erle_mean": float(np.mean(erle_values)),
        "erle_std": float(np.std(erle_values)),
        "seg_snr_mean": float(np.mean(seg_snr_values)),
        "seg_snr_std": float(np.std(seg_snr_values)),
    }
    if pesq_values:
        summary["pesq_mean"] = float(np.mean(pesq_values))
    if stoi_values:
        summary["stoi_mean"] = float(np.mean(stoi_values))
    if dnsmos_ovrl:
        summary["dnsmos_ovrl_mean"] = float(np.mean(dnsmos_ovrl))
    if echo_mos:
        summary["echo_mos_mean"] = float(np.mean(echo_mos))
        summary["deg_mos_mean"] = float(np.mean(deg_mos))

    np.save(out_dir / "summary.npy", summary)

    # Save val_audio metadata and per-sample scores as JSON
    if save_val_audio and all_metadata:
        np.save(out_dir / "val_audio" / "metadata.npy", all_metadata)

        # Build per-sample scores for JSON
        scores_data = {
            "label": "pytorch",
            "n_samples": n_samples,
            "per_sample": [],
            "aggregate": {},
        }
        for m in all_metrics:
            entry = {"idx": m["sample_idx"]}
            for key in ["erle_db", "seg_snr", "pesq", "stoi",
                        "dnsmos_ovrl", "dnsmos_sig", "dnsmos_bak",
                        "echo_mos", "deg_mos"]:
                if key in m:
                    entry[key] = float(m[key])
            # Attach scenario from metadata
            idx = m["sample_idx"]
            if idx < len(all_metadata):
                entry["scenario"] = all_metadata[idx]["scenario"]
            scores_data["per_sample"].append(entry)

        # Aggregate by scenario
        from collections import defaultdict
        by_scenario = defaultdict(lambda: defaultdict(list))
        for entry in scores_data["per_sample"]:
            sc = entry.get("scenario", "unknown")
            for k, v in entry.items():
                if isinstance(v, float):
                    by_scenario[sc][k].append(v)
        for sc, metrics in by_scenario.items():
            scores_data["aggregate"][sc] = {
                k: float(np.mean(v)) for k, v in metrics.items()
            }

        scores_path = out_dir / "scores" / "pt_scores.json"
        with open(scores_path, "w") as f:
            json.dump(scores_data, f, indent=2)
        print(f"Saved scores to {scores_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DeepVQE AEC")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    parser.add_argument("--dummy", action="store_true", help="Use dummy dataset")
    parser.add_argument("--output-dir", default="../eval_output", help="Output directory")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to evaluate")
    parser.add_argument("--save-val-audio", action="store_true",
                        help="Save val_audio + pt_enhanced .npy for GGML eval pipeline")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducible samples")
    args = parser.parse_args()

    cfg = load_config(args.config)
    evaluate(cfg, args.checkpoint, dummy=args.dummy,
             output_dir=args.output_dir, max_samples=args.max_samples,
             save_val_audio=args.save_val_audio, seed=args.seed)
