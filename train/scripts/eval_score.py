#!/usr/bin/env python3
"""Score pre-computed enhanced audio and optionally compare with other scores.

Usage:
    python scripts/eval_score.py \
        --val-dir ../eval_output/val_audio \
        --enh-dir ../eval_output/ggml_enhanced \
        --label ggml_f32 \
        --output ../eval_output/scores/ggml_scores.json \
        [--compare ../eval_output/scores/pt_scores.json] \
        [--max-samples 1000]
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics import evaluate_sample


def load_metadata(val_dir):
    meta_path = Path(val_dir) / "metadata.npy"
    if meta_path.exists():
        return np.load(meta_path, allow_pickle=True)
    return None


def score_enhanced(val_dir, enh_dir, label, max_samples):
    val_dir = Path(val_dir)
    enh_dir = Path(enh_dir)
    metadata = load_metadata(val_dir)

    scores = {"label": label, "per_sample": [], "aggregate": {}}
    n = 0

    for idx in range(max_samples):
        mic_path = val_dir / f"mic_{idx:04d}.npy"
        ref_path = val_dir / f"ref_{idx:04d}.npy"
        clean_path = val_dir / f"clean_{idx:04d}.npy"
        enh_path = enh_dir / f"enhanced_{idx:04d}.npy"

        if not enh_path.exists() or not mic_path.exists():
            if n == 0:
                continue
            break

        mic = np.load(mic_path)
        ref = np.load(ref_path)
        clean = np.load(clean_path)
        enh = np.load(enh_path)

        L = min(len(mic), len(ref), len(clean), len(enh))
        mic, ref, clean, enh = mic[:L], ref[:L], clean[:L], enh[:L]

        # Determine scenario
        if metadata is not None and idx < len(metadata):
            sc = str(metadata[idx].get("scenario", "unknown") if isinstance(metadata[idx], dict)
                     else metadata[idx]["scenario"])
        else:
            sc = "single_talk_farend" if np.sqrt(np.mean(clean**2)) < 1e-5 else "double_talk"

        metrics = evaluate_sample(mic, enh, clean, ref_wav=ref)
        entry = {"idx": idx, "scenario": sc}
        for key in ["erle_db", "seg_snr", "pesq", "stoi",
                    "dnsmos_ovrl", "dnsmos_sig", "dnsmos_bak",
                    "echo_mos", "deg_mos"]:
            if key in metrics:
                entry[key] = float(metrics[key])
        scores["per_sample"].append(entry)

        n += 1
        if n % 20 == 0:
            print(f"  [{n}] scored...")

    scores["n_samples"] = n

    # Aggregate by scenario
    by_scenario = defaultdict(lambda: defaultdict(list))
    for entry in scores["per_sample"]:
        sc = entry.get("scenario", "unknown")
        for k, v in entry.items():
            if isinstance(v, float):
                by_scenario[sc][k].append(v)
    for sc, metrics in by_scenario.items():
        scores["aggregate"][sc] = {k: float(np.mean(v)) for k, v in metrics.items()}

    return scores


METRIC_NAMES = [
    ("erle_db", "ERLE (dB)", "+5.1f"),
    ("pesq", "PESQ", "5.2f"),
    ("stoi", "STOI", "5.3f"),
    ("dnsmos_ovrl", "DNSMOS", "5.2f"),
    ("echo_mos", "AEC echo", "5.2f"),
    ("deg_mos", "AEC deg", "5.2f"),
]


def print_comparison(all_scores):
    """Print side-by-side comparison table."""
    labels = [s["label"] for s in all_scores]
    header = f"{'':>20s}"
    for lbl in labels:
        header += f" {lbl:>12s}"
    print(header)
    print("-" * len(header))

    # Collect all scenarios
    scenarios = []
    for s in all_scores:
        for sc in s["aggregate"]:
            if sc not in scenarios:
                scenarios.append(sc)

    for sc in scenarios:
        n_samples = 0
        for s in all_scores:
            if sc in s["aggregate"]:
                n_vals = len([e for e in s["per_sample"] if e.get("scenario") == sc])
                n_samples = max(n_samples, n_vals)
        print(f"\n  {sc} (n={n_samples})")

        for key, name, fmt in METRIC_NAMES:
            row = f"    {name:>16s}"
            for s in all_scores:
                agg = s["aggregate"].get(sc, {})
                if key in agg:
                    row += f" {agg[key]:{fmt}}"
                else:
                    row += f" {'N/A':>12s}"
            print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--enh-dir", required=True)
    parser.add_argument("--label", default="unknown")
    parser.add_argument("--output", required=True, help="JSON output path")
    parser.add_argument("--compare", nargs="*", default=[],
                        help="Existing score JSON files to include in comparison")
    parser.add_argument("--max-samples", type=int, default=10000)
    args = parser.parse_args()

    print(f"Scoring {args.label} from {args.enh_dir}...")
    scores = score_enhanced(args.val_dir, args.enh_dir, args.label, args.max_samples)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"Saved {scores['n_samples']} scores to {args.output}")

    # Print comparison if other scores provided
    all_scores = []
    for path in args.compare:
        with open(path) as f:
            all_scores.append(json.load(f))
    all_scores.append(scores)

    print(f"\nComparison ({scores['n_samples']} samples):")
    print_comparison(all_scores)


if __name__ == "__main__":
    main()
