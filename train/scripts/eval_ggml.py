#!/usr/bin/env python3
"""Evaluate GGML-enhanced audio quality using AECMOS, DNSMOS, PESQ, STOI.

Loads mic/ref/clean/enhanced .npy files from eval_graph output and computes
quality metrics. Separates single-talk vs double-talk using metadata.

Usage:
    python scripts/eval_ggml.py \
        --val-dir eval_output/val_audio \
        --enh-dir ../ggml_eval \
        [--max-samples 200]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics import (
    erle,
    compute_pesq,
    compute_stoi,
    compute_dnsmos,
    compute_aecmos,
)


def load_metadata(val_dir):
    """Load per-sample metadata if available."""
    meta_path = Path(val_dir) / "metadata.npy"
    if meta_path.exists():
        return np.load(meta_path, allow_pickle=True)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-dir", required=True,
                        help="Directory with mic_XXXX.npy, ref_XXXX.npy, clean_XXXX.npy")
    parser.add_argument("--enh-dir", required=True,
                        help="Directory with enhanced_XXXX.npy from eval_graph")
    parser.add_argument("--max-samples", type=int, default=10000)
    args = parser.parse_args()

    val_dir = Path(args.val_dir)
    enh_dir = Path(args.enh_dir)
    metadata = load_metadata(val_dir)

    results = {
        "double_talk": {"erle": [], "pesq": [], "stoi": [],
                        "dnsmos_mic": [], "dnsmos_enh": [],
                        "aecmos_echo": [], "aecmos_deg": []},
        "single_talk_farend": {"erle": [],
                               "dnsmos_mic": [], "dnsmos_enh": [],
                               "aecmos_echo": [], "aecmos_deg": []},
    }

    n = 0
    for idx in range(args.max_samples):
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
            sc = metadata[idx]["scenario"]
        else:
            # Heuristic: if clean RMS < 1e-5, it's single-talk far-end
            sc = "single_talk_farend" if np.sqrt(np.mean(clean**2)) < 1e-5 else "double_talk"

        r = results[sc]

        # ERLE (for reference, even though it's misleading with reverb)
        e, _ = erle(mic, enh, clean)
        r["erle"].append(e)

        # PESQ/STOI for double-talk only
        if sc == "double_talk":
            p = compute_pesq(clean, enh)
            s = compute_stoi(clean, enh)
            if p is not None:
                r["pesq"].append(p)
            if s is not None:
                r["stoi"].append(s)

        # DNSMOS (no-reference quality)
        dnsmos_mic = compute_dnsmos(mic)
        dnsmos_enh = compute_dnsmos(enh)
        if dnsmos_mic:
            r["dnsmos_mic"].append(dnsmos_mic["OVRL"])
        if dnsmos_enh:
            r["dnsmos_enh"].append(dnsmos_enh["OVRL"])

        # AECMOS (echo + degradation quality)
        aecmos = compute_aecmos(ref, mic, enh)
        if aecmos:
            r["aecmos_echo"].append(aecmos["echo_mos"])
            r["aecmos_deg"].append(aecmos["deg_mos"])

        n += 1
        if n % 20 == 0:
            print(f"  [{n}] processed...")

    print(f"\nGGML evaluation ({n} samples):")
    print(f"{'':>30s} {'ERLE':>6s} {'PESQ':>6s} {'STOI':>6s} "
          f"{'DNS_mic':>7s} {'DNS_enh':>7s} {'AEC_echo':>8s} {'AEC_deg':>7s}")
    for sc in ["double_talk", "single_talk_farend"]:
        r = results[sc]
        if not r["erle"]:
            continue
        erle_s = f"{np.mean(r['erle']):+5.1f}" if r["erle"] else "  N/A"
        pesq_s = f"{np.mean(r['pesq']):5.2f}" if r.get("pesq") else "  N/A"
        stoi_s = f"{np.mean(r['stoi']):5.3f}" if r.get("stoi") else "  N/A"
        dm_s = f"{np.mean(r['dnsmos_mic']):5.2f}" if r["dnsmos_mic"] else "  N/A"
        de_s = f"{np.mean(r['dnsmos_enh']):5.2f}" if r["dnsmos_enh"] else "  N/A"
        ae_s = f"{np.mean(r['aecmos_echo']):6.2f}" if r["aecmos_echo"] else "  N/A"
        ad_s = f"{np.mean(r['aecmos_deg']):5.2f}" if r["aecmos_deg"] else "  N/A"
        count = len(r["erle"])
        print(f"  {sc:>28s} {erle_s} {pesq_s} {stoi_s} {dm_s} {de_s} {ae_s} {ad_s}  (n={count})")


if __name__ == "__main__":
    main()
