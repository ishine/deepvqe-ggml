#!/usr/bin/env python3
"""Comprehensive TensorBoard training report.

Subcommands:
    summary     Enhanced training progress overview (default)
    scalars     Dump scalar tags with percentile samples
    loss        Loss dynamics: component ratios, plateau detection
    gradients   Per-layer gradient norm analysis
    export      Export scalar data to CSV or JSON

Usage:
    python scripts/report_training.py [--log-dir logs/overfit] [subcommand] [options]
"""

import argparse
import csv
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from tb_utils import load_logs, get_latest, get_history


# ─── summary ──────────────────────────────────────────────────────────────────


def cmd_summary(ea, args):
    """Enhanced training progress overview."""
    scalars = ea.Tags().get("scalars", [])

    # Epoch count
    total_events = get_history(ea, "train_epoch/total")
    n_epochs = len(total_events)
    print(f"Epochs completed: {n_epochs}")
    print()

    # Latest snapshot — train and val side by side
    print("=== Latest Values ===")
    pairs = [
        ("total",       "train_epoch/total",   "val/total"),
        ("plcmse",      "train_epoch/plcmse",  "val/plcmse"),
        ("mag_l1",      "train_epoch/mag_l1",  "val/mag_l1"),
        ("time_l1",     "train_epoch/time_l1", "val/time_l1"),
        ("sisdr",       "train_epoch/sisdr",   "val/sisdr"),
        ("smooth_l1",   "train_epoch/smooth_l1", "val/smooth_l1"),
        ("energy_pres", "train_epoch/energy_pres", "val/energy_pres"),
        ("delay",       "train_epoch/delay",   "val/delay"),
        ("entropy",     "train_epoch/entropy", "val/entropy"),
        ("mask_reg",    "train_epoch/mask_reg", "val/mask_reg"),
    ]
    print(f"  {'Component':<16} {'Train':>12} {'Val':>12}")
    print(f"  {'-'*16} {'-'*12} {'-'*12}")
    for label, train_tag, val_tag in pairs:
        t = get_latest(ea, train_tag)
        v = get_latest(ea, val_tag)
        if t is None and v is None:
            continue
        t_str = f"{t.value:12.6f}" if t else "         N/A"
        v_str = f"{v.value:12.6f}" if v else "         N/A"
        print(f"  {label:<16} {t_str} {v_str}")

    # Extra metrics
    for tag, label in [("val/erle_db", "ERLE (dB)"),
                        ("val/delay_acc", "Delay acc"),
                        ("train/lr", "Learning rate"),
                        ("train/temperature", "Temperature")]:
        e = get_latest(ea, tag)
        if e:
            print(f"  {label:<16} {'':>12} {e.value:12.6f}")
    print()

    # Best epoch
    all_val = get_history(ea, "val/total")
    if all_val:
        best = min(all_val, key=lambda e: e.value)
        print(f"=== Best Validation ===")
        print(f"  Epoch {best.step}: val_loss = {best.value:.6f}")
        # Show all metrics at best epoch
        for tag, label in [("val/erle_db", "ERLE"),
                           ("val/mag_l1", "mag_l1"),
                           ("val/time_l1", "time_l1"),
                           ("val/energy_pres", "energy_pres")]:
            events = get_history(ea, tag)
            for e in events:
                if e.step == best.step:
                    print(f"    {label}: {e.value:.6f}")
                    break
        print()

    # ERLE milestones
    erle_events = get_history(ea, "val/erle_db")
    if erle_events:
        print("=== ERLE Milestones ===")
        thresholds = [1, 3, 5, 10, 15, 20, 30, 40, 50]
        for threshold in thresholds:
            first = None
            for e in erle_events:
                if e.value >= threshold:
                    first = e
                    break
            if first:
                print(f"  {threshold:>3} dB first reached at epoch {first.step}")
        max_erle = max(erle_events, key=lambda e: e.value)
        print(f"  Peak: {max_erle.value:.1f} dB at epoch {max_erle.step}")
        print()

    # Epoch history table (last N)
    n = args.n if hasattr(args, "n") else 15
    train_total = get_history(ea, "train_epoch/total", n)
    if train_total:
        val_by_step = {e.step: e.value for e in get_history(ea, "val/total")}
        erle_by_step = {e.step: e.value for e in get_history(ea, "val/erle_db")}
        lr_events = get_history(ea, "train/lr")
        lr_by_step = {e.step: e.value for e in lr_events} if lr_events else {}

        # Best val step
        best_step = min(get_history(ea, "val/total"), key=lambda e: e.value).step if all_val else -1

        print(f"=== Epoch History (last {n}) ===")
        print(f"  {'Epoch':>6}  {'Train':>10}  {'Val':>10}  {'ERLE':>8}  {'LR':>10}")
        print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*10}")
        for t in train_total:
            ep = t.step
            val_v = val_by_step.get(ep)
            erle_v = erle_by_step.get(ep)
            # Find closest LR to this epoch
            lr_v = lr_by_step.get(ep)
            val_str = f"{val_v:10.4f}" if val_v is not None else "       N/A"
            erle_str = f"{erle_v:+7.1f}" if erle_v is not None else "     N/A"
            lr_str = f"{lr_v:10.2e}" if lr_v is not None else "       N/A"
            marker = " *" if ep == best_step else ""
            print(f"  {ep:>6}  {t.value:>10.4f}  {val_str}  {erle_str}  {lr_str}{marker}")
        print()


# ─── scalars ──────────────────────────────────────────────────────────────────


def cmd_scalars(ea, args):
    """Dump scalar tags with percentile samples."""
    scalars = sorted(ea.Tags().get("scalars", []))
    pattern = args.filter if hasattr(args, "filter") and args.filter else None

    for tag in scalars:
        if pattern and not re.search(pattern, tag):
            continue
        events = ea.Scalars(tag)
        n = len(events)
        if n == 0:
            continue

        # Sample at key percentiles
        indices = sorted(set(
            [0] + [min(int(p * (n - 1)), n - 1) for p in [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]]
        ))
        vals = [e.value for e in events]

        print(f"=== {tag} ({n} points) ===")
        for idx in indices:
            e = events[idx]
            print(f"  step={e.step:>6}  value={e.value:.6f}")
        print(f"  min={min(vals):.6f}  max={max(vals):.6f}  final={vals[-1]:.6f}")
        print()


# ─── loss ─────────────────────────────────────────────────────────────────────


def cmd_loss(ea, args):
    """Loss dynamics: component ratios, plateau detection, LR restarts."""
    scalars = ea.Tags().get("scalars", [])

    # Component ratios over time
    ratio_tags = sorted(t for t in scalars if t.startswith("loss_ratio/"))
    if ratio_tags:
        print("=== Loss Component Ratios (weighted % of total) ===")
        # Get epochs from total
        total_events = get_history(ea, "train_epoch/total")
        if not total_events:
            print("  No epoch data")
            return

        # Sample epochs for table
        n_ep = len(total_events)
        sample_eps = sorted(set(
            [0] + [total_events[min(int(p * (n_ep - 1)), n_ep - 1)].step
                   for p in [0.25, 0.5, 0.75, 1.0]]
        ))

        # Build ratio data
        ratio_names = [t.split("/")[1] for t in ratio_tags]
        header = f"  {'Epoch':>6}" + "".join(f"  {n:>10}" for n in ratio_names)
        print(header)
        print(f"  {'-'*6}" + "".join(f"  {'-'*10}" for _ in ratio_names))

        for ep in sample_eps:
            row = f"  {ep:>6}"
            for tag in ratio_tags:
                events = get_history(ea, tag)
                val = None
                for e in events:
                    if e.step == ep:
                        val = e.value
                        break
                if val is not None:
                    row += f"  {val:>9.1%}"
                else:
                    row += "        N/A"
            print(row)
        print()

    # Plateau detection
    print("=== Plateau Detection ===")
    threshold = args.threshold if hasattr(args, "threshold") else 0.001
    window = args.window if hasattr(args, "window") else 10
    total_events = get_history(ea, "train_epoch/total")
    if len(total_events) >= window:
        vals = [e.value for e in total_events]
        plateaus = []
        i = 0
        while i < len(vals) - 1:
            # Check if improvement is below threshold for 'window' consecutive epochs
            run_start = i
            while i < len(vals) - 1 and abs(vals[i] - vals[i+1]) < threshold:
                i += 1
            run_len = i - run_start
            if run_len >= window:
                plateaus.append((total_events[run_start].step,
                                total_events[i].step, run_len,
                                vals[run_start], vals[i]))
            i += 1

        if plateaus:
            print(f"  (threshold={threshold}, window={window} epochs)")
            for start, end, length, v_start, v_end in plateaus:
                print(f"  Plateau: epochs {start}–{end} ({length} epochs), "
                      f"loss {v_start:.6f} → {v_end:.6f}")
        else:
            print(f"  No plateaus detected (threshold={threshold}, window={window})")
    print()

    # LR restart detection
    print("=== Learning Rate Schedule ===")
    lr_events = get_history(ea, "train/lr")
    if lr_events and len(lr_events) > 10:
        lr_vals = [e.value for e in lr_events]
        restarts = []
        for i in range(1, len(lr_vals) - 1):
            # Restart = LR jumps up significantly
            if lr_vals[i] > lr_vals[i-1] * 1.5 and lr_vals[i] > lr_vals[i-1] + 1e-5:
                restarts.append((lr_events[i].step, lr_vals[i-1], lr_vals[i]))

        if restarts:
            print(f"  Detected {len(restarts)} LR restart(s):")
            for step, lr_before, lr_after in restarts:
                print(f"    Step {step}: {lr_before:.2e} → {lr_after:.2e} "
                      f"({lr_after/lr_before:.0f}x)")
        else:
            print("  No LR restarts detected (plateau scheduler?)")

        # LR range
        print(f"  LR range: [{min(lr_vals):.2e}, {max(lr_vals):.2e}]")
        print(f"  Current LR: {lr_vals[-1]:.2e}")
    print()


# ─── gradients ────────────────────────────────────────────────────────────────


def cmd_gradients(ea, args):
    """Per-layer gradient norm analysis from TensorBoard logs."""
    scalars = ea.Tags().get("scalars", [])
    grad_tags = sorted(t for t in scalars if t.startswith("grad_norm/"))

    if not grad_tags:
        print("No grad_norm/* tags found (per-layer gradients not in this log).")
        print("  Only global grad norm available. Run `make diagnose` for per-layer analysis.")
        print()

    # Get recent values for each layer
    n_recent = 100
    print(f"=== Per-Layer Gradient Norms (last {n_recent} steps) ===")
    print(f"  {'Layer':<20} {'Mean':>10} {'Median':>10} {'Max':>10} {'Min':>10} {'Std':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    layer_stats = {}
    for tag in grad_tags:
        name = tag.split("/", 1)[1]
        events = get_history(ea, tag, n_recent)
        if not events:
            continue
        vals = np.array([e.value for e in events])
        layer_stats[name] = vals
        print(f"  {name:<20} {vals.mean():>10.4f} {np.median(vals):>10.4f} "
              f"{vals.max():>10.4f} {vals.min():>10.4f} {vals.std():>10.4f}")
    print()

    # Vanishing/exploding detection
    print("=== Gradient Health ===")
    issues = []
    for name, vals in layer_stats.items():
        median = np.median(vals)
        if median < 0.001:
            issues.append(f"  VANISHING: {name} (median={median:.2e})")
        elif median > 100:
            issues.append(f"  EXPLODING: {name} (median={median:.2e})")

    if issues:
        for msg in issues:
            print(msg)
    else:
        print("  All layers healthy (no vanishing or exploding gradients)")

    # Decoder chain ratio
    dec_layers = {k: v for k, v in layer_stats.items() if k.startswith("dec")}
    if "dec1" in dec_layers and "dec5" in dec_layers:
        ratio = np.median(dec_layers["dec1"]) / (np.median(dec_layers["dec5"]) + 1e-12)
        print(f"\n  Decoder gradient ratio (dec1/dec5): {ratio:.2f}")
        if ratio > 100:
            print("  WARNING: Large gradient imbalance across decoder chain")
        elif ratio < 0.01:
            print("  WARNING: Gradients vanishing through decoder chain")
    print()

    # Global gradient norm summary
    global_events = get_history(ea, "train/grad_norm", n_recent)
    if global_events:
        vals = np.array([e.value for e in global_events])
        clipped = np.sum(vals >= 4.99)
        print(f"=== Global Gradient Norm (last {len(vals)} steps) ===")
        print(f"  Mean: {vals.mean():.2f}  Median: {np.median(vals):.2f}  "
              f"Max: {vals.max():.2f}  Clipped: {clipped}/{len(vals)}")
    print()


# ─── export ───────────────────────────────────────────────────────────────────


def cmd_export(ea, args):
    """Export scalar data to CSV or JSON."""
    scalars = sorted(ea.Tags().get("scalars", []))
    pattern = args.tags if hasattr(args, "tags") and args.tags else None
    fmt = args.format if hasattr(args, "format") else "csv"
    output = args.output if hasattr(args, "output") else None

    filtered = [t for t in scalars if not pattern or re.search(pattern, t)]
    if not filtered:
        print(f"No tags matching '{pattern}'", file=sys.stderr)
        return

    f = open(output, "w") if output else sys.stdout

    if fmt == "csv":
        writer = csv.writer(f)
        writer.writerow(["tag", "step", "wall_time", "value"])
        for tag in filtered:
            for e in ea.Scalars(tag):
                writer.writerow([tag, e.step, e.wall_time, e.value])
    elif fmt == "json":
        data = {}
        for tag in filtered:
            data[tag] = [{"step": e.step, "wall_time": e.wall_time, "value": e.value}
                         for e in ea.Scalars(tag)]
        json.dump(data, f, indent=2)
        f.write("\n")

    if output:
        f.close()
        print(f"Exported {len(filtered)} tags to {output}", file=sys.stderr)


# ─── main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive TensorBoard training report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--log-dir", default="../logs", help="TensorBoard log directory")
    sub = parser.add_subparsers(dest="command")

    # summary
    p_sum = sub.add_parser("summary", help="Enhanced training progress overview")
    p_sum.add_argument("-n", type=int, default=15, help="Number of recent epochs")

    # scalars
    p_sc = sub.add_parser("scalars", help="Dump scalar tags with samples")
    p_sc.add_argument("--filter", default=None, help="Regex filter on tag names")

    # loss
    p_loss = sub.add_parser("loss", help="Loss dynamics analysis")
    p_loss.add_argument("--threshold", type=float, default=0.001, help="Plateau threshold")
    p_loss.add_argument("--window", type=int, default=10, help="Min plateau length (epochs)")

    # gradients
    sub.add_parser("gradients", help="Per-layer gradient norm analysis")

    # export
    p_exp = sub.add_parser("export", help="Export scalar data")
    p_exp.add_argument("--format", choices=["csv", "json"], default="csv")
    p_exp.add_argument("--output", default=None, help="Output file (default: stdout)")
    p_exp.add_argument("--tags", default=None, help="Regex filter on tag names")

    args = parser.parse_args()

    path, ea = load_logs(args.log_dir)
    print(f"Log dir: {path}")
    print()

    commands = {
        "summary": cmd_summary,
        "scalars": cmd_scalars,
        "loss": cmd_loss,
        "gradients": cmd_gradients,
        "export": cmd_export,
    }

    cmd = args.command or "summary"
    commands[cmd](ea, args)


if __name__ == "__main__":
    main()
