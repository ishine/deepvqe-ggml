#!/usr/bin/env python3
"""Diagnose decoder activation collapse from a checkpoint.

Loads a checkpoint, runs one forward pass on dummy data, and prints
detailed diagnostics for each decoder block: weight norms, activation
stats, skip vs main path magnitudes, BN running statistics, and CCM
mask decomposition.

Usage:
    python scripts/diagnose_decoder.py [--checkpoint PATH] [--config PATH]
"""

import argparse
import glob
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import load_config
from src.model import DeepVQEAEC
from src.losses import mask_mag_from_raw
from data.dataset import DummyAECDataset


def find_checkpoint(ckpt_dir="checkpoints/overfit"):
    """Auto-detect checkpoint: best.pt → latest epoch_*.pt → checkpoints/best.pt."""
    best = os.path.join(ckpt_dir, "best.pt")
    if os.path.exists(best):
        return best
    epoch_files = sorted(glob.glob(os.path.join(ckpt_dir, "epoch_*.pt")))
    if epoch_files:
        return epoch_files[-1]
    fallback = "checkpoints/best.pt"
    if os.path.exists(fallback):
        return fallback
    return None


def fmt(val, width=10):
    """Format a float for table display."""
    if abs(val) < 1e-6:
        return f"{val:{width}.2e}"
    return f"{val:{width}.4f}"


def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose decoder activation collapse")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path")
    parser.add_argument("--config", default="configs/overfit.yaml", help="Config path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)

    # Find checkpoint
    ckpt_path = args.checkpoint or find_checkpoint()
    if ckpt_path is None or not os.path.exists(ckpt_path):
        print(f"No checkpoint found (tried: {ckpt_path})")
        sys.exit(1)
    print(f"Config:     {args.config}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Device:     {device}")

    # Load model
    model = DeepVQEAEC.from_config(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')})")

    # Create dummy data
    ds = DummyAECDataset(
        length=cfg.data.num_train,
        n_fft=cfg.audio.n_fft,
        hop_length=cfg.audio.hop_length,
        target_len=int(cfg.training.clip_length_sec * cfg.audio.sample_rate),
        delay_range=tuple(int(x) for x in cfg.data.delay_range),
    )
    batch = ds[0]
    mic_stft = batch["mic_stft"].unsqueeze(0).to(device)
    ref_stft = batch["ref_stft"].unsqueeze(0).to(device)

    # ── Section A: Weight norms per decoder block ──
    section("A. Weight Norms Per Decoder Block")
    dec_names = ["dec5", "dec4", "dec3", "dec2", "dec1"]
    print(f"  {'Block':<8} {'Submodule':<25} {'L2 Norm':>10} {'Mean':>10} {'Std':>10} {'Shape'}")
    print(f"  {'-'*8} {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*20}")
    for dec_name in dec_names:
        dec = getattr(model, dec_name)
        for sub_name, param in dec.named_parameters():
            norm = param.data.norm().item()
            mean = param.data.mean().item()
            std = param.data.std().item()
            shape = str(list(param.shape))
            print(f"  {dec_name:<8} {sub_name:<25} {fmt(norm)} {fmt(mean)} {fmt(std)} {shape}")

    # ── Hooks for activation capture ──
    activations = {}

    def make_hook(name):
        def hook(module, input, output):
            # For decoder blocks, input is (x, x_en) tuple
            if isinstance(input, tuple) and len(input) >= 2:
                activations[f"{name}_input_x"] = input[0].detach()
                activations[f"{name}_input_skip"] = input[1].detach()
            elif isinstance(input, tuple) and len(input) == 1:
                activations[f"{name}_input"] = input[0].detach()
            activations[f"{name}_output"] = output.detach() if isinstance(output, torch.Tensor) else output
            return output
        return hook

    # Hook decoder blocks
    handles = []
    for name in dec_names:
        dec = getattr(model, name)
        handles.append(dec.register_forward_hook(make_hook(name)))

    # Hook skip_conv inside each decoder
    for name in dec_names:
        dec = getattr(model, name)
        handles.append(dec.skip_conv.register_forward_hook(make_hook(f"{name}.skip_conv")))

    # Hook bottleneck output
    handles.append(model.bottleneck.register_forward_hook(make_hook("bottleneck")))

    # Forward pass
    with torch.no_grad():
        enhanced, delay_dist, d1 = model(mic_stft, ref_stft, return_delay=True)

    for h in handles:
        h.remove()

    # ── Section B: Per-block activation stats ──
    section("B. Per-Block Activation Statistics")
    print(f"  {'Tensor':<25} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'P1':>10} {'P99':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    stat_keys = ["bottleneck_output"]
    for name in dec_names:
        stat_keys.extend([f"{name}_input_x", f"{name}_output"])

    for key in stat_keys:
        if key not in activations:
            continue
        t = activations[key].float()
        p1 = torch.quantile(t, 0.01).item()
        p99 = torch.quantile(t, 0.99).item()
        print(f"  {key:<25} {fmt(t.mean().item())} {fmt(t.std().item())} "
              f"{fmt(t.min().item())} {fmt(t.max().item())} {fmt(p1)} {fmt(p99)}")

    # ── Section C: Skip vs main path magnitude ──
    section("C. Skip Connection vs Main Path Magnitude")
    print(f"  {'Block':<8} {'||x|| (main)':>14} {'||skip(en)||':>14} {'Ratio sk/x':>12} {'Dominance'}")
    print(f"  {'-'*8} {'-'*14} {'-'*14} {'-'*12} {'-'*15}")

    for name in dec_names:
        x_key = f"{name}_input_x"
        skip_key = f"{name}.skip_conv_output"
        if x_key not in activations or skip_key not in activations:
            print(f"  {name:<8} (data not captured)")
            continue
        x_norm = activations[x_key].float().norm().item()
        skip_norm = activations[skip_key].float().norm().item()
        ratio = skip_norm / (x_norm + 1e-12)
        if ratio > 2:
            dom = "SKIP dominates"
        elif ratio < 0.5:
            dom = "MAIN dominates"
        else:
            dom = "balanced"
        print(f"  {name:<8} {x_norm:>14.4f} {skip_norm:>14.4f} {ratio:>12.2f} {dom}")

    # ── Section D: BatchNorm running statistics ──
    section("D. BatchNorm Running Statistics")
    print(f"  {'Module':<35} {'run_mean':<22} {'run_var':<22} {'gamma':<22} {'beta':<22}")
    print(f"  {'-'*35} {'-'*22} {'-'*22} {'-'*22} {'-'*22}")

    def bn_stats(name, bn):
        rm = bn.running_mean
        rv = bn.running_var
        w = bn.weight
        b = bn.bias
        rm_str = f"[{rm.min().item():.4f}, {rm.max().item():.4f}]"
        rv_str = f"[{rv.min().item():.4f}, {rv.max().item():.4f}]"
        w_str = f"[{w.min().item():.4f}, {w.max().item():.4f}]"
        b_str = f"[{b.min().item():.4f}, {b.max().item():.4f}]"
        print(f"  {name:<35} {rm_str:<22} {rv_str:<22} {w_str:<22} {b_str:<22}")

    for dec_name in dec_names:
        dec = getattr(model, dec_name)
        # ResidualBlock BN
        bn_stats(f"{dec_name}.resblock.bn", dec.resblock.bn)
        # Outer BN (only for non-last blocks with legacy BN)
        if hasattr(dec, "bn"):
            bn_stats(f"{dec_name}.bn", dec.bn)

    # dec1 identity init stats (is_last=True, no BN)
    conv = model.dec1.deconv.conv
    print()
    print(f"  dec1.deconv.conv (Conv2d {conv.in_channels}→{conv.out_channels}, is_last=True, no BN):")
    print(f"    weight: norm={conv.weight.data.norm().item():.4f}, "
          f"mean={conv.weight.data.mean().item():.6f}, std={conv.weight.data.std().item():.6f}")
    print(f"    bias[7] (identity, even freq): {conv.bias.data[7].item():.6f}")
    print(f"    bias[34] (identity, odd freq): {conv.bias.data[34].item():.6f}")

    # ── Section E: SubpixelConv2d even/odd analysis (dec1) ──
    section("E. SubpixelConv2d Analysis (dec1)")
    deconv = model.dec1.deconv
    w = deconv.conv.weight.data  # (out_ch*2, in_ch, kH, kW)
    b = deconv.conv.bias.data    # (out_ch*2,)
    out_ch = w.shape[0] // 2
    print(f"  Conv shape: {list(w.shape)} (out_ch*2={w.shape[0]}, in_ch={w.shape[1]})")
    print(f"  Bias shape: {list(b.shape)}")
    print()

    # Even group (first out_ch channels) vs odd group (last out_ch channels)
    w_even, w_odd = w[:out_ch], w[out_ch:]
    b_even, b_odd = b[:out_ch], b[out_ch:]
    print(f"  {'Group':<10} {'W norm':>10} {'W mean':>10} {'W std':>10} {'B mean':>10} {'B std':>10}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'even':<10} {fmt(w_even.norm().item())} {fmt(w_even.mean().item())} "
          f"{fmt(w_even.std().item())} {fmt(b_even.mean().item())} {fmt(b_even.std().item())}")
    print(f"  {'odd':<10} {fmt(w_odd.norm().item())} {fmt(w_odd.mean().item())} "
          f"{fmt(w_odd.std().item())} {fmt(b_odd.mean().item())} {fmt(b_odd.std().item())}")
    print()

    # CCM identity init channels
    print(f"  Identity init biases (should be ~1.0 if preserved):")
    print(f"    bias[7]  (even center tap) = {b[7].item():.6f}")
    print(f"    bias[34] (odd center tap)  = {b[34].item():.6f}")
    print(f"    All other biases: mean={torch.cat([b[:7], b[8:34], b[35:]]).mean().item():.6f}, "
          f"std={torch.cat([b[:7], b[8:34], b[35:]]).std().item():.6f}")

    # ── Section F: CCM mask decomposition ──
    section("F. CCM Mask Decomposition")
    d1_tensor = d1.detach()  # (B, 27, T, F)
    print(f"  d1 shape: {list(d1_tensor.shape)}")
    print(f"  d1 stats: mean={d1_tensor.mean().item():.6f}, std={d1_tensor.std().item():.6f}, "
          f"min={d1_tensor.min().item():.6f}, max={d1_tensor.max().item():.6f}")
    print()

    # Per-basis-group means (3 groups of 9 channels)
    print(f"  Per-basis-group (3 groups × 9 kernel taps):")
    for r in range(3):
        group = d1_tensor[:, r*9:(r+1)*9]
        print(f"    Group {r}: mean={group.mean().item():.6f}, std={group.std().item():.6f}")

    # Mask magnitude via cube-root-of-unity decomposition
    mag = mask_mag_from_raw(d1_tensor)  # (B, 9, T, F)
    center = mag[:, 7]   # (B, T, F)
    off = torch.cat([mag[:, :7], mag[:, 8:]], dim=1)  # (B, 8, T, F)

    print()
    print(f"  Mask magnitudes (from cube-root decomposition):")
    print(f"    Center tap |H[7]|:  mean={center.mean().item():.6f}, "
          f"std={center.std().item():.6f}, min={center.min().item():.6f}, "
          f"max={center.max().item():.6f}")
    print(f"    Off-center |H|:     mean={off.mean().item():.6f}, "
          f"std={off.std().item():.6f}")
    print()

    # Per-tap breakdown
    print(f"  Per-tap magnitudes (3×3 kernel, causal padding [1,1,2,0]):")
    print(f"    {'Tap':<6} {'(m,n)':<8} {'Meaning':<25} {'|H| mean':>10} {'|H| std':>10}")
    print(f"    {'-'*6} {'-'*8} {'-'*25} {'-'*10} {'-'*10}")
    tap_labels = [
        (0, 0, "t-2, f-1"),
        (0, 1, "t-2, f"),
        (0, 2, "t-2, f+1"),
        (1, 0, "t-1, f-1"),
        (1, 1, "t-1, f"),
        (1, 2, "t-1, f+1"),
        (2, 0, "t, f-1"),
        (2, 1, "t, f (CENTER)"),
        (2, 2, "t, f+1"),
    ]
    for idx, (m, n, label) in enumerate(tap_labels):
        h = mag[:, idx]
        marker = " <-- identity" if idx == 7 else ""
        print(f"    {idx:<6} ({m},{n})    {label:<25} {h.mean().item():>10.6f} "
              f"{h.std().item():>10.6f}{marker}")

    # ── Summary ──
    section("Summary")
    print(f"  ERLE proxy: center_tap_mean = {center.mean().item():.4f} (ideal: 1.0)")
    print(f"  Decoder output std collapse: "
          f"dec5={activations.get('dec5_output', torch.zeros(1)).std().item():.4f} → "
          f"dec1={d1_tensor.std().item():.4f}")

    # Check if skips dominate
    for name in dec_names:
        x_key = f"{name}_input_x"
        skip_key = f"{name}.skip_conv_output"
        if x_key in activations and skip_key in activations:
            ratio = activations[skip_key].norm().item() / (activations[x_key].norm().item() + 1e-12)
            if ratio > 5:
                print(f"  WARNING: {name} skip connection dominates main path ({ratio:.1f}x)")
            elif ratio < 0.1:
                print(f"  WARNING: {name} skip connection negligible ({ratio:.2f}x)")

    print()


if __name__ == "__main__":
    main()
