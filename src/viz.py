"""Visualization utilities for DeepVQE training and analysis.

Pure functions that take tensors and return matplotlib figures,
plus hook registration and TensorBoard logging helpers.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Spectrogram comparison
# ---------------------------------------------------------------------------

def plot_spectrogram_comparison(mic_stft, enh_stft, clean_stft, sr, hop_length):
    """2x2 grid: mic | enhanced | clean | residual (enhanced - clean).

    Args:
        mic_stft: (F, T, 2) or (B, F, T, 2) — microphone STFT
        enh_stft: same shape — enhanced STFT
        clean_stft: same shape — clean STFT
        sr: sample rate
        hop_length: hop size

    Returns:
        matplotlib Figure
    """
    def _mag_db(x):
        if x.dim() == 4:
            x = x[0]
        mag = torch.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2 + 1e-12)
        return 20 * torch.log10(mag + 1e-12).cpu().numpy()

    mic_db = _mag_db(mic_stft)
    enh_db = _mag_db(enh_stft)
    clean_db = _mag_db(clean_stft)
    res_db = enh_db - clean_db

    vmin = min(mic_db.min(), enh_db.min(), clean_db.min())
    vmax = max(mic_db.max(), enh_db.max(), clean_db.max())

    F, T = mic_db.shape
    t_axis = np.arange(T) * hop_length / sr
    f_axis = np.arange(F) * sr / ((F - 1) * 2)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    titles = ["Mic", "Enhanced", "Clean", "Residual (enh - clean)"]
    data = [mic_db, enh_db, clean_db, res_db]

    for ax, title, d in zip(axes.flat, titles, data):
        if title == "Residual (enh - clean)":
            im = ax.pcolormesh(t_axis, f_axis / 1000, d, cmap="RdBu_r", shading="auto")
        else:
            im = ax.pcolormesh(t_axis, f_axis / 1000, d, vmin=vmin, vmax=vmax,
                               cmap="magma", shading="auto")
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Freq (kHz)")
        fig.colorbar(im, ax=ax, label="dB")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Delay distribution with ground truth
# ---------------------------------------------------------------------------

def plot_delay_with_gt(delay_dist, gt_delay_samples, hop_length, dmax):
    """Delay heatmap (T x dmax) with ground-truth overlay.

    Args:
        delay_dist: (T, dmax) or (B, T, dmax) — predicted delay distribution
        gt_delay_samples: int or scalar tensor — ground truth delay in samples
        hop_length: int
        dmax: int

    Returns:
        matplotlib Figure
    """
    if delay_dist.dim() == 3:
        delay_dist = delay_dist[0]
    dd = delay_dist.detach().cpu().numpy()  # (T, dmax)
    T, D = dd.shape

    if isinstance(gt_delay_samples, torch.Tensor):
        gt_delay_samples = gt_delay_samples.item()

    # GT frame index (same convention as compute_delay_loss)
    gt_frame = dmax - 1 - round(gt_delay_samples / hop_length)
    gt_frame = max(0, min(dmax - 1, gt_frame))

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    im = ax.imshow(dd.T, aspect="auto", origin="lower", cmap="viridis",
                   interpolation="nearest")
    fig.colorbar(im, ax=ax, label="Probability")

    # GT line
    ax.axhline(gt_frame, color="red", linestyle="--", linewidth=1.5,
               label=f"GT delay={gt_delay_samples} samp (frame {gt_frame})")

    # Per-frame correctness dots
    peak_frames = dd.argmax(axis=1)
    for t in range(T):
        correct = abs(int(peak_frames[t]) - gt_frame) <= 1
        color = "lime" if correct else "red"
        ax.plot(t, peak_frames[t], "o", color=color, markersize=2)

    ax.set_xlabel("Frame")
    ax.set_ylabel("Delay index")
    ax.set_title("Delay Distribution")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CCM mask analysis
# ---------------------------------------------------------------------------

def plot_ccm_mask(mask_27ch, mic_stft):
    """Decompose 27ch mask and compare to ideal ratio mask.

    Args:
        mask_27ch: (27, T, F) or (B, 27, T, F) — raw mask from decoder
        mic_stft: (F, T, 2) or (B, F, T, 2) — microphone STFT

    Returns:
        matplotlib Figure
    """
    if mask_27ch.dim() == 4:
        mask_27ch = mask_27ch[0]
    if mic_stft.dim() == 4:
        mic_stft = mic_stft[0]

    mask = mask_27ch.detach().cpu().float()  # (27, T, F)

    # Cube-root basis vectors
    v_real = torch.tensor([1.0, -0.5, -0.5])
    v_imag = torch.tensor([0.0, np.sqrt(3) / 2, -np.sqrt(3) / 2])

    # Reshape: (3, 9, T, F)
    m = mask.view(3, 9, mask.shape[1], mask.shape[2])
    H_real = (v_real[:, None, None, None] * m).sum(dim=0)  # (9, T, F)
    H_imag = (v_imag[:, None, None, None] * m).sum(dim=0)  # (9, T, F)

    # Mean mask magnitude over the 9 kernel elements
    H_mag = torch.sqrt(H_real ** 2 + H_imag ** 2 + 1e-12).mean(dim=0)  # (T, F)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    im0 = axes[0].imshow(H_mag.numpy().T, aspect="auto", origin="lower",
                         cmap="viridis", interpolation="nearest")
    axes[0].set_title("Mean Mask Magnitude |H|")
    axes[0].set_xlabel("Frame")
    axes[0].set_ylabel("Freq bin")
    fig.colorbar(im0, ax=axes[0])

    # Ideal ratio mask: |clean|/|mic| — we don't have clean here, so show |mic| magnitude
    mic_mag = torch.sqrt(mic_stft[..., 0] ** 2 + mic_stft[..., 1] ** 2 + 1e-12)
    mic_mag_db = 20 * torch.log10(mic_mag.cpu() + 1e-12)
    im1 = axes[1].imshow(mic_mag_db.numpy(), aspect="auto", origin="lower",
                         cmap="magma", interpolation="nearest")
    axes[1].set_title("Mic Magnitude (dB) — for reference")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Freq bin")
    fig.colorbar(im1, ax=axes[1], label="dB")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Encoder activation heatmaps
# ---------------------------------------------------------------------------

def plot_encoder_activations(activation_dict):
    """Channel-mean heatmaps for each encoder stage.

    Args:
        activation_dict: dict mapping layer names to tensors (B, C, T, F)

    Returns:
        matplotlib Figure
    """
    enc_keys = [k for k in activation_dict
                if k.startswith(("mic_enc", "far_enc"))]
    if not enc_keys:
        enc_keys = list(activation_dict.keys())[:7]

    n = len(enc_keys)
    if n == 0:
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        ax.text(0.5, 0.5, "No encoder activations captured", ha="center")
        return fig

    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flat

    for i, key in enumerate(enc_keys):
        act = activation_dict[key]
        if act.dim() == 4:
            act = act[0]
        # Channel mean: (T, F)
        hm = act.float().mean(dim=0).cpu().numpy()
        im = axes[i].imshow(hm.T, aspect="auto", origin="lower",
                            cmap="viridis", interpolation="nearest")
        axes[i].set_title(f"{key} ({act.shape[0]}ch)")
        axes[i].set_xlabel("Frame")
        axes[i].set_ylabel("Freq bin")
        fig.colorbar(im, ax=axes[i])

    # Hide unused axes
    for j in range(i + 1, len(list(axes))):
        axes[j].set_visible(False)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Activation statistics
# ---------------------------------------------------------------------------

def plot_activation_stats(activation_dict):
    """Bar charts: mean, std, dead fraction (==0), max per layer.

    Args:
        activation_dict: dict mapping layer names to tensors

    Returns:
        matplotlib Figure
    """
    names = list(activation_dict.keys())
    if not names:
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        ax.text(0.5, 0.5, "No activations captured", ha="center")
        return fig

    means, stds, deads, maxes = [], [], [], []
    for k in names:
        act = activation_dict[k].float()
        means.append(act.mean().item())
        stds.append(act.std().item())
        deads.append((act == 0).float().mean().item())
        maxes.append(act.abs().max().item())

    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    titles = ["Mean", "Std", "Dead Fraction", "Max |act|"]
    data = [means, stds, deads, maxes]
    colors = ["steelblue", "darkorange", "crimson", "seagreen"]

    for ax, title, vals, c in zip(axes, titles, data, colors):
        ax.bar(x, vals, color=c)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_title(title)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Hook registration
# ---------------------------------------------------------------------------

def register_hooks(model):
    """Register forward hooks on key layers to capture activations.

    Args:
        model: DeepVQEAEC instance (unwrapped)

    Returns:
        (activation_store, hook_handles)
    """
    activation_store = {}
    hook_handles = []

    target_names = [
        "fe_mic", "fe_ref",
        "mic_enc1", "mic_enc2", "mic_enc3", "mic_enc4", "mic_enc5",
        "far_enc1", "far_enc2",
        "align", "bottleneck",
        "dec5", "dec4", "dec3", "dec2", "dec1",
        "ccm",
    ]

    for name in target_names:
        module = getattr(model, name, None)
        if module is None:
            continue

        def _make_hook(n):
            def hook_fn(module, input, output):
                # AlignBlock returns tuple when return_delay=True
                if isinstance(output, tuple):
                    activation_store[n] = output[0].detach().cpu()
                else:
                    activation_store[n] = output.detach().cpu()
            return hook_fn

        h = module.register_forward_hook(_make_hook(name))
        hook_handles.append(h)

    return activation_store, hook_handles


def remove_hooks(hook_handles):
    """Remove all registered hooks."""
    for h in hook_handles:
        h.remove()


# ---------------------------------------------------------------------------
# TensorBoard logging helpers
# ---------------------------------------------------------------------------

def log_weight_histograms(writer, model, epoch):
    """Log weight and gradient histograms to TensorBoard.

    Args:
        writer: SummaryWriter
        model: nn.Module
        epoch: current epoch
    """
    for name, param in model.named_parameters():
        writer.add_histogram(f"weights/{name}", param.detach().cpu(), epoch)
        if param.grad is not None:
            writer.add_histogram(f"gradients/{name}", param.grad.detach().cpu(), epoch)


def log_per_layer_grad_norms(writer, model, global_step):
    """Log L2 gradient norm per top-level module.

    Args:
        writer: SummaryWriter
        model: nn.Module (unwrapped)
        global_step: int
    """
    for name, module in model.named_children():
        params = [p for p in module.parameters() if p.grad is not None]
        if not params:
            continue
        total_norm = torch.sqrt(
            sum(p.grad.detach().float().pow(2).sum() for p in params)
        ).item()
        writer.add_scalar(f"grad_norm/{name}", total_norm, global_step)


def log_loss_ratios(writer, components, global_step):
    """Log fraction each loss term contributes to total.

    Args:
        writer: SummaryWriter
        components: dict of loss component name -> value (tensor or float)
        global_step: int
    """
    total = components.get("total", None)
    if total is None:
        return
    total_val = total.item() if isinstance(total, torch.Tensor) else total
    if abs(total_val) < 1e-12:
        return

    for name, val in components.items():
        if name == "total":
            continue
        v = val.item() if isinstance(val, torch.Tensor) else val
        writer.add_scalar(f"loss_ratio/{name}", abs(v) / abs(total_val), global_step)
