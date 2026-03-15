"""DeepVQE AEC training script.

Features:
- HF Accelerate for multi-GPU (DDP) and single-GPU training
- AdamW optimizer with linear warmup + cosine restarts / ReduceLROnPlateau
- Mixed precision (BF16 via Accelerate)
- Gradient accumulation for large effective batch sizes
- Gradient clipping
- TensorBoard logging (loss, lr, grad norms, audio, spectrograms, delay heatmaps)
- Checkpointing (last N + best)
- HF Hub uploads (checkpoints + TensorBoard logs)
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.dataset import AECDataset, DummyAECDataset
from src.config import load_config
from src.losses import DeepVQELoss, mask_magnitude_regularizer
from src.model import DeepVQEAEC
from src.stft import istft
from src.viz import (
    add_scalar_with_help,
    log_loss_ratios,
    log_per_layer_grad_norms,
    log_weight_histograms,
    plot_activation_stats,
    plot_ccm_mask,
    plot_delay_with_gt,
    plot_encoder_activations,
    plot_spectrogram_comparison,
    register_hooks,
    remove_hooks,
)


def collate_fn(batch):
    """Custom collate that handles metadata dicts."""
    mic_stft = torch.stack([b["mic_stft"] for b in batch])
    ref_stft = torch.stack([b["ref_stft"] for b in batch])
    clean_stft = torch.stack([b["clean_stft"] for b in batch])
    mic_wav = torch.stack([b["mic_wav"] for b in batch])
    clean_wav = torch.stack([b["clean_wav"] for b in batch])
    delay_samples = torch.tensor([b["delay_samples"] for b in batch], dtype=torch.long)
    metadata = [b["metadata"] for b in batch]
    return {
        "mic_stft": mic_stft,
        "ref_stft": ref_stft,
        "clean_stft": clean_stft,
        "mic_wav": mic_wav,
        "clean_wav": clean_wav,
        "delay_samples": delay_samples,
        "metadata": metadata,
    }


def delay_samples_to_frame(delay_samples, hop_length, dmax):
    """Convert delay in samples to AlignBlock frame index.

    AlignBlock unfold convention: index d corresponds to ref[t - (dmax-1) + d].
    So d = dmax-1 means no delay, d = 0 means max delay.
    target_frame = dmax - 1 - round(delay_samples / hop_length)
    """
    delay_frames = torch.round(delay_samples.float() / hop_length).long()
    target_frames = (dmax - 1) - delay_frames
    return target_frames.clamp(0, dmax - 1)


def compute_delay_loss(delay_dist, delay_samples, hop_length, dmax):
    """Cross-entropy loss between predicted delay distribution and ground truth.

    Args:
        delay_dist: (B, T, dmax) — predicted delay distribution from AlignBlock
        delay_samples: (B,) — ground truth delay in samples
        hop_length: int
        dmax: int

    Returns:
        loss: scalar cross-entropy
        accuracy: fraction of examples where peak is within ±1 frame of truth
    """
    B, T, D = delay_dist.shape
    target_frames = delay_samples_to_frame(delay_samples, hop_length, dmax)  # (B,)

    # Cross-entropy: average over all frames (same target for all frames in an example)
    # delay_dist is already a probability distribution, use NLL
    log_probs = torch.log(delay_dist + 1e-10)  # (B, T, dmax)
    # Gather the log-prob at the target frame for each example, averaged over frames
    target_expanded = target_frames[:, None, None].expand(B, T, 1)  # (B, T, 1)
    nll = -log_probs.gather(dim=-1, index=target_expanded).squeeze(-1)  # (B, T)
    loss = nll.mean()

    # Accuracy: peak of mean attention over frames, within ±1 frame of target
    avg_dist = delay_dist.mean(dim=1)  # (B, dmax)
    peak_frames = avg_dist.argmax(dim=-1)  # (B,)
    correct = (peak_frames - target_frames).abs() <= 1
    accuracy = correct.float().mean()

    return loss, accuracy


def compute_attention_entropy(delay_dist):
    """Entropy of the delay attention distribution: H = -sum(p * log(p))."""
    return -(delay_dist * torch.log(delay_dist + 1e-10)).sum(dim=-1).mean()


def compute_erle(mic_wav, enhanced_wav, clean_wav):
    """Compute Echo Return Loss Enhancement in dB.

    ERLE = 10 * log10(||mic - clean||^2 / ||enhanced - clean||^2)
    Positive means echo was reduced.
    """
    echo_plus_noise = mic_wav - clean_wav
    residual = enhanced_wav - clean_wav
    echo_power = (echo_plus_noise ** 2).sum(dim=-1)
    residual_power = (residual ** 2).sum(dim=-1)
    erle = 10 * torch.log10(echo_power / (residual_power + 1e-10))
    return erle.mean()


def get_warmup_scheduler(optimizer, cfg, steps_per_epoch):
    """Linear warmup scheduler (per-step) for the first few epochs."""
    warmup_steps = cfg.training.warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, step / warmup_steps)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_epoch_scheduler(optimizer, cfg):
    """Per-epoch scheduler: plateau or cosine with warm restarts."""
    if cfg.training.lr_scheduler == "cosine_restarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg.training.lr_cosine_t0,
            T_mult=cfg.training.lr_cosine_tmult,
            eta_min=cfg.training.lr_min,
        )
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.training.lr_factor,
        patience=cfg.training.lr_patience,
        min_lr=cfg.training.lr_min,
    )


def _unwrap(model, accelerator=None):
    """Get the raw model, unwrapping Accelerate DDP + torch.compile."""
    if accelerator:
        model = accelerator.unwrap_model(model)
    return getattr(model, "_orig_mod", model)


def save_checkpoint(model, optimizer, schedulers, epoch, loss, path, accelerator=None):
    sched_states = {k: s.state_dict() for k, s in schedulers.items()}
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": _unwrap(model, accelerator).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_states": sched_states,
            "loss": loss,
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None, schedulers=None, accelerator=None):
    ckpt = torch.load(path, weights_only=False)
    state = ckpt["model_state_dict"]
    # Strip _orig_mod. prefix for backward compat with old checkpoints
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    _unwrap(model, accelerator).load_state_dict(state)
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if schedulers and "scheduler_states" in ckpt:
        for k, s in schedulers.items():
            if k in ckpt["scheduler_states"]:
                s.load_state_dict(ckpt["scheduler_states"][k])
    return ckpt["epoch"]


def manage_checkpoints(ckpt_dir, keep_n):
    """Keep only the last N checkpoints (excluding best)."""
    ckpts = sorted(ckpt_dir.glob("epoch_*.pt"), key=lambda p: p.stat().st_mtime)
    while len(ckpts) > keep_n:
        ckpts.pop(0).unlink()


def log_audio_and_spectrograms(writer, model, val_batch, epoch, cfg, device,
                               accelerator=None):
    """Log audio samples, spectrograms, and diagnostic figures to TensorBoard."""
    import matplotlib.pyplot as plt

    raw_model = _unwrap(model, accelerator)
    model.eval()

    # Register hooks for activation capture
    activation_store, hook_handles = register_hooks(raw_model)

    with torch.no_grad():
        mic_stft = val_batch["mic_stft"][:1].to(device)
        ref_stft = val_batch["ref_stft"][:1].to(device)
        clean_stft = val_batch["clean_stft"][:1].to(device)

        # Use unwrapped model for inference to avoid DDP forward issues with B=1
        enhanced, delay_dist, mask_raw = raw_model(mic_stft, ref_stft, return_delay=True)
        length = val_batch["clean_wav"].shape[-1]

        mic_wav = istft(mic_stft, cfg.audio.n_fft, cfg.audio.hop_length, length=length)
        enh_wav = istft(enhanced, cfg.audio.n_fft, cfg.audio.hop_length, length=length)
        clean_wav = val_batch["clean_wav"][:1].to(device)

        sr = cfg.audio.sample_rate
        writer.add_audio("audio/mic", mic_wav[0].cpu(), epoch, sample_rate=sr)
        writer.add_audio("audio/enhanced", enh_wav[0].cpu(), epoch, sample_rate=sr)
        writer.add_audio("audio/clean", clean_wav[0].cpu(), epoch, sample_rate=sr)

        # Spectrogram comparison
        fig = plot_spectrogram_comparison(
            mic_stft, enhanced, clean_stft, sr, cfg.audio.hop_length,
        )
        writer.add_figure("spectrograms/comparison", fig, epoch)
        plt.close(fig)

        # Delay distribution with ground truth
        if delay_dist is not None:
            gt_delay = val_batch["delay_samples"][0]
            fig = plot_delay_with_gt(
                delay_dist, gt_delay, cfg.audio.hop_length, cfg.model.dmax,
            )
            writer.add_figure("delay/distribution_with_gt", fig, epoch)
            plt.close(fig)

        # CCM mask analysis (from mask_head output = 27ch mask)
        mask_key = "mask_head" if "mask_head" in activation_store else "dec1"
        if mask_key in activation_store:
            fig = plot_ccm_mask(activation_store[mask_key], mic_stft)
            writer.add_figure("ccm/mask_magnitude", fig, epoch)
            plt.close(fig)

        # Encoder activations
        if activation_store:
            fig = plot_encoder_activations(activation_store)
            writer.add_figure("activations/encoder_stages", fig, epoch)
            plt.close(fig)

            fig = plot_activation_stats(activation_store)
            writer.add_figure("activations/statistics", fig, epoch)
            plt.close(fig)

    remove_hooks(hook_handles)
    model.train()


def _init_hub_api(cfg):
    """Initialize HF Hub API if push_to_hub is enabled. Returns (api, repo_id) or (None, None)."""
    if not cfg.hub.push_to_hub or not cfg.hub.hub_model_id:
        return None, None
    from huggingface_hub import HfApi
    api = HfApi()
    repo_id = cfg.hub.hub_model_id
    api.create_repo(repo_id, exist_ok=True)
    return api, repo_id


def _hub_upload_file(api, repo_id, local_path, path_in_repo):
    """Upload a single file to HF Hub (async, fire-and-forget)."""
    if api is None:
        return
    try:
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            run_as_future=True,
        )
    except Exception as e:
        print(f"  Hub upload failed for {path_in_repo}: {e}")


def _hub_upload_logs(api, repo_id, log_dir):
    """Upload TensorBoard log directory to HF Hub (async)."""
    if api is None:
        return
    try:
        api.upload_folder(
            folder_path=str(log_dir),
            path_in_repo="logs",
            repo_id=repo_id,
            run_as_future=True,
        )
    except Exception as e:
        print(f"  Hub log upload failed: {e}")


def train(cfg, resume=None, dummy=False):
    # --- Accelerator setup ---
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.grad_accum_steps,
        mixed_precision="bf16" if cfg.training.amp else "no",
    )
    device = accelerator.device
    is_main = accelerator.is_main_process

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    if is_main:
        print(f"Device: {device}, num_processes: {accelerator.num_processes}")

    # Create directories
    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    log_dir = Path(cfg.paths.log_dir)
    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Clean stale TensorBoard event files from previous runs
        if not resume:
            stale = list(log_dir.glob("events.out.tfevents.*"))
            if stale:
                print(f"Cleaning {len(stale)} stale event file(s) from {log_dir}")
                for f in stale:
                    f.unlink()

    writer = SummaryWriter(log_dir) if is_main else None

    # HF Hub
    hub_api, hub_repo_id = None, None
    if is_main:
        hub_api, hub_repo_id = _init_hub_api(cfg)

    # Model — compile before prepare()
    raw_model = DeepVQEAEC.from_config(cfg)
    n_params = sum(p.numel() for p in raw_model.parameters())
    if is_main:
        print(f"Parameters: {n_params:,}")
    if device.type == "cuda":
        raw_model = torch.compile(raw_model)
        if is_main:
            print("Model compiled with torch.compile")

    # Loss
    criterion = DeepVQELoss.from_config(cfg)

    # Optimizer
    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    # Dataset
    if dummy:
        train_ds = DummyAECDataset(
            length=cfg.data.num_train,
            target_len=int(cfg.training.clip_length_sec * cfg.audio.sample_rate),
            n_fft=cfg.audio.n_fft,
            hop_length=cfg.audio.hop_length,
            delay_range=tuple(int(x) for x in cfg.data.delay_range),
        )
        val_ds = DummyAECDataset(
            length=cfg.data.num_val,
            target_len=int(cfg.training.clip_length_sec * cfg.audio.sample_rate),
            n_fft=cfg.audio.n_fft,
            hop_length=cfg.audio.hop_length,
            delay_range=tuple(int(x) for x in cfg.data.delay_range),
        )
    else:
        train_ds = AECDataset(cfg, split="train")
        val_ds = AECDataset(cfg, split="val")
        if is_main:
            print(f"Train: {len(train_ds)} clean files, Val: {len(val_ds)} examples")

    num_workers = cfg.training.num_workers
    pin = device.type == "cuda"

    def worker_init_fn(worker_id):
        """Seed Python random and numpy per worker to avoid duplicate examples."""
        worker_seed = torch.initial_seed() % 2**32
        random.seed(worker_seed + worker_id)
        np.random.seed(worker_seed + worker_id)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=pin,
        persistent_workers=num_workers > 0,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin,
        persistent_workers=num_workers > 0,
        worker_init_fn=worker_init_fn,
    )

    # --- Accelerate prepare ---
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        raw_model, optimizer, train_loader, val_loader,
    )

    # Move criterion to device (not wrapped by Accelerate since it has no parameters to optimize)
    criterion = criterion.to(device)

    steps_per_epoch = len(train_loader) // cfg.training.grad_accum_steps
    warmup_scheduler = get_warmup_scheduler(optimizer, cfg, steps_per_epoch)
    epoch_scheduler = get_epoch_scheduler(optimizer, cfg)

    schedulers = {"warmup": warmup_scheduler, "epoch": epoch_scheduler}

    # Resume
    start_epoch = 0
    best_val_loss = float("inf")
    if resume:
        start_epoch = load_checkpoint(resume, model, optimizer, schedulers,
                                      accelerator=accelerator)
        if is_main:
            print(f"Resumed from epoch {start_epoch}")

    global_step = start_epoch * steps_per_epoch
    warmup_done = start_epoch >= cfg.training.warmup_epochs
    patience = cfg.training.early_stop_patience
    min_delta = cfg.training.early_stop_min_delta
    epochs_without_improvement = 0

    for epoch in range(start_epoch, cfg.training.epochs):
        model.train()

        # Anneal AlignBlock temperature: linear decay from start to end
        t_start = cfg.model.align_temp_start
        t_end = cfg.model.align_temp_end
        t_epochs = cfg.model.align_temp_epochs
        if t_epochs > 0 and epoch < t_epochs:
            temperature = t_start + (t_end - t_start) * epoch / t_epochs
        else:
            temperature = t_end
        align = _unwrap(model, accelerator).align
        align.temperature = temperature
        if is_main:
            add_scalar_with_help(writer, "train/temperature", temperature, epoch)

        epoch_losses = {"total": 0, "plcmse": 0}
        if cfg.loss.mag_l1_weight > 0:
            epoch_losses["mag_l1"] = 0
        if cfg.loss.time_l1_weight > 0:
            epoch_losses["time_l1"] = 0
        if cfg.loss.sisdr_weight > 0:
            epoch_losses["sisdr"] = 0
        if cfg.loss.energy_preservation_weight > 0:
            epoch_losses["energy_pres"] = 0
        if cfg.loss.delay_weight > 0:
            epoch_losses["delay"] = 0
        if cfg.loss.entropy_weight > 0:
            epoch_losses["entropy"] = 0
        if cfg.loss.mask_reg_weight > 0:
            epoch_losses["mask_reg"] = 0
        epoch_delay_acc = 0
        n_batches = 0

        gn = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.training.epochs}",
                     disable=not is_main)
        for batch_idx, batch in enumerate(pbar):
            mic_stft = batch["mic_stft"].to(device, non_blocking=True)
            ref_stft = batch["ref_stft"].to(device, non_blocking=True)
            clean_stft = batch["clean_stft"].to(device, non_blocking=True)
            clean_wav = batch["clean_wav"].to(device, non_blocking=True)
            delay_samp = batch["delay_samples"].to(device, non_blocking=True)

            with accelerator.accumulate(model):
                enhanced, delay_dist, mask_raw = model(mic_stft, ref_stft, return_delay=True)
                loss, components = criterion(enhanced, clean_stft, clean_wav)

                # Delay supervision loss
                delay_loss, delay_acc = compute_delay_loss(
                    delay_dist, delay_samp, cfg.audio.hop_length, cfg.model.dmax,
                )
                if cfg.loss.delay_weight > 0:
                    components["delay"] = delay_loss
                    loss = loss + cfg.loss.delay_weight * delay_loss

                # Entropy penalty on delay attention
                if cfg.loss.entropy_weight > 0:
                    entropy = compute_attention_entropy(delay_dist)
                    components["entropy"] = entropy
                    loss = loss + cfg.loss.entropy_weight * entropy

                # Mask magnitude regularizer
                if cfg.loss.mask_reg_weight > 0:
                    mask_reg = mask_magnitude_regularizer(mask_raw)
                    components["mask_reg"] = mask_reg
                    loss = loss + cfg.loss.mask_reg_weight * mask_reg

                # Update total so loss_ratio/* metrics use the true total
                components["total"] = loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    raw_gn = accelerator.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
                    gn = raw_gn.item() if hasattr(raw_gn, 'item') else raw_gn

                optimizer.step()
                if not warmup_done:
                    warmup_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1

                # Log per-step (rank 0 only)
                if is_main:
                    cur_lr = optimizer.param_groups[0]["lr"]
                    add_scalar_with_help(writer, "train/loss", components["total"].item(), global_step)
                    add_scalar_with_help(writer, "train/plcmse", components["plcmse"].item(), global_step)
                    add_scalar_with_help(writer, "train/delay_acc", delay_acc.item(), global_step)
                    add_scalar_with_help(writer, "train/lr", cur_lr, global_step)
                    add_scalar_with_help(writer, "train/grad_norm", gn, global_step)
                    log_per_layer_grad_norms(writer, _unwrap(model, accelerator), global_step)
                    # Log active auxiliary components
                    loss_weights = {"plcmse": cfg.loss.plcmse_weight}
                    for key, w in [("mag_l1", cfg.loss.mag_l1_weight),
                                   ("time_l1", cfg.loss.time_l1_weight),
                                   ("sisdr", cfg.loss.sisdr_weight),
                                   ("smooth_l1", cfg.loss.smooth_l1_weight),
                                   ("energy_pres", cfg.loss.energy_preservation_weight),
                                   ("delay", cfg.loss.delay_weight),
                                   ("entropy", cfg.loss.entropy_weight),
                                   ("mask_reg", cfg.loss.mask_reg_weight)]:
                        if w > 0 and key in components:
                            add_scalar_with_help(writer, f"train/{key}", components[key].item(), global_step)
                            loss_weights[key] = w
                    log_loss_ratios(writer, components, global_step, weights=loss_weights)

            for k in epoch_losses:
                epoch_losses[k] += components[k].item()
            epoch_delay_acc += delay_acc.item()
            n_batches += 1

            if is_main:
                pbar.set_postfix(
                    loss=f"{components['total'].item():.4f}",
                    dacc=f"{delay_acc.item():.0%}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )

        # Epoch averages
        if is_main:
            for k in epoch_losses:
                epoch_losses[k] /= max(n_batches, 1)
                add_scalar_with_help(writer, f"train_epoch/{k}", epoch_losses[k], epoch)
            epoch_delay_acc /= max(n_batches, 1)
            add_scalar_with_help(writer, "train_epoch/delay_acc", epoch_delay_acc, epoch)

            # Weight histograms every 5 epochs
            if (epoch + 1) % 5 == 0:
                log_weight_histograms(writer, _unwrap(model, accelerator), epoch)

        # Validation
        model.eval()
        val_loss_sum = torch.tensor(0.0, device=device)
        val_n = torch.tensor(0, device=device)
        val_delay_acc_sum = torch.tensor(0.0, device=device)
        val_erle_sum = torch.tensor(0.0, device=device)

        # Per-component sums for rank-0 logging
        val_comp_sums = {k: 0.0 for k in epoch_losses}
        val_sample_batch = None

        with torch.no_grad():
            for batch in val_loader:
                mic_stft = batch["mic_stft"].to(device, non_blocking=True)
                ref_stft = batch["ref_stft"].to(device, non_blocking=True)
                clean_stft = batch["clean_stft"].to(device, non_blocking=True)
                clean_wav = batch["clean_wav"].to(device, non_blocking=True)
                delay_samp = batch["delay_samples"].to(device, non_blocking=True)

                enhanced, delay_dist, mask_raw = model(mic_stft, ref_stft, return_delay=True)
                _, components = criterion(enhanced, clean_stft, clean_wav)

                # Delay accuracy (always computed as diagnostic)
                delay_loss, delay_acc = compute_delay_loss(
                    delay_dist, delay_samp, cfg.audio.hop_length, cfg.model.dmax,
                )

                # Add active auxiliary losses to total
                if cfg.loss.delay_weight > 0:
                    components["delay"] = delay_loss
                    components["total"] = components["total"] + cfg.loss.delay_weight * delay_loss
                if cfg.loss.entropy_weight > 0:
                    entropy = compute_attention_entropy(delay_dist)
                    components["entropy"] = entropy
                    components["total"] = components["total"] + cfg.loss.entropy_weight * entropy
                if cfg.loss.mask_reg_weight > 0:
                    mask_reg = mask_magnitude_regularizer(mask_raw)
                    components["mask_reg"] = mask_reg
                    components["total"] = components["total"] + cfg.loss.mask_reg_weight * mask_reg

                # ERLE
                length = clean_wav.shape[-1]
                mic_wav = batch["mic_wav"].to(device, non_blocking=True)
                enh_wav = istft(enhanced, cfg.audio.n_fft, cfg.audio.hop_length, length=length)
                erle = compute_erle(mic_wav, enh_wav, clean_wav)

                val_loss_sum += components["total"].detach()
                val_n += 1
                val_delay_acc_sum += delay_acc.detach()
                val_erle_sum += erle.detach()

                for k in val_comp_sums:
                    if k in components:
                        val_comp_sums[k] += components[k].item()

                if val_sample_batch is None and is_main:
                    val_sample_batch = batch

        # Gather validation metrics across all ranks
        val_loss_sum = accelerator.gather(val_loss_sum.unsqueeze(0)).sum()
        val_n_total = accelerator.gather(val_n.unsqueeze(0)).sum()
        val_delay_acc_sum = accelerator.gather(val_delay_acc_sum.unsqueeze(0)).sum()
        val_erle_sum = accelerator.gather(val_erle_sum.unsqueeze(0)).sum()

        avg_val_loss = (val_loss_sum / val_n_total).item()
        avg_val_delay_acc = (val_delay_acc_sum / val_n_total).item()
        avg_val_erle = (val_erle_sum / val_n_total).item()

        # Rank-0 logging and checkpointing
        if is_main:
            n_val_local = max(val_n.item(), 1)
            for k in val_comp_sums:
                val_comp_sums[k] /= n_val_local
                add_scalar_with_help(writer, f"val/{k}", val_comp_sums[k], epoch)
            add_scalar_with_help(writer, "val/delay_acc", avg_val_delay_acc, epoch)
            add_scalar_with_help(writer, "val/erle_db", avg_val_erle, epoch)

        # Step epoch scheduler (only after warmup) — all ranks
        if warmup_done:
            if cfg.training.lr_scheduler == "plateau":
                epoch_scheduler.step(avg_val_loss)
            else:
                epoch_scheduler.step()
        if not warmup_done and (epoch + 1) >= cfg.training.warmup_epochs:
            warmup_done = True

        if is_main:
            print(
                f"Epoch {epoch+1}: train_loss={epoch_losses['total']:.4f}, "
                f"val_loss={avg_val_loss:.4f}, "
                f"delay_acc={avg_val_delay_acc:.1%}, erle={avg_val_erle:+.1f}dB, "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

            # Log audio/spectrograms
            if val_sample_batch:
                log_audio_and_spectrograms(writer, model, val_sample_batch, epoch, cfg,
                                           device, accelerator=accelerator)

            # Checkpointing
            if (epoch + 1) % cfg.training.checkpoint_every == 0:
                ckpt_path = ckpt_dir / f"epoch_{epoch+1:04d}.pt"
                save_checkpoint(
                    model, optimizer, schedulers, epoch + 1,
                    avg_val_loss, ckpt_path, accelerator=accelerator,
                )
                manage_checkpoints(ckpt_dir, cfg.training.keep_checkpoints)
                if cfg.hub.push_checkpoints:
                    _hub_upload_file(hub_api, hub_repo_id, ckpt_path,
                                     f"checkpoints/{ckpt_path.name}")

            # Hub log uploads
            if hub_api and (epoch + 1) % cfg.hub.push_logs_every == 0:
                _hub_upload_logs(hub_api, hub_repo_id, log_dir)

        # Gate: check delay accuracy and ERLE minimums after warmup (all ranks)
        gate_epoch = max(cfg.training.warmup_epochs + 10, 20)

        if (epoch + 1) >= gate_epoch and avg_val_delay_acc < cfg.training.delay_acc_min:
            if is_main:
                print(
                    f"  FAIL: delay accuracy {avg_val_delay_acc:.1%} < "
                    f"{cfg.training.delay_acc_min:.0%} after {epoch+1} epochs. Stopping."
                )
            break

        if (epoch + 1) >= gate_epoch and avg_val_erle < cfg.training.erle_min_db:
            if is_main:
                print(
                    f"  FAIL: ERLE {avg_val_erle:+.1f}dB < "
                    f"{cfg.training.erle_min_db:+.1f}dB after {epoch+1} epochs. Stopping."
                )
            break

        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            if is_main:
                best_path = ckpt_dir / "best.pt"
                save_checkpoint(
                    model, optimizer, schedulers, epoch + 1,
                    avg_val_loss, best_path, accelerator=accelerator,
                )
                print(f"  New best val loss: {best_val_loss:.4f} "
                      f"(delay_acc={avg_val_delay_acc:.1%}, erle={avg_val_erle:+.1f}dB)")
                _hub_upload_file(hub_api, hub_repo_id, best_path,
                                 "checkpoints/best.pt")
        else:
            epochs_without_improvement += 1
            if is_main:
                print(
                    f"  No improvement for {epochs_without_improvement}/{patience} epochs"
                )
            if patience > 0 and epochs_without_improvement >= patience:
                if is_main:
                    print(f"Early stopping at epoch {epoch+1}.")
                break

    # End of training: final Hub uploads
    if is_main:
        if hub_api:
            _hub_upload_logs(hub_api, hub_repo_id, log_dir)
            best_path = ckpt_dir / "best.pt"
            if best_path.exists():
                _hub_upload_file(hub_api, hub_repo_id, best_path,
                                 "checkpoints/best.pt")
        if writer:
            writer.close()
        print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DeepVQE AEC")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from")
    parser.add_argument("--dummy", action="store_true", help="Use dummy dataset for testing")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, resume=args.resume, dummy=args.dummy)
