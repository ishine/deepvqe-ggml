"""DeepVQE AEC training script.

Features:
- AdamW optimizer with cosine annealing + warmup
- Mixed precision (AMP) with GradScaler
- Gradient accumulation for large effective batch sizes
- Gradient clipping
- TensorBoard logging (loss, lr, grad norms, audio, spectrograms, delay heatmaps)
- Checkpointing (last N + best)
"""

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.dataset import AECDataset, DummyAECDataset
from src.config import load_config
from src.losses import DeepVQELoss
from src.model import DeepVQEAEC
from src.stft import istft


def collate_fn(batch):
    """Custom collate that handles metadata dicts."""
    mic_stft = torch.stack([b["mic_stft"] for b in batch])
    ref_stft = torch.stack([b["ref_stft"] for b in batch])
    clean_stft = torch.stack([b["clean_stft"] for b in batch])
    clean_wav = torch.stack([b["clean_wav"] for b in batch])
    metadata = [b["metadata"] for b in batch]
    return {
        "mic_stft": mic_stft,
        "ref_stft": ref_stft,
        "clean_stft": clean_stft,
        "clean_wav": clean_wav,
        "metadata": metadata,
    }


def get_lr_scheduler(optimizer, cfg, steps_per_epoch):
    """Cosine annealing with linear warmup."""
    warmup_steps = cfg.training.warmup_epochs * steps_per_epoch
    total_steps = cfg.training.epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def grad_norm(model):
    """Compute total gradient norm."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total**0.5


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, loss, path):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "loss": loss,
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None):
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler and ckpt.get("scaler_state_dict"):
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt["epoch"]


def manage_checkpoints(ckpt_dir, keep_n):
    """Keep only the last N checkpoints (excluding best)."""
    ckpts = sorted(ckpt_dir.glob("epoch_*.pt"), key=lambda p: p.stat().st_mtime)
    while len(ckpts) > keep_n:
        ckpts.pop(0).unlink()


def log_audio_and_spectrograms(writer, model, val_batch, epoch, cfg, device):
    """Log audio samples and spectrograms to TensorBoard."""
    model.eval()
    with torch.no_grad():
        mic_stft = val_batch["mic_stft"][:1].to(device)
        ref_stft = val_batch["ref_stft"][:1].to(device)
        clean_stft = val_batch["clean_stft"][:1].to(device)

        enhanced, delay_dist = model(mic_stft, ref_stft, return_delay=True)
        length = val_batch["clean_wav"].shape[-1]

        mic_wav = istft(mic_stft, cfg.audio.n_fft, cfg.audio.hop_length, length=length)
        enh_wav = istft(enhanced, cfg.audio.n_fft, cfg.audio.hop_length, length=length)
        clean_wav = val_batch["clean_wav"][:1].to(device)

        sr = cfg.audio.sample_rate
        writer.add_audio("audio/mic", mic_wav[0].cpu(), epoch, sample_rate=sr)
        writer.add_audio("audio/enhanced", enh_wav[0].cpu(), epoch, sample_rate=sr)
        writer.add_audio("audio/clean", clean_wav[0].cpu(), epoch, sample_rate=sr)

        # Log delay distribution heatmap
        if delay_dist is not None:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 1, figsize=(10, 4))
            ax.imshow(
                delay_dist[0].cpu().numpy().T,
                aspect="auto",
                origin="lower",
                cmap="viridis",
            )
            ax.set_xlabel("Frame")
            ax.set_ylabel("Delay (frames)")
            ax.set_title("Delay Distribution")
            writer.add_figure("delay_distribution", fig, epoch)
            plt.close(fig)
    model.train()


def train(cfg, resume=None, dummy=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create directories
    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg.paths.log_dir)

    writer = SummaryWriter(log_dir)

    # Model
    model = DeepVQEAEC.from_config(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Loss
    criterion = DeepVQELoss.from_config(cfg).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
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
            delay_samples=160,
        )
        val_ds = DummyAECDataset(
            length=cfg.data.num_val,
            target_len=int(cfg.training.clip_length_sec * cfg.audio.sample_rate),
            n_fft=cfg.audio.n_fft,
            hop_length=cfg.audio.hop_length,
            delay_samples=160,
        )
    else:
        train_ds = AECDataset(cfg, split="train")
        val_ds = AECDataset(cfg, split="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    steps_per_epoch = len(train_loader) // cfg.training.grad_accum_steps
    scheduler = get_lr_scheduler(optimizer, cfg, steps_per_epoch)

    # AMP
    use_amp = cfg.training.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    autocast_ctx = lambda: torch.autocast(
        device_type=device.type,
        dtype=torch.float16 if use_amp else torch.bfloat16,
        enabled=use_amp or device.type == "cpu",
    )

    # Resume
    start_epoch = 0
    best_val_loss = float("inf")
    if resume:
        start_epoch = load_checkpoint(resume, model, optimizer, scheduler, scaler)
        print(f"Resumed from epoch {start_epoch}")

    global_step = start_epoch * steps_per_epoch
    accum_steps = cfg.training.grad_accum_steps

    for epoch in range(start_epoch, cfg.training.epochs):
        model.train()
        epoch_losses = {"total": 0, "plcmse": 0, "mag_l1": 0, "time_l1": 0}
        n_batches = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.training.epochs}")
        for batch_idx, batch in enumerate(pbar):
            mic_stft = batch["mic_stft"].to(device)
            ref_stft = batch["ref_stft"].to(device)
            clean_stft = batch["clean_stft"].to(device)
            clean_wav = batch["clean_wav"].to(device)

            with autocast_ctx():
                enhanced = model(mic_stft, ref_stft)
                loss, components = criterion(enhanced, clean_stft, clean_wav)
                loss = loss / accum_steps

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % accum_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)

                gn = grad_norm(model)

                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Log per-step
                writer.add_scalar("train/loss", components["total"].item(), global_step)
                writer.add_scalar("train/plcmse", components["plcmse"].item(), global_step)
                writer.add_scalar("train/mag_l1", components["mag_l1"].item(), global_step)
                writer.add_scalar("train/time_l1", components["time_l1"].item(), global_step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
                writer.add_scalar("train/grad_norm", gn, global_step)

            for k in epoch_losses:
                epoch_losses[k] += components[k].item()
            n_batches += 1

            pbar.set_postfix(
                loss=f"{components['total'].item():.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        # Epoch averages
        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches, 1)
            writer.add_scalar(f"train_epoch/{k}", epoch_losses[k], epoch)

        # Validation
        model.eval()
        val_losses = {"total": 0, "plcmse": 0, "mag_l1": 0, "time_l1": 0}
        n_val = 0
        val_sample_batch = None

        with torch.no_grad():
            for batch in val_loader:
                mic_stft = batch["mic_stft"].to(device)
                ref_stft = batch["ref_stft"].to(device)
                clean_stft = batch["clean_stft"].to(device)
                clean_wav = batch["clean_wav"].to(device)

                enhanced = model(mic_stft, ref_stft)
                _, components = criterion(enhanced, clean_stft, clean_wav)

                for k in val_losses:
                    val_losses[k] += components[k].item()
                n_val += 1

                if val_sample_batch is None:
                    val_sample_batch = batch

        for k in val_losses:
            val_losses[k] /= max(n_val, 1)
            writer.add_scalar(f"val/{k}", val_losses[k], epoch)

        print(
            f"Epoch {epoch+1}: train_loss={epoch_losses['total']:.4f}, "
            f"val_loss={val_losses['total']:.4f}"
        )

        # Log audio/spectrograms
        if val_sample_batch:
            log_audio_and_spectrograms(writer, model, val_sample_batch, epoch, cfg, device)

        # Checkpointing
        if (epoch + 1) % cfg.training.checkpoint_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch + 1,
                val_losses["total"],
                ckpt_dir / f"epoch_{epoch+1:04d}.pt",
            )
            manage_checkpoints(ckpt_dir, cfg.training.keep_checkpoints)

        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch + 1,
                val_losses["total"],
                ckpt_dir / "best.pt",
            )
            print(f"  New best val loss: {best_val_loss:.4f}")

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
