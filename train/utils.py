"""Shared utilities for DeepVQE training, evaluation, and export."""

import torch


def _unwrap(model):
    """Get the raw model, unwrapping torch.compile's OptimizedModule."""
    return getattr(model, "_orig_mod", model)


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


def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": _unwrap(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, weights_only=False)
    state = ckpt["model_state_dict"]
    # Strip _orig_mod. prefix for backward compat with old checkpoints
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    _unwrap(model).load_state_dict(state)
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt["epoch"]
