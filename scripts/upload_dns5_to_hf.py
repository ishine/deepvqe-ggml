#!/usr/bin/env python3
"""Resample DNS5 data to 16kHz FLAC and upload to HuggingFace Hub.

One-time script to prepare DNS5 data for HF Jobs training.
Reads 48kHz WAV files, resamples to 16kHz, saves as FLAC (lossless,
~6x smaller than 48kHz WAV), and uploads to an HF dataset.

Usage:
    # Default: reads from ./datasets_fullband, uploads to richiejp/dns5-16k
    python scripts/upload_dns5_to_hf.py

    # Custom paths:
    python scripts/upload_dns5_to_hf.py --source /path/to/datasets_fullband \
        --repo richiejp/dns5-16k --staging /tmp/dns5-16k

    # Resample only (no upload):
    python scripts/upload_dns5_to_hf.py --no-upload

    # Upload only (already resampled):
    python scripts/upload_dns5_to_hf.py --no-resample --staging /path/to/resampled

Prerequisites:
    pip install soundfile scipy huggingface_hub tqdm
    huggingface-cli login
"""

import argparse
import os
import sys
from math import gcd
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm

TARGET_SR = 16000

# Mapping from source subdirectory names to output names
# (the Docker entrypoint expects clean/, noise/, impulse_responses/)
SUBDIR_MAP = {
    "clean_fullband": "clean",
    "noise_fullband": "noise",
    "impulse_responses": "impulse_responses",
}


def resample_file(args):
    """Resample a single audio file to 16kHz FLAC. Worker function for Pool."""
    src_path, dst_path, target_sr = args
    try:
        info = sf.info(src_path)
        file_sr = info.samplerate

        audio, _ = sf.read(src_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]  # mono

        if file_sr != target_sr:
            g = gcd(target_sr, file_sr)
            audio = resample_poly(audio, target_sr // g, file_sr // g).astype(np.float32)

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(dst_path), audio, target_sr, format="FLAC")
        return None  # success
    except Exception as e:
        return f"{src_path}: {e}"


def collect_audio_files(directory):
    """Recursively collect .wav and .flac files."""
    files = []
    for ext in ("*.wav", "*.flac"):
        files.extend(Path(directory).rglob(ext))
    files.sort()
    return files


def resample_subdir(src_dir, dst_dir, subdir_name, dst_name, workers):
    """Resample all audio files in a subdirectory."""
    src = src_dir / subdir_name
    if not src.exists():
        print(f"  SKIP: {src} does not exist")
        return 0

    dst = dst_dir / dst_name
    files = collect_audio_files(src)
    if not files:
        print(f"  SKIP: no audio files in {src}")
        return 0

    # Build work items, preserving relative paths
    work = []
    skipped = 0
    for f in files:
        rel = f.relative_to(src)
        # Change extension to .flac
        dst_path = dst / rel.with_suffix(".flac")
        if dst_path.exists():
            skipped += 1
            continue
        work.append((str(f), dst_path, TARGET_SR))

    if skipped:
        print(f"  {dst_name}: {skipped} files already resampled, {len(work)} remaining")
    else:
        print(f"  {dst_name}: {len(files)} files to resample")

    if not work:
        return len(files)

    errors = []
    with Pool(workers) as pool:
        for result in tqdm(pool.imap_unordered(resample_file, work),
                           total=len(work), desc=f"  {dst_name}", unit="file"):
            if result is not None:
                errors.append(result)

    if errors:
        print(f"  WARNING: {len(errors)} errors:")
        for e in errors[:10]:
            print(f"    {e}")
        if len(errors) > 10:
            print(f"    ... and {len(errors) - 10} more")

    return len(files) - len(errors)


def upload_to_hub(staging_dir, repo_id):
    """Upload the resampled dataset to HuggingFace Hub."""
    from huggingface_hub import HfApi

    api = HfApi()

    # Create the dataset repo
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)

    # Write a README
    readme = f"""---
license: cc-by-4.0
task_categories:
  - audio-classification
tags:
  - speech
  - noise
  - room-impulse-response
  - acoustic-echo-cancellation
  - dns-challenge
pretty_name: DNS5 16kHz (resampled)
---

# DNS5 16kHz

Resampled subset of the [ICASSP 2022 DNS Challenge](https://github.com/microsoft/DNS-Challenge) dataset.

All audio files resampled from 48kHz to **16kHz** and stored as **FLAC** (lossless compression).

## Structure

```
clean/              # Clean speech (from VCTK and other corpora)
noise/              # Environmental noise (AudioSet, Freesound)
impulse_responses/  # Room impulse responses
```

## Usage

```python
from huggingface_hub import snapshot_download
snapshot_download("{repo_id}", local_dir="/data/dns5", repo_type="dataset")
```

## Source

Original data from Microsoft's DNS Challenge:
- https://github.com/microsoft/DNS-Challenge
- License: CC-BY-4.0 (see original repo for details)
"""
    readme_path = staging_dir / "README.md"
    readme_path.write_text(readme)

    print(f"Uploading to {repo_id}...")
    print("  This may take a while for large datasets. Progress shown per-folder.")

    # Upload each subdirectory separately to show progress
    for subdir in ["clean", "noise", "impulse_responses"]:
        subdir_path = staging_dir / subdir
        if not subdir_path.exists():
            continue
        n_files = sum(1 for _ in subdir_path.rglob("*") if _.is_file())
        print(f"  Uploading {subdir}/ ({n_files} files)...")
        api.upload_folder(
            folder_path=str(subdir_path),
            path_in_repo=subdir,
            repo_id=repo_id,
            repo_type="dataset",
        )

    # Upload README
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )

    print(f"Done! Dataset available at: https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Resample DNS5 to 16kHz and upload to HF Hub")
    parser.add_argument("--source", type=str, default="datasets_fullband",
                        help="Source directory with DNS5 data (default: ./datasets_fullband)")
    parser.add_argument("--staging", type=str, default="datasets_fullband/dns5_16k",
                        help="Staging directory for resampled files")
    parser.add_argument("--repo", type=str, default="richiejp/dns5-16k",
                        help="HuggingFace dataset repo ID")
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 2),
                        help=f"Number of parallel workers (default: {max(1, cpu_count() - 2)})")
    parser.add_argument("--no-upload", action="store_true",
                        help="Resample only, skip upload")
    parser.add_argument("--no-resample", action="store_true",
                        help="Upload only (staging dir must exist)")
    args = parser.parse_args()

    src_dir = Path(args.source)
    staging_dir = Path(args.staging)

    if not args.no_resample:
        if not src_dir.exists():
            print(f"ERROR: Source directory {src_dir} does not exist")
            print("Run ./scripts/download_dns5_minimal.sh first")
            sys.exit(1)

        print(f"Resampling DNS5 from {src_dir} to {staging_dir} at {TARGET_SR} Hz")
        print(f"Using {args.workers} workers\n")

        total = 0
        for src_name, dst_name in SUBDIR_MAP.items():
            n = resample_subdir(src_dir, staging_dir, src_name, dst_name, args.workers)
            total += n

        print(f"\nResampling complete: {total} files in {staging_dir}")

        # Summary of disk usage
        total_bytes = sum(f.stat().st_size for f in staging_dir.rglob("*") if f.is_file())
        print(f"Total size: {total_bytes / 1e9:.1f} GB")

    if not args.no_upload:
        if not staging_dir.exists():
            print(f"ERROR: Staging directory {staging_dir} does not exist")
            sys.exit(1)
        upload_to_hub(staging_dir, args.repo)


if __name__ == "__main__":
    main()
