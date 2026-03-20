#!/usr/bin/env python3
"""Resample DNS5 data to 16kHz FLAC and upload to HuggingFace Hub in batches.

Streams data: resamples a batch of files to a temp directory, uploads that
batch as a single HF commit, deletes the local copies, then repeats. Only
needs disk space for one batch (~500 files) at a time.

Progress is tracked in a manifest file so the script can resume after
interruption — already-uploaded files are skipped on restart.

Usage:
    # Default: reads from ./datasets_fullband, uploads to richiejp/dns5-16k
    python scripts/upload_dns5_to_hf.py

    # Custom paths and batch size:
    python scripts/upload_dns5_to_hf.py --source /path/to/datasets_fullband \
        --repo richiejp/dns5-16k --batch-size 200

    # Dry run (resample first batch only, no upload):
    python scripts/upload_dns5_to_hf.py --dry-run

Prerequisites:
    pip install soundfile scipy huggingface_hub tqdm
    huggingface-cli login
"""

import argparse
import json
import shutil
import sys
import tempfile
from math import gcd
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm

TARGET_SR = 16000

# Mapping from source subdirectory names to repo path prefixes
# (the training script expects clean/, noise/, impulse_responses/)
SUBDIR_MAP = {
    "clean_fullband": "clean",
    "noise_fullband": "noise",
    "impulse_responses": "impulse_responses",
}


def resample_one(args):
    """Resample a single file to 16kHz FLAC. Returns (repo_path, local_path) or error string."""
    src_path, dst_path, target_sr = args
    try:
        info = sf.info(src_path)
        file_sr = info.samplerate

        audio, _ = sf.read(src_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]

        if file_sr != target_sr:
            g = gcd(target_sr, file_sr)
            audio = resample_poly(audio, target_sr // g, file_sr // g).astype(np.float32)

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(dst_path), audio, target_sr, format="FLAC")
        return None
    except Exception as e:
        return f"{src_path}: {e}"


def collect_audio_files(directory):
    """Recursively collect .wav and .flac files, sorted."""
    files = []
    for ext in ("*.wav", "*.flac"):
        files.extend(Path(directory).rglob(ext))
    files.sort()
    return files


class Manifest:
    """Tracks uploaded files for resume support."""

    def __init__(self, path):
        self.path = Path(path)
        self.uploaded = set()
        if self.path.exists():
            data = json.loads(self.path.read_text())
            self.uploaded = set(data.get("uploaded", []))

    def mark_batch(self, repo_paths):
        self.uploaded.update(repo_paths)
        self._save()

    def is_uploaded(self, repo_path):
        return repo_path in self.uploaded

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "uploaded": sorted(self.uploaded),
        }))

    def __len__(self):
        return len(self.uploaded)


def build_work_list(src_dir, manifest):
    """Build list of (src_path, repo_path) for all files not yet uploaded."""
    work = []
    for src_name, dst_name in SUBDIR_MAP.items():
        src_sub = src_dir / src_name
        if not src_sub.exists():
            print(f"  SKIP: {src_sub} does not exist")
            continue
        files = collect_audio_files(src_sub)
        for f in files:
            rel = f.relative_to(src_sub)
            repo_path = f"{dst_name}/{rel.with_suffix('.flac')}"
            if not manifest.is_uploaded(repo_path):
                work.append((str(f), repo_path))
    return work


def process_and_upload_batch(batch, tmp_dir, repo_id, api, workers, manifest, dry_run):
    """Resample a batch, upload as one commit, clean up."""
    from huggingface_hub import CommitOperationAdd

    # Resample
    resample_work = []
    for src_path, repo_path in batch:
        dst_path = Path(tmp_dir) / repo_path
        resample_work.append((src_path, dst_path, TARGET_SR))

    errors = []
    with Pool(min(workers, len(resample_work))) as pool:
        for result in pool.imap_unordered(resample_one, resample_work):
            if result is not None:
                errors.append(result)

    if errors:
        print(f"    {len(errors)} resample errors (skipping those files):")
        for e in errors[:5]:
            print(f"      {e}")

    # Build commit operations for successfully resampled files
    operations = []
    repo_paths = []
    for src_path, repo_path in batch:
        local = Path(tmp_dir) / repo_path
        if local.exists():
            operations.append(CommitOperationAdd(
                path_in_repo=repo_path,
                path_or_fileobj=str(local),
            ))
            repo_paths.append(repo_path)

    if not operations:
        return 0

    if dry_run:
        print(f"    [dry run] Would upload {len(operations)} files")
        # Still mark them so --dry-run + resume works for testing
        return len(operations)

    # Upload as single commit
    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message=f"Add {len(operations)} files ({repo_paths[0].split('/')[0]})",
    )

    # Mark uploaded and clean up
    manifest.mark_batch(repo_paths)

    # Delete resampled files from tmp
    for repo_path in repo_paths:
        local = Path(tmp_dir) / repo_path
        local.unlink(missing_ok=True)

    return len(operations)


def upload_readme(repo_id, api):
    """Create/update the dataset README."""
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
    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Resample DNS5 to 16kHz FLAC and upload to HF Hub in batches")
    parser.add_argument("--source", type=str, default="datasets_fullband",
                        help="Source directory with DNS5 data (default: ./datasets_fullband)")
    parser.add_argument("--repo", type=str, default="richiejp/dns5-16k",
                        help="HuggingFace dataset repo ID")
    parser.add_argument("--batch-size", type=int, default=500,
                        help="Files per upload batch (default: 500)")
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 2),
                        help=f"Parallel resample workers (default: {max(1, cpu_count() - 2)})")
    parser.add_argument("--manifest", type=str, default=".cache/dns5_upload_manifest.json",
                        help="Progress tracking file for resume")
    parser.add_argument("--dry-run", action="store_true",
                        help="Resample first batch only, don't upload")
    args = parser.parse_args()

    src_dir = Path(args.source)
    if not src_dir.exists():
        print(f"ERROR: Source directory {src_dir} does not exist")
        print("Run ./scripts/download_dns5_minimal.sh first")
        sys.exit(1)

    manifest = Manifest(args.manifest)
    if len(manifest):
        print(f"Resuming: {len(manifest)} files already uploaded")

    # Build work list (skipping already-uploaded files)
    print(f"Scanning {src_dir} for audio files...")
    work = build_work_list(src_dir, manifest)
    if not work:
        print("All files already uploaded!")
        return

    print(f"{len(work)} files to process in batches of {args.batch_size}")
    n_batches = (len(work) + args.batch_size - 1) // args.batch_size
    print(f"{n_batches} batches total\n")

    # Set up HF API
    api = None
    if not args.dry_run:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.repo, repo_type="dataset", exist_ok=True)
        upload_readme(args.repo, api)
        print(f"Uploading to: https://huggingface.co/datasets/{args.repo}\n")

    total_uploaded = 0
    with tempfile.TemporaryDirectory(prefix="dns5_upload_") as tmp_dir:
        for batch_idx in range(n_batches):
            start = batch_idx * args.batch_size
            end = min(start + args.batch_size, len(work))
            batch = work[start:end]

            # Show which subdir this batch is from
            subdirs = set(rp.split("/")[0] for _, rp in batch)
            print(f"Batch {batch_idx + 1}/{n_batches}: {len(batch)} files ({', '.join(sorted(subdirs))})")

            n = process_and_upload_batch(
                batch, tmp_dir, args.repo, api, args.workers, manifest, args.dry_run)
            total_uploaded += n
            print(f"    Uploaded {n} files (total: {total_uploaded + len(manifest) - n})")

            if args.dry_run:
                print("\n[dry run] Stopping after first batch")
                break

    print(f"\nDone! {total_uploaded} files uploaded this run, "
          f"{len(manifest)} total in manifest.")
    if not args.dry_run:
        print(f"Dataset: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
