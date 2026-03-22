#!/usr/bin/env python3
"""Resample DNS5 data to 16kHz FLAC and upload to HuggingFace Hub as tar shards.

Each batch of files is packed into a tar archive. Multiple shards are grouped
into a single HF commit to stay under the 128 commits/hour rate limit.

Shards are named: {category}/shard_{NNNN}.tar
E.g.: clean/shard_0000.tar, noise/shard_0012.tar

Progress is tracked in a manifest file so the script can resume after
interruption — already-uploaded files are skipped on restart.

Usage:
    python scripts/upload_dns5_to_hf.py --source /path/to/datasets_fullband
    python scripts/upload_dns5_to_hf.py --source /path --batch-size 2000
    python scripts/upload_dns5_to_hf.py --dry-run

Prerequisites:
    pip install soundfile scipy huggingface_hub
    huggingface-cli login
"""

import argparse
import json
import shutil
import sys
import tarfile
import tempfile
from math import gcd
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 16000

# Max shards to group into a single HF commit (avoids 128 commits/hr limit)
SHARDS_PER_COMMIT = 100

# Mapping from source subdirectory names to repo path prefixes
# (the training script expects clean/, noise/, impulse_responses/)
SUBDIR_MAP = {
    "clean_fullband": "clean",
    "noise_fullband": "noise",
    "impulse_responses": "impulse_responses",
}


def resample_one(args):
    """Resample a single file to 16kHz FLAC."""
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
    """Tracks uploaded files and shard counts for resume support."""

    def __init__(self, path):
        self.path = Path(path)
        self.uploaded = set()
        self.shard_counts = {}  # category -> next shard index
        if self.path.exists():
            data = json.loads(self.path.read_text())
            self.uploaded = set(data.get("uploaded", []))
            self.shard_counts = data.get("shard_counts", {})

    def mark_shards(self, repo_paths, category, n_shards):
        self.uploaded.update(repo_paths)
        self.shard_counts[category] = self.shard_counts.get(category, 0) + n_shards
        self._save()

    def next_shard(self, category):
        return self.shard_counts.get(category, 0)

    def is_uploaded(self, repo_path):
        return repo_path in self.uploaded

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "uploaded": sorted(self.uploaded),
            "shard_counts": self.shard_counts,
        }))

    def __len__(self):
        return len(self.uploaded)


def build_work_list(src_dir, manifest):
    """Build per-category work lists for files not yet uploaded."""
    work = {}
    for src_name, dst_name in SUBDIR_MAP.items():
        src_sub = src_dir / src_name
        if not src_sub.exists():
            print(f"  SKIP: {src_sub} does not exist")
            continue
        files = collect_audio_files(src_sub)
        items = []
        for f in files:
            rel = f.relative_to(src_sub)
            repo_path = f"{dst_name}/{rel.with_suffix('.flac')}"
            if not manifest.is_uploaded(repo_path):
                items.append((str(f), repo_path))
        if items:
            work[dst_name] = items
    return work


def create_shard(batch, category, shard_idx, tmp_dir, workers):
    """Resample a batch of files and pack into a tar shard.

    Returns (shard_name, tar_path, repo_paths) or None on failure.
    """
    resampled_dir = Path(tmp_dir) / "resampled"

    # Resample
    resample_work = []
    for src_path, repo_path in batch:
        inner_path = repo_path[len(category) + 1:]
        dst_path = resampled_dir / inner_path
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

    # Pack into tar
    shard_name = f"{category}/shard_{shard_idx:04d}.tar"
    tar_path = Path(tmp_dir) / "tars" / shard_name
    tar_path.parent.mkdir(parents=True, exist_ok=True)

    repo_paths = []
    with tarfile.open(tar_path, "w") as tf:
        for src_path, repo_path in batch:
            inner_path = repo_path[len(category) + 1:]
            local = resampled_dir / inner_path
            if local.exists():
                tf.add(str(local), arcname=inner_path)
                repo_paths.append(repo_path)

    # Clean up resampled files (keep tar)
    shutil.rmtree(str(resampled_dir), ignore_errors=True)

    if not repo_paths:
        tar_path.unlink(missing_ok=True)
        return None

    return shard_name, tar_path, repo_paths


def upload_shard_group(group, repo_id, api, manifest, category, dry_run):
    """Upload a group of tar shards in a single HF commit."""
    from huggingface_hub import CommitOperationAdd

    total_size = sum(p.stat().st_size for _, p, _ in group)
    total_files = sum(len(rp) for _, _, rp in group)
    size_mb = total_size / (1024 * 1024)
    shard_names = [name for name, _, _ in group]

    if dry_run:
        print(f"    [dry run] Would upload {len(group)} shards "
              f"({total_files} files, {size_mb:.1f} MB)")
        return

    print(f"    Uploading {len(group)} shards ({total_files} files, "
          f"{size_mb:.1f} MB) ...")

    operations = [
        CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path))
        for name, path, _ in group
    ]

    first = shard_names[0].split("/")[-1]
    last = shard_names[-1].split("/")[-1]
    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message=f"Add {category}/{first}..{last} ({total_files} files)",
    )

    # Mark all files as uploaded
    all_repo_paths = []
    for _, _, rp in group:
        all_repo_paths.extend(rp)
    manifest.mark_shards(all_repo_paths, category, len(group))

    # Clean up tar files
    for _, path, _ in group:
        path.unlink(missing_ok=True)


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

All audio files resampled from 48kHz to **16kHz** and stored as **FLAC** (lossless compression),
packed into tar shards.

## Structure

```
clean/shard_0000.tar          # Clean speech (VCTK and other corpora)
clean/shard_0001.tar
...
noise/shard_0000.tar          # Environmental noise (AudioSet, Freesound)
...
impulse_responses/shard_0000.tar  # Room impulse responses
...
```

Each tar contains FLAC files with their original directory structure preserved.

## Usage

```python
from huggingface_hub import snapshot_download
import tarfile
from pathlib import Path

# Download
local = snapshot_download("{repo_id}", local_dir="/data/dns5", repo_type="dataset")

# Extract all shards
for tar_path in sorted(Path(local).rglob("*.tar")):
    with tarfile.open(tar_path) as tf:
        tf.extractall(tar_path.parent)
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
        description="Resample DNS5 to 16kHz FLAC tar shards and upload to HF Hub")
    parser.add_argument("--source", type=str, default="datasets_fullband",
                        help="Source directory with DNS5 data")
    parser.add_argument("--repo", type=str, default="richiejp/dns5-16k",
                        help="HuggingFace dataset repo ID")
    parser.add_argument("--batch-size", type=int, default=5000,
                        help="Files per tar shard (default: 5000)")
    parser.add_argument("--shards-per-commit", type=int,
                        default=SHARDS_PER_COMMIT,
                        help=f"Shards per HF commit (default: {SHARDS_PER_COMMIT})")
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 2),
                        help="Parallel resample workers")
    parser.add_argument("--manifest", type=str,
                        default=".cache/dns5_upload_manifest.json",
                        help="Progress tracking file for resume")
    parser.add_argument("--dry-run", action="store_true",
                        help="Resample first batch only, don't upload")
    args = parser.parse_args()

    src_dir = Path(args.source)
    if not src_dir.exists():
        print(f"ERROR: Source directory {src_dir} does not exist")
        print("Run ./scripts/upload_dns5_full.sh to download and upload")
        sys.exit(1)

    manifest = Manifest(args.manifest)
    if len(manifest):
        print(f"Resuming: {len(manifest)} files already uploaded")
        for cat, count in manifest.shard_counts.items():
            print(f"  {cat}: {count} shards")

    # Build per-category work lists
    print(f"Scanning {src_dir} for audio files...")
    work = build_work_list(src_dir, manifest)
    if not work:
        print("All files already uploaded!")
        return

    total_files = sum(len(v) for v in work.values())
    print(f"{total_files} files to process across {len(work)} categories")
    print(f"Shard size: {args.batch_size} files per tar")
    print(f"Shards per commit: {args.shards_per_commit}\n")

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
        for category, items in work.items():
            n_shards = (len(items) + args.batch_size - 1) // args.batch_size
            shard_start = manifest.next_shard(category)
            n_commits = (n_shards + args.shards_per_commit - 1) // args.shards_per_commit
            print(f"=== {category}: {len(items)} files -> "
                  f"{n_shards} shards -> {n_commits} commits ===")

            pending_group = []

            for i in range(n_shards):
                start = i * args.batch_size
                end = min(start + args.batch_size, len(items))
                batch = items[start:end]
                shard_idx = shard_start + i

                print(f"  Shard {shard_idx:04d}: {len(batch)} files")

                result = create_shard(
                    batch, category, shard_idx, tmp_dir, args.workers)
                if result is None:
                    continue

                pending_group.append(result)

                # Upload when group is full or this is the last shard
                if (len(pending_group) >= args.shards_per_commit
                        or i == n_shards - 1):
                    upload_shard_group(
                        pending_group, args.repo, api, manifest,
                        category, args.dry_run)
                    total_uploaded += sum(len(rp) for _, _, rp in pending_group)
                    pending_group = []

                    if args.dry_run:
                        print("\n[dry run] Stopping after first group")
                        return

            print()

    print(f"Done! {total_uploaded} files uploaded this run, "
          f"{len(manifest)} total in manifest.")
    if not args.dry_run:
        print(f"Dataset: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
