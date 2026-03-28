#!/usr/bin/env python3
"""Download and extract the DNS5 16kHz FLAC dataset from HuggingFace.

Downloads tar shards from richiejp/dns5-16k, extracts FLAC files into
datasets_fullband/{clean,noise,impulse_responses}/.

Usage:
    python scripts/download_dns5_hf.py
    python scripts/download_dns5_hf.py --keep-tars
    python scripts/download_dns5_hf.py --output /workspace/deepvqe/datasets_fullband
"""

import argparse
import tarfile
from pathlib import Path

from huggingface_hub import snapshot_download
from tqdm import tqdm


def extract_shards(data_dir: Path, keep_tars: bool = False):
    """Extract all tar shards and optionally remove them."""
    tar_files = sorted(data_dir.rglob("*.tar"))
    if not tar_files:
        print("No tar files found to extract.")
        return

    print(f"Extracting {len(tar_files)} tar shards...")
    for tar_path in tqdm(tar_files, desc="Extracting"):
        marker = tar_path.with_suffix(".extracted")
        if marker.exists():
            continue
        with tarfile.open(tar_path) as tf:
            tf.extractall(tar_path.parent, filter="data")
        marker.touch()
        if not keep_tars:
            tar_path.unlink()


def count_files(data_dir: Path):
    """Print file counts per subdirectory."""
    for subdir in ["clean", "noise", "impulse_responses"]:
        d = data_dir / subdir
        if d.exists():
            flac_count = 0
            wav_count = 0
            for f in d.rglob("*"):
                if f.suffix == ".flac":
                    flac_count += 1
                elif f.suffix == ".wav":
                    wav_count += 1
            total = flac_count + wav_count
            print(f"  {subdir}: {total} audio files ({flac_count} flac, {wav_count} wav)")
        else:
            print(f"  {subdir}: not found")


def main():
    parser = argparse.ArgumentParser(description="Download DNS5 16kHz from HuggingFace")
    parser.add_argument("--output", default="/workspace/deepvqe/datasets_fullband",
                        help="Output directory (default: /workspace/deepvqe/datasets_fullband)")
    parser.add_argument("--keep-tars", action="store_true",
                        help="Keep tar files after extraction")
    args = parser.parse_args()

    data_dir = Path(args.output)
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=== Downloading DNS5 16kHz dataset from HuggingFace ===")
    print(f"Output: {data_dir}")
    print()

    snapshot_download(
        "richiejp/dns5-16k",
        repo_type="dataset",
        local_dir=str(data_dir),
    )

    print()
    extract_shards(data_dir, keep_tars=args.keep_tars)

    print()
    print("=== Download complete ===")
    count_files(data_dir)


if __name__ == "__main__":
    main()
