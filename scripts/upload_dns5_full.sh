#!/usr/bin/env bash
# Download full DNS5 dataset piece by piece, upload each piece to HF Hub,
# then delete the local files. Only needs disk space for one corpus at a time.
#
# Processes in order of size (smallest first) so progress is visible quickly.
# Resume-safe: uses marker files for downloads and a manifest for uploads.
#
# Disk usage: peak ~350 GB (german_speech download + extraction)
# Total upload: ~890 GB of 48kHz WAV → ~150 GB of 16kHz FLAC on HF Hub
#
# Usage:
#   ./scripts/upload_dns5_full.sh [data_dir]
#   Default data_dir: ~/mnt/home2/data/deepvqe

set -euo pipefail

DATA_DIR="${1:-$HOME/mnt/home2/data/deepvqe}"
DSET="$DATA_DIR/datasets_fullband"
MANIFEST="$DATA_DIR/.dns5_upload_manifest.json"
REPO="richiejp/dns5-16k"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

DNS5="https://dnschallengepublic.blob.core.windows.net/dns5archive/V5_training_dataset"
DNS4="https://dns4public.blob.core.windows.net/dns4archive/datasets_fullband"

mkdir -p "$DSET"/{clean_fullband,noise_fullband,tmp}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

upload_new_files() {
    echo "  [upload] Uploading new files to HF Hub..."
    python "$SCRIPT_DIR/upload_dns5_to_hf.py" \
        --source "$DSET" \
        --repo "$REPO" \
        --manifest "$MANIFEST" \
        --batch-size 500
}

# Noise/IR tars extract with a datasets_fullband/ prefix inside the tar.
# Move files to the expected location.
normalize_paths() {
    if [ -d "$DSET/datasets_fullband/noise_fullband" ]; then
        cp -a "$DSET/datasets_fullband/noise_fullband/"* "$DSET/noise_fullband/" 2>/dev/null || true
        rm -rf "$DSET/datasets_fullband/noise_fullband"
    fi
    if [ -d "$DSET/datasets_fullband/impulse_responses" ]; then
        mkdir -p "$DSET/impulse_responses"
        cp -a "$DSET/datasets_fullband/impulse_responses/"* "$DSET/impulse_responses/" 2>/dev/null || true
        rm -rf "$DSET/datasets_fullband/impulse_responses"
    fi
    rmdir "$DSET/datasets_fullband" 2>/dev/null || true
}

download_noise_shard() {
    local shard="$1"
    local base_url="$2"
    local marker="$DSET/noise_fullband/.${shard}.done"

    if [ -f "$marker" ]; then
        echo "  [skip] $shard (already done)"
        return
    fi

    local url="${base_url}/noise_fullband/datasets_fullband.noise_fullband.${shard}.tar.bz2"
    echo "  [download+extract] $shard ..."
    curl -L --retry 3 "$url" | tar -C "$DSET" -xjf -
    normalize_paths
    touch "$marker"
}

download_speech_single() {
    local name="$1"
    local marker="$DSET/clean_fullband/.${name}.done"

    if [ -f "$marker" ]; then
        echo "  [skip] $name (already done)"
        return 1  # nothing new to upload
    fi

    local dest="$DSET/tmp/${name}.tgz"
    if [ ! -f "$dest" ]; then
        echo "  [download] $name ..."
        curl -L --retry 3 -C - "$DNS5/Track1_Headset/${name}.tgz" -o "$dest"
    fi

    echo "  [extract] $name ..."
    tar -C "$DSET/clean_fullband" -xzf "$dest"
    rm -f "$dest"
    touch "$marker"
    return 0
}

download_speech_split() {
    local name="$1"
    shift
    local parts=("$@")
    local marker="$DSET/clean_fullband/.${name}.done"

    if [ -f "$marker" ]; then
        echo "  [skip] $name (already done)"
        return 1
    fi

    for part in "${parts[@]}"; do
        local dest="$DSET/tmp/${name}.tgz.${part}"
        if [ -f "$dest" ]; then
            echo "  [skip] $name $part (already downloaded)"
        else
            echo "  [download] $name $part ..."
            curl -L --retry 3 -C - "$DNS5/Track1_Headset/${name}.tgz.${part}" -o "$dest"
        fi
    done

    echo "  [extract] $name (${#parts[@]} parts) ..."
    cat "$DSET"/tmp/${name}.tgz.part* | tar -C "$DSET/clean_fullband" -xzf -
    rm -f "$DSET"/tmp/${name}.tgz.part*
    touch "$marker"
    return 0
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

echo "=== DNS5 Full Upload to HuggingFace Hub ==="
echo "Data dir:  $DATA_DIR"
echo "Repo:      https://huggingface.co/datasets/$REPO"
echo "Manifest:  $MANIFEST"
echo "Available: $(df -h "$DATA_DIR" --output=avail | tail -1 | tr -d ' ')"
echo ""

# --- 1. Impulse responses (~265 MB compressed → 5.9 GB) ---
echo "--- [1/11] Impulse responses (~265 MB) ---"
marker="$DSET/.impulse_responses.done"
if [ -f "$marker" ]; then
    echo "  [skip] already done"
else
    curl -L --retry 3 "$DNS5/datasets_fullband.impulse_responses_000.tar.bz2" \
        | tar -C "$DSET" -xjf -
    normalize_paths
    touch "$marker"
fi
upload_new_files

# --- 2. Noise shards (9 total, ~6-8 GB each) ---
echo ""
echo "--- [2/11] Noise shards (9 shards, ~58 GB total) ---"
for shard in audioset_000 freesound_000; do
    download_noise_shard "$shard" "$DNS5"
done
for shard in audioset_001 audioset_002 audioset_003 audioset_004 audioset_005 audioset_006 freesound_001; do
    download_noise_shard "$shard" "$DNS4"
done
upload_new_files

# --- 3. Clean speech corpora (smallest first) ---

# VocalSet (~1 GB)
echo ""
echo "--- [3/11] VocalSet_48kHz_mono (~1 GB) ---"
if download_speech_single "VocalSet_48kHz_mono"; then
    upload_new_files
    rm -rf "$DSET/clean_fullband/VocalSet_48kHz_mono"
    echo "  [cleanup] Removed local files"
fi

# emotional_speech (~2.4 GB)
echo ""
echo "--- [4/11] emotional_speech (~2.4 GB) ---"
if download_speech_single "emotional_speech"; then
    upload_new_files
    rm -rf "$DSET/clean_fullband/emotional_speech"
    echo "  [cleanup] Removed local files"
fi

# russian_speech (~12 GB)
echo ""
echo "--- [5/11] russian_speech (~12 GB) ---"
if download_speech_single "russian_speech"; then
    upload_new_files
    rm -rf "$DSET/clean_fullband/russian_speech"
    echo "  [cleanup] Removed local files"
fi

# VCTK (~14.3 GB compressed, 27 GB unpacked)
echo ""
echo "--- [6/11] VCTK clean speech (~14.3 GB compressed) ---"
if download_speech_split "vctk_wav48_silence_trimmed" partaa partab partac; then
    upload_new_files
    rm -rf "$DSET/clean_fullband/vctk_wav48_silence_trimmed"
    echo "  [cleanup] Removed local files"
fi

# italian_speech (~42 GB, 4 parts)
echo ""
echo "--- [7/11] italian_speech (~42 GB) ---"
if download_speech_split "italian_speech" partaa partab partac partad; then
    upload_new_files
    rm -rf "$DSET/clean_fullband/italian_speech"
    echo "  [cleanup] Removed local files"
fi

# french_speech (~62 GB, 6 parts)
echo ""
echo "--- [8/11] french_speech (~62 GB) ---"
if download_speech_split "french_speech" partaa partab partac partad partae partah; then
    upload_new_files
    rm -rf "$DSET/clean_fullband/french_speech"
    echo "  [cleanup] Removed local files"
fi

# spanish_speech (~65 GB, 7 parts)
echo ""
echo "--- [9/11] spanish_speech (~65 GB) ---"
if download_speech_split "spanish_speech" partaa partab partac partad partae partaf partag; then
    upload_new_files
    rm -rf "$DSET/clean_fullband/spanish_speech"
    echo "  [cleanup] Removed local files"
fi

# read_speech (~299 GB, 21 parts)
echo ""
echo "--- [10/11] read_speech (~299 GB) ---"
if download_speech_split "read_speech" partaa partab partac partad partae partaf partag partah partai partaj partak partal partam partan partao partap partaq partar partas partat partau; then
    upload_new_files
    rm -rf "$DSET/clean_fullband/read_speech"
    echo "  [cleanup] Removed local files"
fi

# german_speech (~319 GB, 23 parts)
echo ""
echo "--- [11/11] german_speech (~319 GB) ---"
if download_speech_split "german_speech" partaa partab partac partad partae partaf partag partah partai partaj partak partal partam partan partao partap partaq partar partas partat partau partav partaw; then
    upload_new_files
    rm -rf "$DSET/clean_fullband/german_speech"
    echo "  [cleanup] Removed local files"
fi

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
echo ""
echo "--- Cleanup ---"
rm -rf "$DSET/tmp"
echo "Noise + IR files still at $DSET/{noise_fullband,impulse_responses}/"
echo "Remove manually with: rm -rf $DATA_DIR"

echo ""
echo "=== Upload complete ==="
echo "Dataset: https://huggingface.co/datasets/$REPO"
