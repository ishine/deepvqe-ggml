#!/usr/bin/env bash
# Download minimal DNS Challenge 5 subset for initial DeepVQE experiments.
#
# Downloads:
#   - VCTK clean speech (14.3 GB compressed → 27 GB unpacked, 109 speakers)
#   - 2 noise shards   (10.7 GB compressed → ~17 GB unpacked)
#   - Impulse responses (265 MB compressed → 5.9 GB unpacked)
#
# Total: ~25 GB download, ~50 GB unpacked
#
# Usage:
#   ./scripts/download_dns5_minimal.sh [output_dir]
#   Default output_dir: ./datasets_fullband

set -euo pipefail

OUT="${1:-./datasets_fullband}"
BASE="https://dnschallengepublic.blob.core.windows.net/dns5archive/V5_training_dataset"

mkdir -p "$OUT"/{clean_fullband,noise_fullband,tmp}

echo "=== Downloading DNS Challenge 5 minimal subset ==="
echo "Output: $OUT"
echo ""

# ---------------------------------------------------------------------------
# 1. VCTK clean speech (Track 1 headset — split .tgz parts)
# ---------------------------------------------------------------------------
echo "--- VCTK clean speech (14.3 GB compressed) ---"
for part in partaa partab partac; do
    url="$BASE/Track1_Headset/vctk_wav48_silence_trimmed.tgz.$part"
    dest="$OUT/tmp/vctk_wav48_silence_trimmed.tgz.$part"
    if [ -f "$dest" ]; then
        echo "  [skip] $part (already downloaded)"
    else
        echo "  [download] $part ..."
        curl -L --retry 3 -C - "$url" -o "$dest"
    fi
done

echo "  [extract] Reassembling and extracting VCTK ..."
cat "$OUT"/tmp/vctk_wav48_silence_trimmed.tgz.part* \
    | tar -C "$OUT/clean_fullband" -xzf -
echo "  [done] VCTK extracted to $OUT/clean_fullband/vctk_wav48_silence_trimmed/"

# ---------------------------------------------------------------------------
# 2. Noise — 2 shards (audioset_000 + freesound_000)
# ---------------------------------------------------------------------------
echo ""
echo "--- Noise shards (10.7 GB compressed) ---"
for shard in audioset_000 freesound_000; do
    url="$BASE/noise_fullband/datasets_fullband.noise_fullband.${shard}.tar.bz2"
    marker="$OUT/noise_fullband/.${shard}.done"
    if [ -f "$marker" ]; then
        echo "  [skip] $shard (already extracted)"
    else
        echo "  [download+extract] $shard ..."
        curl -L --retry 3 "$url" | tar -C "$OUT" -xjf -
        touch "$marker"
    fi
done
echo "  [done] Noise extracted to $OUT/noise_fullband/"

# ---------------------------------------------------------------------------
# 3. Impulse responses (265 MB)
# ---------------------------------------------------------------------------
echo ""
echo "--- Impulse responses (265 MB compressed) ---"
url="$BASE/datasets_fullband.impulse_responses_000.tar.bz2"
marker="$OUT/.impulse_responses.done"
if [ -f "$marker" ]; then
    echo "  [skip] (already extracted)"
else
    echo "  [download+extract] impulse_responses ..."
    curl -L --retry 3 "$url" | tar -C "$OUT" -xjf -
    touch "$marker"
fi
echo "  [done] IRs extracted to $OUT/impulse_responses/"

# ---------------------------------------------------------------------------
# Cleanup temp files
# ---------------------------------------------------------------------------
echo ""
echo "--- Cleanup ---"
rm -rf "$OUT/tmp"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "=== Download complete ==="
echo ""
echo "Directory structure:"
echo "  $OUT/clean_fullband/vctk_wav48_silence_trimmed/  (~27 GB, 109 speakers)"
echo "  $OUT/noise_fullband/                             (~17 GB)"
echo "  $OUT/impulse_responses/                          (~5.9 GB)"
echo ""
echo "Note: Audio is 48 kHz WAV. The data pipeline resamples to 16 kHz automatically."
echo ""
echo "Next: pack all training data into a single squashfs image:"
echo "  mkdir -p $OUT/_dns5_staging"
echo "  cp -rl $OUT/clean_fullband $OUT/_dns5_staging/clean"
echo "  cp -rl $OUT/datasets_fullband/noise_fullband $OUT/_dns5_staging/noise"
echo "  cp -rl $OUT/datasets_fullband/impulse_responses $OUT/_dns5_staging/impulse_responses"
echo "  mksquashfs $OUT/_dns5_staging $OUT/sqsh/dns5.sqsh -comp zstd -Xcompression-level 3"
echo "  rm -rf $OUT/_dns5_staging"
echo ""
echo "The Docker entrypoint mounts dns5.sqsh -> /data/dns5/{clean,noise,impulse_responses}"
