#!/usr/bin/env bash
# Upload the DeepVQE GGUF model to HuggingFace and print the SHA256 checksum
# for embedding in VoxInput's model.go.
#
# Prerequisites:
#   pip install huggingface_hub  (or: pipx install huggingface_hub)
#   huggingface-cli login
#
# Usage:
#   ./scripts/upload_model.sh [path/to/deepvqe.gguf] [checkpoint]
#
# If no GGUF path is given, exports from the latest checkpoint first.

set -euo pipefail

REPO_ID="richiejp/deepvqe-aec-gguf"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

GGUF_PATH="${1:-$PROJECT_DIR/deepvqe.gguf}"
CHECKPOINT="${2:-}"

# Export if GGUF doesn't exist or a checkpoint was explicitly given
if [ ! -f "$GGUF_PATH" ] || [ -n "$CHECKPOINT" ]; then
    if [ -z "$CHECKPOINT" ]; then
        # Find the most recent checkpoint
        CHECKPOINT="$(ls -t "$PROJECT_DIR"/checkpoints/*.pt 2>/dev/null | head -1 || true)"
        if [ -z "$CHECKPOINT" ]; then
            echo "Error: No GGUF file at $GGUF_PATH and no checkpoints found in $PROJECT_DIR/checkpoints/"
            echo "Either provide a GGUF path or a checkpoint path."
            exit 1
        fi
        echo "Using latest checkpoint: $CHECKPOINT"
    fi
    echo "Exporting GGUF from checkpoint..."
    make -C "$PROJECT_DIR" export CHECKPOINT="$CHECKPOINT"
    GGUF_PATH="$PROJECT_DIR/deepvqe.gguf"
fi

if [ ! -f "$GGUF_PATH" ]; then
    echo "Error: GGUF file not found at $GGUF_PATH"
    exit 1
fi

# Compute checksum
echo ""
echo "Computing SHA256..."
CHECKSUM=$(sha256sum "$GGUF_PATH" | cut -d' ' -f1)
FILE_SIZE=$(stat --format=%s "$GGUF_PATH" 2>/dev/null || stat -f%z "$GGUF_PATH")
echo "  File:     $GGUF_PATH"
echo "  Size:     $FILE_SIZE bytes ($(echo "scale=1; $FILE_SIZE / 1048576" | bc) MB)"
echo "  SHA256:   $CHECKSUM"

# Upload
echo ""
echo "Uploading to huggingface.co/$REPO_ID ..."
huggingface-cli upload "$REPO_ID" "$GGUF_PATH" deepvqe.gguf --repo-type model

echo ""
echo "Done! Update VoxInput's internal/deepvqe/model.go with:"
echo ""
echo "  modelSHA256 = \"$CHECKSUM\""
echo ""
