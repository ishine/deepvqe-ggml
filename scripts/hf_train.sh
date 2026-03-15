#!/bin/bash
# Launch DeepVQE training on HuggingFace Jobs (8xH200).
#
# Prerequisites:
#   pip install huggingface_hub[cli]
#   huggingface-cli login
#
# Usage:
#   ./scripts/hf_train.sh              # full 250 epoch run
#   ./scripts/hf_train.sh --timeout 1h # quick test run (edit epochs in cloud.yaml)

set -euo pipefail

EXTRA_ARGS="${*}"

hf jobs uv run \
    --flavor h200x8 \
    --timeout 24h \
    --secrets HF_TOKEN \
    --with accelerate --with huggingface_hub \
    --with einops --with pesq --with pystoi --with soundfile \
    --with scipy --with tensorboard --with gguf --with einops \
    ${EXTRA_ARGS} \
    -- bash -c '
        set -euo pipefail
        echo "=== Downloading DNS5 minimal subset ==="
        bash scripts/download_dns5_minimal.sh /data
        echo "=== Starting training ==="
        accelerate launch --num_processes 8 --mixed_precision bf16 \
            train.py --config configs/cloud.yaml
    '
