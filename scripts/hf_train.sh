#!/bin/bash
# Launch DeepVQE full training on HuggingFace Jobs (4x A100).
#
# Self-contained: scripts/hf_train_all.py has all model/training code inlined.
# Downloads DNS5 data from richiejp/dns5-16k HF dataset at startup (~50 GB).
# Auto-detects GPUs and re-launches with accelerate for multi-GPU DDP.
#
# Prerequisites:
#   hf login
#   Upload DNS5 data: python scripts/upload_dns5_to_hf.py  (one-time)
#
# Usage:
#   ./scripts/hf_train.sh                    # default 24h timeout
#   ./scripts/hf_train.sh --timeout 6h       # shorter test

set -euo pipefail

export HF_HUB_DISABLE_EXPERIMENTAL_WARNING=1
export PYTHONUNBUFFERED=1

EXTRA_ARGS="${*:-}"

hf jobs uv run \
    --flavor a100x4 \
    --timeout 24h \
    --secrets HF_TOKEN \
    --with torch --with accelerate --with huggingface_hub \
    --with einops --with soundfile --with matplotlib \
    --with scipy --with tensorboard --with tqdm \
    ${EXTRA_ARGS} \
    scripts/hf_train_all.py
