#!/bin/bash
# Launch DeepVQE overfit test on HuggingFace Jobs (4x A100).
#
# Self-contained: scripts/hf_overfit_all.py has all model/training code inlined.
# Downloads tiny dataset from richiejp/deepvqe-overfit-data (~5 MB) at startup.
# Auto-detects GPUs and re-launches with accelerate for multi-GPU DDP.
#
# Prerequisites:
#   hf login
#   Create HF dataset: richiejp/deepvqe-overfit-data (with 3 audio files)
#
# Usage:
#   ./scripts/hf_overfit.sh              # default 12 min timeout
#   ./scripts/hf_overfit.sh --timeout 5m # shorter test

set -euo pipefail

export HF_HUB_DISABLE_EXPERIMENTAL_WARNING=1
export PYTHONUNBUFFERED=1

EXTRA_ARGS="${*:-}"

hf jobs uv run \
    --flavor a100x4 \
    --timeout 12m \
    --secrets HF_TOKEN \
    --with torch --with accelerate --with huggingface_hub \
    --with einops --with soundfile \
    --with scipy --with tensorboard --with tqdm \
    ${EXTRA_ARGS} \
    scripts/hf_overfit_all.py
