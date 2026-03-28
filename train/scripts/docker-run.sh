#!/usr/bin/env bash
# Run a command (or script via stdin) inside the deepvqe Docker container.
#
# Usage:
#   ./scripts/docker-run.sh python train.py --config configs/default.yaml
#   ./scripts/docker-run.sh bash -c 'echo hi && python -c "import torch; print(torch.cuda.is_available())"'
#   echo "import torch; print(torch.cuda.get_device_name())" | ./scripts/docker-run.sh
#   ./scripts/docker-run.sh < my_script.py
#   ./scripts/docker-run.sh --docker-args "-it" bash    # interactive shell
set -euo pipefail

TRAIN_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT_ROOT="$(cd "$TRAIN_DIR/.." && pwd)"

IMAGE="${DEEPVQE_IMAGE:-deepvqe}"
CONTAINER="${DEEPVQE_CONTAINER:-deepvqe-run}"
GPU="${DEEPVQE_GPU:-all}"

DATA_DIR="${DEEPVQE_DATA_DIR:-$PROJECT_ROOT/datasets_fullband}"
CKPT_DIR="${DEEPVQE_CKPT_DIR:-$PROJECT_ROOT/checkpoints}"
LOG_DIR="${DEEPVQE_LOG_DIR:-$PROJECT_ROOT/logs}"
EVAL_DIR="${DEEPVQE_EVAL_DIR:-$PROJECT_ROOT/eval_output}"
CACHE_DIR="${DEEPVQE_CACHE_DIR:-$PROJECT_ROOT/.cache/torch_inductor}"

DOCKER_ARGS=(
    --rm
    --device "nvidia.com/gpu=$GPU"
    --name "$CONTAINER"
    --shm-size=4g
    -e TORCHINDUCTOR_FX_GRAPH_CACHE=1
    -e TORCHINDUCTOR_CACHE_DIR=/cache/torch_inductor
    -v "$PROJECT_ROOT:/workspace/deepvqe"
    -v "$DATA_DIR:/workspace/deepvqe/datasets_fullband"
    -v "$CKPT_DIR:/workspace/deepvqe/checkpoints"
    -v "$LOG_DIR:/workspace/deepvqe/logs"
    -v "$EVAL_DIR:/workspace/deepvqe/eval_output"
    -v "$CACHE_DIR:/cache/torch_inductor"
    -w /workspace/deepvqe/train
)

# Pull out --docker-args flags (extra args passed to docker run).
EXTRA_DOCKER_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --docker-args)
            shift
            # shellcheck disable=SC2206
            EXTRA_DOCKER_ARGS+=($1)
            shift
            ;;
        *)
            break
            ;;
    esac
done

# If stdin is not a terminal and no arguments given, read stdin as a Python script.
if [ $# -eq 0 ] && [ ! -t 0 ]; then
    TMPSCRIPT="$(mktemp "$TRAIN_DIR/.tmp_docker_run_XXXXXX.py")"
    trap 'rm -f "$TMPSCRIPT"' EXIT
    cat > "$TMPSCRIPT"
    exec docker run "${DOCKER_ARGS[@]}" "${EXTRA_DOCKER_ARGS[@]}" \
        "$IMAGE" python "/workspace/deepvqe/train/$(basename "$TMPSCRIPT")"
fi

if [ $# -eq 0 ]; then
    echo "Usage: $0 [COMMAND...]" >&2
    echo "       $0 < script.py" >&2
    echo "       $0 --docker-args \"-it\" bash" >&2
    exit 1
fi

exec docker run "${DOCKER_ARGS[@]}" "${EXTRA_DOCKER_ARGS[@]}" "$IMAGE" "$@"
