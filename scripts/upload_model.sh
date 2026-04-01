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
HF_HUB="huggingface_hub==1.8.0"
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
echo "  Size:     $FILE_SIZE bytes ($((FILE_SIZE / 1048576)) MB)"
echo "  SHA256:   $CHECKSUM"

# Generate model card
MODEL_CARD="$(mktemp)"
cat > "$MODEL_CARD" <<'CARD'
---
license: apache-2.0
tags:
  - audio
  - speech-enhancement
  - echo-cancellation
  - noise-suppression
  - ggml
  - gguf
pipeline_tag: audio-to-audio
---

# DeepVQE-AEC (GGUF)

GGML/GGUF inference model for **DeepVQE** (Indenbom et al., Interspeech 2023) —
joint acoustic echo cancellation (AEC), noise suppression, and dereverberation.

## Quick Start

### Build

Requires CMake 3.20+ and a C++17 compiler. The ggml library is included as a
git submodule.

```bash
git clone --recursive https://github.com/richiejp/deepvqe-ggml
cd deepvqe-ggml/ggml

# CLI only
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# With shared library (C API for FFI from Python, Go, etc.)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DDEEPVQE_BUILD_SHARED=ON
cmake --build build
```

### CLI

Process STFT-domain audio (NumPy .npy files):

```bash
./build/deepvqe deepvqe.gguf --input-npy mic_stft.npy ref_stft.npy
```

### C API

The shared library (`libdeepvqe.so`) exposes a simple C API for integration
into any language:

```c
#include "deepvqe_api.h"

// Load model
uintptr_t ctx = deepvqe_new("deepvqe.gguf");

// Process 16kHz mono float32 audio
//   mic: microphone input (with echo + noise)
//   ref: far-end reference (what the speaker is hearing)
//   out: cleaned output (pre-allocated, same length)
int ret = deepvqe_process_f32(ctx, mic, ref, n_samples, out);

// int16 PCM variant also available
int ret = deepvqe_process_s16(ctx, mic_s16, ref_s16, n_samples, out_s16);

deepvqe_free(ctx);
```

### Python (ctypes)

```python
import ctypes, numpy as np

lib = ctypes.CDLL("./build/libdeepvqe.so")
lib.deepvqe_new.restype = ctypes.c_void_p
lib.deepvqe_new.argtypes = [ctypes.c_char_p]
lib.deepvqe_process_f32.restype = ctypes.c_int
lib.deepvqe_process_f32.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_void_p,
]
lib.deepvqe_free.argtypes = [ctypes.c_void_p]

ctx = lib.deepvqe_new(b"deepvqe.gguf")

mic = np.zeros(16000, dtype=np.float32)  # 1 second of 16kHz audio
ref = np.zeros(16000, dtype=np.float32)
out = np.empty_like(mic)

ret = lib.deepvqe_process_f32(
    ctx,
    mic.ctypes.data,
    ref.ctypes.data,
    len(mic),
    out.ctypes.data,
)

lib.deepvqe_free(ctx)
```

Used in production by [VoxInput](https://github.com/richiejp/VoxInput) for
real-time voice input with echo cancellation.

## Model Details

| | |
|---|---|
| **Architecture** | DeepVQE with AlignBlock (soft delay estimation) |
| **Parameters** | ~8.0M |
| **Sample rate** | 16 kHz |
| **STFT** | 512 FFT, 256 hop (16 ms), sqrt-Hann window |
| **Delay range** | dmax=32 frames (320 ms) |
| **Format** | GGUF |
| **Variants** | F32 (31 MB), Q8_0 (8.5 MB) |

## Quantization

The Q8_0 variant (`deepvqe_q8.gguf`) reduces model size by 73% (31 MB to
8.5 MB) using GGML Q8_0 quantization with selective layer preservation.

| Layer group | Quantization | Reason |
|-------------|-------------|--------|
| Encoder/decoder (2-5) weights | Q8_0 | Residual connections mitigate error |
| Bottleneck GRU + FC weights | Q8_0 | Largest tensors (~3.6M params) |
| AlignBlock (attention) | F32 | Softmax precision for delay estimation |
| dec1 (mask output) | F32 | Directly controls complex convolving mask |
| All biases, ChannelAffine | F32 | Small tensors, negligible size savings |

**Divergence from F32:** output max error 5e-2, mean error 7e-4.

## Training

Trained on the full [DNS5 16 kHz dataset](https://huggingface.co/datasets/richiejp/dns5-16k)
(~300K clean speech files after DNSMOS quality filtering, 64K noise, 60K impulse
responses) on a single NVIDIA RTX 5070 (16 GB).

**Safety note:** Training data was filtered by DNSMOS perceived quality scores,
which can misclassify distressed speech (e.g. screaming, crying) as noise. This
model may therefore attenuate or distort such signals and **should not be relied
upon for emergency call or safety-critical applications**.

### Data

- [DNS5](https://github.com/microsoft/DNS-Challenge) (Microsoft, CC BY 4.0)
- [ICASSP 2022 AEC Challenge](https://github.com/microsoft/AEC-Challenge) — echo scenarios

See [deepvqe-ggml](https://github.com/richiejp/deepvqe-ggml) for training
code and full documentation.

## References

- [DeepVQE paper](https://arxiv.org/abs/2306.03177) (Indenbom et al., 2023)
- [deepvqe-ggml](https://github.com/richiejp/deepvqe-ggml) — training & export code
CARD
trap 'rm -f "$MODEL_CARD"' EXIT

# Check for quantized model
Q8_PATH="${GGUF_PATH%.gguf}_q8.gguf"
if [ -f "$Q8_PATH" ]; then
    Q8_CHECKSUM=$(sha256sum "$Q8_PATH" | cut -d' ' -f1)
    Q8_SIZE=$(stat --format=%s "$Q8_PATH" 2>/dev/null || stat -f%z "$Q8_PATH")
    echo ""
    echo "Quantized model found:"
    echo "  File:     $Q8_PATH"
    echo "  Size:     $Q8_SIZE bytes ($((Q8_SIZE / 1048576)) MB)"
    echo "  SHA256:   $Q8_CHECKSUM"
fi

# Upload
echo ""
echo "Uploading to huggingface.co/$REPO_ID ..."
uvx --from "$HF_HUB" hf upload "$REPO_ID" "$MODEL_CARD" README.md --repo-type model
uvx --from "$HF_HUB" hf upload "$REPO_ID" "$GGUF_PATH" deepvqe.gguf --repo-type model

if [ -f "$Q8_PATH" ]; then
    uvx --from "$HF_HUB" hf upload "$REPO_ID" "$Q8_PATH" deepvqe_q8.gguf --repo-type model
fi

echo ""
echo "Done! Update VoxInput's internal/deepvqe/model.go with:"
echo ""
echo "  modelSHA256 = \"$CHECKSUM\""
echo ""
