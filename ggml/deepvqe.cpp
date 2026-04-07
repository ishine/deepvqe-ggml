/**
 * DeepVQE AEC inference CLI (C API).
 *
 * Build:
 *   cd ggml && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
 *
 * Usage:
 *   ./build/deepvqe model.gguf --input-npy mic.npy ref.npy --output enhanced.npy
 */

#include "deepvqe_api.h"
#include "common.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    const char* model_path = nullptr;
    const char* mic_path   = nullptr;
    const char* ref_path   = nullptr;
    std::string out_path   = "enhanced.npy";

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--input-npy" && i + 2 < argc) {
            mic_path = argv[++i];
            ref_path = argv[++i];
        } else if (a == "--output" && i + 1 < argc) {
            out_path = argv[++i];
        } else if (a == "-h" || a == "--help") {
            fprintf(stderr, "Usage: deepvqe <model.gguf> --input-npy mic.npy ref.npy [--output enhanced.npy]\n");
            return 0;
        } else if (!model_path) {
            model_path = argv[i];
        } else {
            fprintf(stderr, "Unknown arg: %s\n", argv[i]);
            return 1;
        }
    }

    if (!model_path || !mic_path || !ref_path) {
        fprintf(stderr, "Usage: deepvqe <model.gguf> --input-npy mic.npy ref.npy [--output enhanced.npy]\n");
        return 1;
    }

    // Load PCM inputs
    NpyArray mic = npy_load(mic_path);
    NpyArray ref = npy_load(ref_path);
    int n_mic = (int)mic.numel();
    int n_ref = (int)ref.numel();
    if (n_mic != n_ref) {
        fprintf(stderr, "Error: mic (%d) and ref (%d) sample counts differ\n", n_mic, n_ref);
        return 1;
    }
    printf("Input: %d samples (%.2f s)\n", n_mic, n_mic / 16000.0f);

    // Init model
    uintptr_t ctx = deepvqe_new(model_path);
    if (!ctx) {
        fprintf(stderr, "Error: failed to load model: %s\n", model_path);
        return 1;
    }

    // Process
    std::vector<float> enhanced(n_mic);
    int ret = deepvqe_process_f32(ctx, mic.data.data(), ref.data.data(), n_mic, enhanced.data());
    if (ret != 0) {
        fprintf(stderr, "Error: deepvqe_process_f32 returned %d: %s\n",
                ret, deepvqe_last_error(ctx));
        deepvqe_free(ctx);
        return 1;
    }

    // Save output
    npy_save(out_path, enhanced.data(), {(int64_t)n_mic});
    printf("Saved enhanced to: %s\n", out_path.c_str());

    deepvqe_free(ctx);
    return 0;
}
