/**
 * Quantization divergence test: compare F32 vs Q8_0 model outputs.
 *
 * Loads both models, runs forward_graph on the same input, and reports
 * final output max/mean error between the two.
 *
 * Usage:
 *   test_quantize --f32 model.gguf --q8 model_q8.gguf \
 *     --input-npy mic.npy ref.npy
 */

#include "deepvqe_graph.h"

#include <cstdio>
#include <string>

int main(int argc, char** argv) {
    const char* f32_path = nullptr;
    const char* q8_path  = nullptr;
    const char* mic_path = nullptr;
    const char* ref_path = nullptr;
    float ok_thresh   = 1e-3f;
    float fail_thresh = 5e-2f;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--f32" && i + 1 < argc) f32_path = argv[++i];
        else if (arg == "--q8" && i + 1 < argc) q8_path = argv[++i];
        else if (arg == "--input-npy" && i + 2 < argc) {
            mic_path = argv[++i];
            ref_path = argv[++i];
        }
        else if (arg == "--ok-thresh" && i + 1 < argc) ok_thresh = std::stof(argv[++i]);
        else if (arg == "--fail-thresh" && i + 1 < argc) fail_thresh = std::stof(argv[++i]);
        else {
            fprintf(stderr,
                "Usage: test_quantize --f32 model.gguf --q8 model_q8.gguf "
                "--input-npy mic.npy ref.npy\n");
            return 1;
        }
    }

    if (!f32_path || !q8_path || !mic_path || !ref_path) {
        fprintf(stderr, "Error: --f32, --q8, and --input-npy required\n");
        return 1;
    }

    printf("=== Loading F32 model: %s ===\n", f32_path);
    dvqe_graph_model model_f32;
    if (!load_graph_model(f32_path, model_f32, false)) return 1;

    printf("=== Loading Q8 model: %s ===\n", q8_path);
    dvqe_graph_model model_q8;
    if (!load_graph_model(q8_path, model_q8, true)) return 1;

    NpyArray mic = npy_load(mic_path);
    NpyArray ref = npy_load(ref_path);
    printf("Input: F=%lld T=%lld\n",
           (long long)mic.dim(1), (long long)mic.dim(2));

    printf("\n=== Running F32 forward_graph ===\n");
    NpyArray out_f32 = forward_graph(mic, ref, model_f32, nullptr, false);

    printf("=== Running Q8 forward_graph ===\n");
    NpyArray out_q8 = forward_graph(mic, ref, model_q8, nullptr, false);

    printf("\n=== Output comparison (F32 vs Q8_0) ===\n");
    float max_err = max_abs_diff(out_f32.data.data(), out_q8.data.data(), out_f32.numel());
    float mean_err = mean_abs_diff(out_f32.data.data(), out_q8.data.data(), out_f32.numel());
    bool pass = print_result("output", max_err, mean_err, ok_thresh, fail_thresh);

    free_graph_model(model_f32);
    free_graph_model(model_q8);

    return pass ? 0 : 1;
}
