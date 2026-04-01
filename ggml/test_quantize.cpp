/**
 * Quantization divergence test: compare F32 vs Q8_0 model outputs.
 *
 * Loads both models, runs forward on the same input, and reports
 * per-block max/mean error between the two.
 *
 * Usage:
 *   test_quantize --f32 model.gguf --q8 model_q8.gguf \
 *     --input-npy mic.npy ref.npy
 */

#include "deepvqe_model.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <sys/stat.h>

static bool mkdirp(const std::string& path) {
    return mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
}

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

    // Load models
    printf("=== Loading F32 model: %s ===\n", f32_path);
    deepvqe_model model_f32;
    if (!load_model(f32_path, model_f32, false)) return 1;

    printf("=== Loading Q8 model: %s ===\n", q8_path);
    deepvqe_model model_q8;
    if (!load_model(q8_path, model_q8, true)) return 1;

    // Load inputs
    NpyArray mic = npy_load(mic_path);
    NpyArray ref = npy_load(ref_path);
    printf("Input: mic (%lld,%lld,%lld,%lld) ref (%lld,%lld,%lld,%lld)\n",
           (long long)mic.dim(0), (long long)mic.dim(1),
           (long long)mic.dim(2), (long long)mic.dim(3),
           (long long)ref.dim(0), (long long)ref.dim(1),
           (long long)ref.dim(2), (long long)ref.dim(3));

    // Create temp dirs for intermediates
    std::string tmp_f32 = "/tmp/dvqe_quant_f32";
    std::string tmp_q8  = "/tmp/dvqe_quant_q8";
    mkdirp(tmp_f32);
    mkdirp(tmp_q8);

    // Run forward on both
    printf("\n=== Running F32 forward ===\n");
    NpyArray out_f32 = forward(mic, ref, model_f32, tmp_f32, false);

    printf("=== Running Q8 forward ===\n");
    NpyArray out_q8  = forward(mic, ref, model_q8, tmp_q8, false);

    // Compare intermediates
    printf("\n=== Per-block divergence (F32 vs Q8_0) ===\n");
    printf("  %-16s %12s %12s %s\n", "Block", "Max Error", "Mean Error", "Status");
    printf("  %-16s %12s %12s %s\n", "-----", "---------", "----------", "------");

    const char* blocks[] = {
        "fe_mic", "fe_ref",
        "mic_enc1", "mic_enc2",
        "far_enc1", "far_enc2",
        "align",
        "mic_enc3", "mic_enc4", "mic_enc5",
        "bottleneck",
        "dec5", "dec4", "dec3", "dec2", "dec1",
    };

    int n_warn = 0, n_fail = 0;

    for (const char* name : blocks) {
        // FE has no weights — outputs are always identical
        if (std::strncmp(name, "fe_", 3) == 0) {
            printf("  [OK]   %s: max=0.00e+00 mean=0.00e+00 (no weights)\n", name);
            continue;
        }

        std::string path_a = tmp_f32 + "/" + name + ".npy";
        std::string path_b = tmp_q8  + "/" + name + ".npy";

        NpyArray a = npy_load(path_a);
        NpyArray b = npy_load(path_b);

        if (a.numel() != b.numel()) {
            fprintf(stderr, "  [FAIL] %s: element count mismatch\n", name);
            n_fail++;
            continue;
        }

        float me = max_abs_diff(a.data.data(), b.data.data(), a.numel());
        float ae = mean_abs_diff(a.data.data(), b.data.data(), a.numel());

        if (!print_result(name, me, ae, ok_thresh, fail_thresh))
            n_fail++;
        else if (me >= ok_thresh)
            n_warn++;
    }

    // Compare final output
    printf("\n=== Final output comparison ===\n");
    float max_err = max_abs_diff(out_f32.data.data(), out_q8.data.data(), out_f32.numel());
    float mean_err = mean_abs_diff(out_f32.data.data(), out_q8.data.data(), out_f32.numel());
    if (!print_result("output", max_err, mean_err, ok_thresh, fail_thresh))
        n_fail++;

    // Summary
    printf("\n=== Summary ===\n");
    printf("  Blocks with warnings: %d\n", n_warn);
    printf("  Blocks with failures: %d\n", n_fail);

    if (n_fail > 0) {
        printf("  RESULT: FAIL (some blocks exceed %.2e threshold)\n", fail_thresh);
        return 1;
    } else if (n_warn > 0) {
        printf("  RESULT: WARN (all blocks < %.2e but some > %.2e)\n", fail_thresh, ok_thresh);
        return 0;
    } else {
        printf("  RESULT: OK (all blocks < %.2e)\n", ok_thresh);
        return 0;
    }
}
