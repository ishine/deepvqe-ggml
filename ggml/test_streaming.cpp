/**
 * Streaming equivalence test: compare batch vs frame-by-frame processing.
 *
 * Loads a model and test audio, processes it both ways, and verifies
 * the outputs match within floating-point tolerance.
 *
 * Usage:
 *   test_streaming model.gguf --input-npy mic.npy ref.npy
 */

#include "deepvqe_model.h"
#include "deepvqe_api.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    const char* model_path = nullptr;
    const char* mic_path = nullptr;
    const char* ref_path = nullptr;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--input-npy" && i + 2 < argc) {
            mic_path = argv[++i];
            ref_path = argv[++i];
        } else if (!model_path) {
            model_path = argv[i];
        } else {
            fprintf(stderr, "Usage: test_streaming model.gguf --input-npy mic.npy ref.npy\n");
            return 1;
        }
    }

    if (!model_path || !mic_path || !ref_path) {
        fprintf(stderr, "Error: model path and --input-npy required\n");
        return 1;
    }

    // Load model
    printf("Loading model: %s\n", model_path);
    uintptr_t ctx = deepvqe_new(model_path);
    if (!ctx) { fprintf(stderr, "Failed to load model\n"); return 1; }

    int hop = deepvqe_hop_length(ctx);
    int n_fft = deepvqe_fft_size(ctx);
    int n_freq = n_fft / 2 + 1;
    printf("  hop=%d n_fft=%d n_freq=%d\n", hop, n_fft, n_freq);

    // Load STFT input (1, F, T, 2)
    printf("Loading inputs: %s, %s\n", mic_path, ref_path);
    NpyArray mic_stft = npy_load(mic_path);
    NpyArray ref_stft = npy_load(ref_path);
    int T = (int)mic_stft.dim(2);
    int F = (int)mic_stft.dim(1);
    printf("  STFT shape: F=%d T=%d\n", F, T);

    // ── Batch forward ─────────────────────────────────────────────────────
    printf("\n--- Batch forward ---\n");
    deepvqe_model model;
    if (!load_model(model_path, model, false)) {
        fprintf(stderr, "Failed to load model for batch forward\n");
        return 1;
    }
    NpyArray batch_out = forward(mic_stft, ref_stft, model, "", false);
    printf("  Output: (%lld, %lld, %lld, %lld)\n",
           (long long)batch_out.dim(0), (long long)batch_out.dim(1),
           (long long)batch_out.dim(2), (long long)batch_out.dim(3));

    // ── Streaming forward ─────────────────────────────────────────────────
    printf("\n--- Streaming forward ---\n");
    deepvqe_stream_state stream_state;
    init_stream_state(stream_state, model);

    // Process frame-by-frame: extract STFT frames and feed to forward_frame
    // Output: (F, T, 2)
    std::vector<float> stream_out(F * T * 2, 0.0f);

    for (int t = 0; t < T; t++) {
        // Extract single STFT frame: (F, 1, 2) interleaved
        std::vector<float> mic_frame(F * 2), ref_frame(F * 2);
        for (int f = 0; f < F; f++) {
            mic_frame[f * 2 + 0] = mic_stft.data[f * T * 2 + t * 2 + 0];
            mic_frame[f * 2 + 1] = mic_stft.data[f * T * 2 + t * 2 + 1];
            ref_frame[f * 2 + 0] = ref_stft.data[f * T * 2 + t * 2 + 0];
            ref_frame[f * 2 + 1] = ref_stft.data[f * T * 2 + t * 2 + 1];
        }

        std::vector<float> enh_frame(F * 2);
        forward_frame(mic_frame.data(), ref_frame.data(), model,
                      stream_state, enh_frame.data());

        // Store in (F, T, 2) layout
        for (int f = 0; f < F; f++) {
            stream_out[f * T * 2 + t * 2 + 0] = enh_frame[f * 2 + 0];
            stream_out[f * T * 2 + t * 2 + 1] = enh_frame[f * 2 + 1];
        }
    }
    printf("  Processed %d frames\n", T);

    // ── Compare ───────────────────────────────────────────────────────────
    printf("\n--- Comparison ---\n");
    int n_total = F * T * 2;
    float max_err = max_abs_diff(batch_out.data.data(), stream_out.data(), n_total);
    float mean_err = mean_abs_diff(batch_out.data.data(), stream_out.data(), n_total);
    printf("  max_abs_diff:  %.2e\n", max_err);
    printf("  mean_abs_diff: %.2e\n", mean_err);

    // Per-frame errors
    float worst_frame_err = 0.0f;
    int worst_frame = 0;
    for (int t = 0; t < T; t++) {
        float frame_max = 0.0f;
        for (int f = 0; f < F; f++) {
            for (int c = 0; c < 2; c++) {
                int idx = f * T * 2 + t * 2 + c;
                float err = std::fabs(batch_out.data[idx] - stream_out[idx]);
                if (err > frame_max) frame_max = err;
            }
        }
        if (frame_max > worst_frame_err) {
            worst_frame_err = frame_max;
            worst_frame = t;
        }
    }
    printf("  Worst frame: t=%d (max_err=%.2e)\n", worst_frame, worst_frame_err);

    bool pass = max_err < 1e-5f;
    printf("\n  %s (threshold: 1e-5)\n", pass ? "PASS" : "FAIL");

    deepvqe_free(ctx);
    return pass ? 0 : 1;
}
