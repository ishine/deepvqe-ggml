/**
 * Benchmark: hand-rolled C++ vs GGML graph inference.
 */

#include "deepvqe_model.h"
#include "deepvqe_graph.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    const char* model_path = nullptr;
    const char* mic_path = nullptr;
    const char* ref_path = nullptr;
    int iters = 3;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--input-npy" && i + 2 < argc) {
            mic_path = argv[++i]; ref_path = argv[++i];
        } else if (arg == "--iters" && i + 1 < argc) {
            iters = std::stoi(argv[++i]);
        } else if (!model_path) {
            model_path = argv[i];
        }
    }
    if (!model_path || !mic_path || !ref_path) {
        fprintf(stderr, "Usage: bench model.gguf --input-npy mic.npy ref.npy [--iters N]\n");
        return 1;
    }

    NpyArray mic_stft = npy_load(mic_path);
    NpyArray ref_stft = npy_load(ref_path);
    printf("Input: F=%d T=%d, %d iterations\n\n", (int)mic_stft.dim(1), (int)mic_stft.dim(2), iters);

    deepvqe_model hr_model;
    if (!load_model(model_path, hr_model, false)) return 1;
    dvqe_graph_model gr_model;
    if (!load_graph_model(model_path, gr_model, false)) return 1;

    // Batch graph (warmup + reference)
    NpyArray hr = forward(mic_stft, ref_stft, hr_model, "", false);
    NpyArray gr = forward_graph(mic_stft, ref_stft, gr_model, nullptr, false);

    int64_t n = std::min(hr.numel(), gr.numel());
    float max_err = max_abs_diff(hr.data.data(), gr.data.data(), n);
    float mean_err = mean_abs_diff(hr.data.data(), gr.data.data(), n);
    printf("Batch graph vs hand-rolled: max=%.2e mean=%.2e %s\n",
           max_err, mean_err, max_err < 1e-3 ? "OK" : "FAIL");

    // Streaming graph: loop over frames
    int F = (int)mic_stft.dim(1);
    int T = (int)mic_stft.dim(2);

    dvqe_stream_graph sg;
    if (!build_stream_graph(gr_model, sg)) {
        fprintf(stderr, "Failed to build stream graph\n");
        return 1;
    }

    std::vector<float> stream_out(F * T * 2, 0.0f);
    std::vector<float> mic_frame(F * 2), ref_frame(F * 2), enh_frame(F * 2);
    for (int t = 0; t < T; t++) {
        extract_stft_frame(mic_stft.data.data(), F, T, t, mic_frame.data());
        extract_stft_frame(ref_stft.data.data(), F, T, t, ref_frame.data());
        process_frame_graph(sg, gr_model, mic_frame.data(), ref_frame.data(), enh_frame.data());
        scatter_stft_frame(enh_frame.data(), stream_out.data(), F, T, t);
    }

    // Per-frame error (first few frames + overall)
    for (int t = 0; t < std::min(T, 5); t++) {
        float frame_max = 0.0f;
        for (int f = 0; f < F; f++) {
            for (int c = 0; c < 2; c++) {
                int idx = f * T * 2 + t * 2 + c;
                float err = std::fabs(gr.data[idx] - stream_out[idx]);
                if (err > frame_max) frame_max = err;
            }
        }
        printf("  Frame %d: max_err=%.2e\n", t, frame_max);
    }

    float s_max = max_abs_diff(gr.data.data(), stream_out.data(), gr.numel());
    float s_mean = mean_abs_diff(gr.data.data(), stream_out.data(), gr.numel());
    printf("Stream graph vs batch graph: max=%.2e mean=%.2e %s\n\n",
           s_max, s_mean, s_max < 1e-4 ? "OK" : "FAIL");

    // Timing
    int64_t hr_total = 0, gr_total = 0, sg_total = 0;
    for (int i = 0; i < iters; i++) {
        int64_t t0 = ggml_time_us();
        forward(mic_stft, ref_stft, hr_model, "", false);
        hr_total += ggml_time_us() - t0;
    }
    for (int i = 0; i < iters; i++) {
        int64_t t0 = ggml_time_us();
        forward_graph(mic_stft, ref_stft, gr_model, nullptr, false);
        gr_total += ggml_time_us() - t0;
    }
    for (int i = 0; i < iters; i++) {
        int64_t t0 = ggml_time_us();
        reset_stream_graph(sg, gr_model);
        for (int t = 0; t < T; t++) {
            extract_stft_frame(mic_stft.data.data(), F, T, t, mic_frame.data());
            extract_stft_frame(ref_stft.data.data(), F, T, t, ref_frame.data());
            process_frame_graph(sg, gr_model, mic_frame.data(), ref_frame.data(), enh_frame.data());
        }
        sg_total += ggml_time_us() - t0;
    }

    printf("Hand-rolled:   %.1f ms/iter\n", (double)hr_total / iters / 1000.0);
    printf("Batch graph:   %.1f ms/iter\n", (double)gr_total / iters / 1000.0);
    printf("Stream graph:  %.1f ms/iter\n", (double)sg_total / iters / 1000.0);
    printf("Batch speedup: %.1fx over hand-rolled\n", (double)hr_total / (double)gr_total);

    free_stream_graph(sg);

    free_graph_model(gr_model);
    return 0;
}
