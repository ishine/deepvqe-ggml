/**
 * STFT verification tests.
 *
 * Test 1: Roundtrip — sine wave → STFT → iSTFT → compare with original.
 * Test 2: Python reference — compare compute_stft() output against
 *         torch.stft reference saved as .npy.
 * Test 3: End-to-end model — compare C++ deepvqe_process_f32() output
 *         against PyTorch model output on the same PCM input.
 *
 * Usage:
 *   test_stft [--ref-input stft_input.npy --ref-stft stft_ref.npy]
 *             [--model model.gguf --val-dir test_data/ --n-samples 3]
 */

#include "common.h"
#include "deepvqe_api.h"
#include "deepvqe_graph.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// ── Test 1: STFT → iSTFT roundtrip ─────────────────────────────────────────

static bool test_roundtrip() {
    printf("=== Test 1: STFT/iSTFT roundtrip ===\n");

    const int n_fft = 512;
    const int hop = 256;
    const int sr = 16000;
    const int n_freq = n_fft / 2 + 1;

    // Generate 1 second of 440Hz sine wave
    const int N = sr;
    std::vector<float> signal(N);
    for (int i = 0; i < N; i++)
        signal[i] = 0.5f * sinf(2.0f * (float)M_PI * 440.0f * i / sr);

    auto window = make_sqrt_hann(n_fft);
    int n_frames = stft_n_frames(N, hop);

    // STFT
    std::vector<float> stft_data(n_freq * n_frames * 2, 0.0f);
    compute_stft(signal.data(), N, n_fft, hop, window.data(),
                 stft_data.data(), n_freq, n_frames);

    // Verify STFT produced non-zero output
    float stft_max = 0.0f;
    for (auto v : stft_data) {
        float a = std::fabs(v);
        if (a > stft_max) stft_max = a;
    }
    printf("  STFT max magnitude: %.4f\n", stft_max);
    if (stft_max < 1e-6f) {
        printf("  FAIL: STFT output is all zeros\n\n");
        return false;
    }

    // iSTFT
    std::vector<float> reconstructed(N, 0.0f);
    compute_istft(stft_data.data(), n_freq, n_frames,
                  n_fft, hop, window.data(), reconstructed.data(), N);

    // Compare: skip first and last hop (edge effects from center padding)
    int skip = hop;
    int cmp_len = N - 2 * skip;
    float max_err = max_abs_diff(signal.data() + skip,
                                  reconstructed.data() + skip, cmp_len);
    float mean_err = mean_abs_diff(signal.data() + skip,
                                    reconstructed.data() + skip, cmp_len);
    printf("  Roundtrip (skip %d edge samples):\n", skip);
    printf("    max_abs_diff:  %.2e\n", max_err);
    printf("    mean_abs_diff: %.2e\n", mean_err);

    bool pass = max_err < 1e-5f;
    printf("  %s (threshold: 1e-5)\n\n", pass ? "PASS" : "FAIL");
    return pass;
}

// ── Test 2: Compare against Python torch.stft reference ─────────────────────

static bool test_python_reference(const char* input_path, const char* ref_path) {
    printf("=== Test 2: Python STFT reference comparison ===\n");

    NpyArray input_npy, ref_npy;
    try {
        input_npy = npy_load(input_path);
        ref_npy = npy_load(ref_path);
    } catch (const std::exception& e) {
        printf("  SKIP: %s\n\n", e.what());
        return true;  // not a failure, just missing reference data
    }

    const int n_fft = 512;
    const int hop = 256;
    const int n_freq = n_fft / 2 + 1;
    int N = (int)input_npy.numel();

    auto window = make_sqrt_hann(n_fft);
    int n_frames = stft_n_frames(N, hop);

    printf("  Input: %d samples, STFT: %d freq × %d frames\n", N, n_freq, n_frames);

    // Compute C++ STFT
    std::vector<float> cpp_stft(n_freq * n_frames * 2, 0.0f);
    compute_stft(input_npy.data.data(), N, n_fft, hop, window.data(),
                 cpp_stft.data(), n_freq, n_frames);

    // Python ref is from torch.stft with return_complex=False → (F, T, 2)
    // Our layout is also (F, T, 2) with F = n_freq
    int64_t expected_numel = (int64_t)n_freq * n_frames * 2;
    if (ref_npy.numel() != expected_numel) {
        printf("  FAIL: ref shape mismatch (got %lld, expected %lld)\n\n",
               (long long)ref_npy.numel(), (long long)expected_numel);
        return false;
    }

    float max_err = max_abs_diff(cpp_stft.data(), ref_npy.data.data(),
                                  expected_numel);
    float mean_err = mean_abs_diff(cpp_stft.data(), ref_npy.data.data(),
                                    expected_numel);
    printf("  C++ vs Python STFT:\n");
    printf("    max_abs_diff:  %.2e\n", max_err);
    printf("    mean_abs_diff: %.2e\n", mean_err);

    // Float32 FFT precision: C++ radix-2 vs cuFFT/FFTW may differ by ~2e-4
    bool pass = max_err < 5e-4f;
    printf("  %s (threshold: 5e-4)\n\n", pass ? "PASS" : "FAIL");
    return pass;
}

// ── Test 3: End-to-end model comparison ────────────────────────────────────

static bool test_model_e2e(const char* model_path, const char* val_dir,
                           const char* ref_dir, int n_samples) {
    printf("=== Test 3: End-to-end model output (C++ vs Python) ===\n");

    uintptr_t ctx = deepvqe_new(model_path);
    if (!ctx) {
        printf("  FAIL: could not load model %s\n\n", model_path);
        return false;
    }

    int hop = deepvqe_hop_length(ctx);
    bool all_ok = true;

    for (int i = 0; i < n_samples; i++) {
        char mic_path[512], ref_path[512], py_path[512];
        snprintf(mic_path, sizeof(mic_path), "%s/mic_%04d.npy", val_dir, i);
        snprintf(ref_path, sizeof(ref_path), "%s/ref_%04d.npy", val_dir, i);
        snprintf(py_path, sizeof(py_path), "%s/val_enhanced_py_%04d.npy", ref_dir, i);

        NpyArray mic_npy, ref_npy, py_enh;
        try {
            mic_npy = npy_load(mic_path);
            ref_npy = npy_load(ref_path);
            py_enh = npy_load(py_path);
        } catch (...) {
            if (i == 0) {
                printf("  SKIP (no reference data at %s)\n\n", py_path);
                deepvqe_free(ctx);
                return true;
            }
            break;
        }

        int len = (int)std::min({mic_npy.numel(), ref_npy.numel(), py_enh.numel()});
        std::vector<float> cpp_enh(len);
        deepvqe_reset(ctx);
        int ret = deepvqe_process_f32(ctx, mic_npy.data.data(), ref_npy.data.data(),
                                       len, cpp_enh.data());
        if (ret != 0) {
            printf("  [%d] FAIL: process error %d\n", i, ret);
            all_ok = false;
            continue;
        }

        // Compare, skipping first 2 hops (edge effects)
        int skip = 2 * hop;
        int cmp_len = len - skip;
        if (cmp_len <= 0) continue;

        float max_err = max_abs_diff(cpp_enh.data() + skip, py_enh.data.data() + skip, cmp_len);
        float mean_err = mean_abs_diff(cpp_enh.data() + skip, py_enh.data.data() + skip, cmp_len);

        // RMS of both outputs
        float rms_cpp = 0, rms_py = 0;
        for (int j = skip; j < len; j++) {
            rms_cpp += cpp_enh[j] * cpp_enh[j];
            rms_py += py_enh.data[j] * py_enh.data[j];
        }
        rms_cpp = sqrtf(rms_cpp / cmp_len);
        rms_py = sqrtf(rms_py / cmp_len);

        // Python reference is generated on CPU to match GGML. STFT precision
        // (KissFFT vs torch FFT) compounds through GRU but stays < 1e-3 in PCM.
        bool ok = max_err < 1e-3f;
        printf("  [%d] max=%.2e mean=%.2e  rms_cpp=%.4f rms_py=%.4f  %s\n",
               i, max_err, mean_err, rms_cpp, rms_py, ok ? "OK" : "FAIL");
        if (!ok) all_ok = false;
    }

    deepvqe_free(ctx);
    printf("  %s\n\n", all_ok ? "PASS" : "FAIL");
    return all_ok;
}

// ── Test 4: Feed Python STFT into GGML graph, compare enhanced STFT ───────

static bool test_stft_bypass(const char* model_path, const char* ref_dir, int n_samples) {
    printf("=== Test 4: Python STFT → GGML graph → compare enhanced STFT ===\n");

    dvqe_graph_model model;
    if (!load_graph_model(model_path, model, false)) {
        printf("  FAIL: could not load model\n\n");
        return false;
    }

    bool all_ok = true;

    for (int i = 0; i < n_samples; i++) {
        char mic_stft_path[512], ref_stft_path[512], py_enh_stft_path[512];
        snprintf(mic_stft_path, sizeof(mic_stft_path),
                 "%s/val_mic_stft_%04d.npy", ref_dir, i);
        snprintf(ref_stft_path, sizeof(ref_stft_path),
                 "%s/val_ref_stft_%04d.npy", ref_dir, i);
        snprintf(py_enh_stft_path, sizeof(py_enh_stft_path),
                 "%s/val_enhanced_stft_py_%04d.npy", ref_dir, i);

        NpyArray mic_stft, ref_stft, py_enh_stft;
        try {
            mic_stft = npy_load(mic_stft_path);
            ref_stft = npy_load(ref_stft_path);
            py_enh_stft = npy_load(py_enh_stft_path);
        } catch (...) {
            if (i == 0) {
                printf("  SKIP (no STFT reference data)\n\n");
                free_graph_model(model);
                return true;
            }
            break;
        }

        // Run GGML forward_graph with Python's STFT input
        NpyArray cpp_enh_stft = forward_graph(mic_stft, ref_stft, model, nullptr, false);

        int64_t n = std::min(cpp_enh_stft.numel(), py_enh_stft.numel());
        float max_err = max_abs_diff(cpp_enh_stft.data.data(),
                                      py_enh_stft.data.data(), n);
        float mean_err = mean_abs_diff(cpp_enh_stft.data.data(),
                                        py_enh_stft.data.data(), n);

        // Python reference is generated on CPU to match GGML exactly.
        // STFT precision (KissFFT vs torch FFT) is ~1e-5, model output ~1e-4.
        bool ok = max_err < 1e-3f;
        printf("  [%d] max=%.2e mean=%.2e  %s\n", i, max_err, mean_err,
               ok ? "OK" : "FAIL");
        if (!ok) all_ok = false;
    }

    free_graph_model(model);
    printf("  %s\n\n", all_ok ? "PASS" : "FAIL");
    return all_ok;
}

// ── Test 5: C++ STFT → forward_graph (batch) vs Python model output ───────

static bool test_batch_vs_py(const char* model_path, const char* val_dir,
                              const char* ref_dir, int n_samples) {
    printf("=== Test 5: C++ STFT → batch forward_graph → compare enhanced STFT ===\n");

    dvqe_graph_model model;
    if (!load_graph_model(model_path, model, false)) {
        printf("  FAIL: could not load model\n\n");
        return false;
    }

    auto& hp = model.hparams;
    int n_fft = hp.n_fft, hop = hp.hop_length, n_freq = n_fft / 2 + 1;
    auto window = make_sqrt_hann(n_fft);
    bool all_ok = true;

    for (int i = 0; i < n_samples; i++) {
        char mic_path[512], ref_path[512], py_enh_stft_path[512];
        snprintf(mic_path, sizeof(mic_path), "%s/mic_%04d.npy", val_dir, i);
        snprintf(ref_path, sizeof(ref_path), "%s/ref_%04d.npy", val_dir, i);
        snprintf(py_enh_stft_path, sizeof(py_enh_stft_path),
                 "%s/val_enhanced_stft_py_%04d.npy", ref_dir, i);

        NpyArray mic_pcm, ref_pcm, py_enh_stft;
        try {
            mic_pcm = npy_load(mic_path);
            ref_pcm = npy_load(ref_path);
            py_enh_stft = npy_load(py_enh_stft_path);
        } catch (...) { break; }

        int N = (int)std::min(mic_pcm.numel(), ref_pcm.numel());
        int n_frames = stft_n_frames(N, hop);

        // Compute C++ STFT
        NpyArray mic_stft, ref_stft;
        mic_stft.shape = {1, (int64_t)n_freq, (int64_t)n_frames, 2};
        mic_stft.data.resize(n_freq * n_frames * 2);
        ref_stft.shape = {1, (int64_t)n_freq, (int64_t)n_frames, 2};
        ref_stft.data.resize(n_freq * n_frames * 2);

        compute_stft(mic_pcm.data.data(), N, n_fft, hop, window.data(),
                     mic_stft.data.data(), n_freq, n_frames);
        compute_stft(ref_pcm.data.data(), N, n_fft, hop, window.data(),
                     ref_stft.data.data(), n_freq, n_frames);

        // Run batch forward_graph
        NpyArray cpp_enh = forward_graph(mic_stft, ref_stft, model, nullptr, false);

        int64_t n = std::min(cpp_enh.numel(), py_enh_stft.numel());
        float max_err = max_abs_diff(cpp_enh.data.data(), py_enh_stft.data.data(), n);
        float mean_err = mean_abs_diff(cpp_enh.data.data(), py_enh_stft.data.data(), n);

        bool ok = max_err < 1e-3f;
        printf("  [%d] max=%.2e mean=%.2e  %s\n", i, max_err, mean_err,
               ok ? "OK" : "FAIL");
        if (!ok) all_ok = false;
    }

    free_graph_model(model);
    printf("  %s\n\n", all_ok ? "PASS" : "FAIL");
    return all_ok;
}

// ── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    const char* ref_input = nullptr;
    const char* ref_stft = nullptr;
    const char* model_path = nullptr;
    const char* val_dir = nullptr;
    const char* py_ref_dir = nullptr;
    int n_samples = 3;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--ref-input" && i + 1 < argc) ref_input = argv[++i];
        else if (arg == "--ref-stft" && i + 1 < argc) ref_stft = argv[++i];
        else if (arg == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (arg == "--val-dir" && i + 1 < argc) val_dir = argv[++i];
        else if (arg == "--py-ref-dir" && i + 1 < argc) py_ref_dir = argv[++i];
        else if (arg == "--n-samples" && i + 1 < argc) n_samples = std::stoi(argv[++i]);
    }

    bool all_pass = true;

    // Test 1: always runs
    if (!test_roundtrip()) all_pass = false;

    // Test 2: only if reference data provided
    if (ref_input && ref_stft) {
        if (!test_python_reference(ref_input, ref_stft)) all_pass = false;
    } else {
        printf("=== Test 2: Python STFT reference comparison ===\n");
        printf("  SKIP (use --ref-input INPUT.npy --ref-stft REF.npy)\n\n");
    }

    // Test 3: only if model, val_dir, and py_ref_dir provided
    if (model_path && val_dir && py_ref_dir) {
        if (!test_model_e2e(model_path, val_dir, py_ref_dir, n_samples)) all_pass = false;
    } else {
        printf("=== Test 3: End-to-end model output (C++ vs Python) ===\n");
        printf("  SKIP (use --model MODEL.gguf --val-dir DIR --py-ref-dir DIR)\n\n");
    }

    // Test 4: feed Python STFT into GGML graph (bypasses C++ STFT)
    if (model_path && py_ref_dir) {
        if (!test_stft_bypass(model_path, py_ref_dir, n_samples)) all_pass = false;
    } else {
        printf("=== Test 4: Python STFT → GGML graph → compare enhanced STFT ===\n");
        printf("  SKIP (use --model MODEL.gguf --py-ref-dir DIR)\n\n");
    }

    // Test 5: C++ STFT → batch forward_graph vs Python model output
    if (model_path && val_dir && py_ref_dir) {
        if (!test_batch_vs_py(model_path, val_dir, py_ref_dir, n_samples)) all_pass = false;
    } else {
        printf("=== Test 5: C++ STFT → batch forward_graph → compare ===\n");
        printf("  SKIP\n\n");
    }

    printf(all_pass ? "All STFT tests PASSED\n" : "Some STFT tests FAILED\n");
    return all_pass ? 0 : 1;
}
