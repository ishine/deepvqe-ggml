/**
 * Standalone STFT verification tests.
 *
 * Test 1: Roundtrip — sine wave → STFT → iSTFT → compare with original.
 * Test 2: Python reference — compare compute_stft() output against
 *         torch.stft reference saved as .npy.
 *
 * Usage:
 *   test_stft [--ref-input stft_input.npy --ref-stft stft_ref.npy]
 */

#include "common.h"

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

// ── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    const char* ref_input = nullptr;
    const char* ref_stft = nullptr;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--ref-input" && i + 1 < argc) ref_input = argv[++i];
        else if (arg == "--ref-stft" && i + 1 < argc) ref_stft = argv[++i];
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

    printf(all_pass ? "All STFT tests PASSED\n" : "Some STFT tests FAILED\n");
    return all_pass ? 0 : 1;
}
