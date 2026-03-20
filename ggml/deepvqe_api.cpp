/**
 * DeepVQE C API implementation.
 *
 * Provides a purego-compatible shared library interface:
 *   - Accepts raw PCM audio (float32 or int16)
 *   - Handles STFT/iSTFT internally
 *   - Returns enhanced PCM audio
 */

#include "deepvqe_api.h"
#include "deepvqe_model.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <new>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ── Internal context ─────────────────────────────────────────────────────────

struct deepvqe_ctx {
    deepvqe_model model;
    std::string last_error;
    std::vector<float> window;  // sqrt-Hann, length n_fft
};

// ── FFT (radix-2 Cooley-Tukey, in-place, interleaved complex) ────────────────

static void fft_radix2(float* x, int n, bool inverse) {
    // Bit-reversal permutation
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            std::swap(x[2*i], x[2*j]);
            std::swap(x[2*i+1], x[2*j+1]);
        }
    }

    // Butterfly stages
    for (int len = 2; len <= n; len <<= 1) {
        float angle = (inverse ? 1.0f : -1.0f) * 2.0f * (float)M_PI / len;
        float wR = cosf(angle), wI = sinf(angle);
        for (int i = 0; i < n; i += len) {
            float curR = 1.0f, curI = 0.0f;
            for (int j = 0; j < len / 2; j++) {
                int u = i + j, v = i + j + len / 2;
                float tR = curR * x[2*v] - curI * x[2*v+1];
                float tI = curR * x[2*v+1] + curI * x[2*v];
                x[2*v] = x[2*u] - tR;
                x[2*v+1] = x[2*u+1] - tI;
                x[2*u] += tR;
                x[2*u+1] += tI;
                float newR = curR * wR - curI * wI;
                curI = curR * wI + curI * wR;
                curR = newR;
            }
        }
    }

    if (inverse) {
        float inv_n = 1.0f / n;
        for (int i = 0; i < 2 * n; i++) x[i] *= inv_n;
    }
}

// ── STFT / iSTFT ─────────────────────────────────────────────────────────────

static void make_sqrt_hann(float* window, int n) {
    for (int i = 0; i < n; i++) {
        float hann = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / n));
        window[i] = sqrtf(hann + 1e-12f);
    }
}

// Reflect-pad index helper (mirrors at boundaries without including boundary).
static inline int reflect(int idx, int len) {
    if (idx < 0) idx = -idx;
    if (idx >= len) idx = 2 * (len - 1) - idx;
    return idx;
}

// STFT with center=True, reflection padding (matches torch.stft defaults).
// Output layout: (F, T, 2) where F = n_fft/2 + 1.
static void compute_stft(const float* signal, int N,
                         int n_fft, int hop, const float* window,
                         float* out, int n_freq, int n_frames) {
    int pad = n_fft / 2;

    std::vector<float> frame(2 * n_fft);

    for (int t = 0; t < n_frames; t++) {
        int center = t * hop;

        // Window the frame with reflection padding
        for (int i = 0; i < n_fft; i++) {
            int src = center - pad + i;
            int idx = reflect(src, N);
            frame[2*i] = signal[idx] * window[i];
            frame[2*i+1] = 0.0f;
        }

        // FFT
        fft_radix2(frame.data(), n_fft, false);

        // Store positive frequencies: out[f * n_frames * 2 + t * 2 + {0,1}]
        for (int f = 0; f < n_freq; f++) {
            out[f * n_frames * 2 + t * 2 + 0] = frame[2*f];
            out[f * n_frames * 2 + t * 2 + 1] = frame[2*f+1];
        }
    }
}

// iSTFT with center=True, overlap-add (matches torch.istft).
static void compute_istft(const float* stft_data, int n_freq, int n_frames,
                          int n_fft, int hop, const float* window,
                          float* signal, int N) {
    int pad = n_fft / 2;
    int padded_len = (n_frames - 1) * hop + n_fft;

    std::vector<float> output(padded_len, 0.0f);
    std::vector<float> window_sum(padded_len, 0.0f);
    std::vector<float> frame(2 * n_fft);

    for (int t = 0; t < n_frames; t++) {
        // Reconstruct full spectrum from half spectrum (conjugate symmetry)
        for (int f = 0; f < n_freq; f++) {
            frame[2*f]   = stft_data[f * n_frames * 2 + t * 2 + 0];
            frame[2*f+1] = stft_data[f * n_frames * 2 + t * 2 + 1];
        }
        for (int f = n_freq; f < n_fft; f++) {
            int mirror = n_fft - f;
            frame[2*f]   =  frame[2*mirror];
            frame[2*f+1] = -frame[2*mirror+1];
        }

        // IFFT
        fft_radix2(frame.data(), n_fft, true);

        // Synthesis window and overlap-add
        int start = t * hop;
        for (int i = 0; i < n_fft; i++) {
            output[start + i]     += frame[2*i] * window[i];
            window_sum[start + i] += window[i] * window[i];
        }
    }

    // Normalize and strip center padding
    float threshold = 1e-11f;
    for (int i = 0; i < N; i++) {
        float ws = window_sum[i + pad];
        signal[i] = ws > threshold ? output[i + pad] / ws : 0.0f;
    }
}

// ── C API ────────────────────────────────────────────────────────────────────

extern "C" {

DEEPVQE_API uintptr_t deepvqe_new(const char* model_path) {
    auto* ctx = new (std::nothrow) deepvqe_ctx;
    if (!ctx) return 0;

    if (!load_model(model_path, ctx->model, /*verbose=*/false)) {
        delete ctx;
        return 0;
    }

    int n_fft = ctx->model.hparams.n_fft;
    ctx->window.resize(n_fft);
    make_sqrt_hann(ctx->window.data(), n_fft);

    return reinterpret_cast<uintptr_t>(ctx);
}

DEEPVQE_API void deepvqe_free(uintptr_t handle) {
    if (!handle) return;
    delete reinterpret_cast<deepvqe_ctx*>(handle);
}

DEEPVQE_API int deepvqe_process_f32(uintptr_t handle,
                                     const float* mic, const float* ref,
                                     int n_samples, float* out) {
    if (!handle) return -1;
    auto* ctx = reinterpret_cast<deepvqe_ctx*>(handle);
    ctx->last_error.clear();

    auto& hp = ctx->model.hparams;
    int n_fft  = hp.n_fft;
    int hop    = hp.hop_length;
    int n_freq = n_fft / 2 + 1;

    if (n_samples < n_fft) {
        ctx->last_error = "Input too short: need at least " + std::to_string(n_fft) + " samples";
        return -2;
    }

    int n_frames = n_samples / hop + 1;

    // Compute STFTs: output layout (F, T, 2) matching model expectation
    int stft_size = n_freq * n_frames * 2;
    NpyArray mic_stft, ref_stft;
    mic_stft.shape = {1, (int64_t)n_freq, (int64_t)n_frames, 2};
    mic_stft.data.resize(stft_size);
    ref_stft.shape = {1, (int64_t)n_freq, (int64_t)n_frames, 2};
    ref_stft.data.resize(stft_size);

    compute_stft(mic, n_samples, n_fft, hop, ctx->window.data(),
                 mic_stft.data.data(), n_freq, n_frames);
    compute_stft(ref, n_samples, n_fft, hop, ctx->window.data(),
                 ref_stft.data.data(), n_freq, n_frames);

    // Forward pass (silent)
    NpyArray enhanced;
    try {
        enhanced = forward(mic_stft, ref_stft, ctx->model, "", /*verbose=*/false);
    } catch (const std::exception& e) {
        ctx->last_error = std::string("Forward pass failed: ") + e.what();
        return -3;
    }

    // iSTFT back to time domain
    compute_istft(enhanced.data.data(), n_freq, n_frames,
                  n_fft, hop, ctx->window.data(), out, n_samples);

    return 0;
}

DEEPVQE_API int deepvqe_process_s16(uintptr_t handle,
                                     const int16_t* mic, const int16_t* ref,
                                     int n_samples, int16_t* out) {
    if (!handle) return -1;

    // Convert s16 -> f32
    std::vector<float> mic_f(n_samples), ref_f(n_samples), out_f(n_samples);
    const float scale_in = 1.0f / 32768.0f;
    for (int i = 0; i < n_samples; i++) {
        mic_f[i] = mic[i] * scale_in;
        ref_f[i] = ref[i] * scale_in;
    }

    int ret = deepvqe_process_f32(handle, mic_f.data(), ref_f.data(), n_samples, out_f.data());
    if (ret != 0) return ret;

    // Convert f32 -> s16 with clamping
    for (int i = 0; i < n_samples; i++) {
        float v = out_f[i] * 32768.0f;
        v = std::max(-32768.0f, std::min(32767.0f, v));
        out[i] = (int16_t)v;
    }

    return 0;
}

DEEPVQE_API const char* deepvqe_last_error(uintptr_t handle) {
    if (!handle) return "null context";
    return reinterpret_cast<deepvqe_ctx*>(handle)->last_error.c_str();
}

DEEPVQE_API int deepvqe_sample_rate(uintptr_t handle) {
    if (!handle) return 0;
    return reinterpret_cast<deepvqe_ctx*>(handle)->model.hparams.sample_rate;
}

DEEPVQE_API int deepvqe_hop_length(uintptr_t handle) {
    if (!handle) return 0;
    return reinterpret_cast<deepvqe_ctx*>(handle)->model.hparams.hop_length;
}

DEEPVQE_API int deepvqe_fft_size(uintptr_t handle) {
    if (!handle) return 0;
    return reinterpret_cast<deepvqe_ctx*>(handle)->model.hparams.n_fft;
}

} // extern "C"
