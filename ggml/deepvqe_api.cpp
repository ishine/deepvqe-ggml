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
    deepvqe_stream_state stream;
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

    init_stream_state(ctx->stream, ctx->model);

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

// ── s16/f32 conversion helpers ────────────────────────────────────────────

static void s16_to_f32(const int16_t* in, float* out, int n) {
    const float scale = 1.0f / 32768.0f;
    for (int i = 0; i < n; i++) out[i] = in[i] * scale;
}

static void f32_to_s16(const float* in, int16_t* out, int n) {
    for (int i = 0; i < n; i++) {
        float v = std::max(-32768.0f, std::min(32767.0f, in[i] * 32768.0f));
        out[i] = (int16_t)v;
    }
}

DEEPVQE_API int deepvqe_process_s16(uintptr_t handle,
                                     const int16_t* mic, const int16_t* ref,
                                     int n_samples, int16_t* out) {
    if (!handle) return -1;

    std::vector<float> mic_f(n_samples), ref_f(n_samples), out_f(n_samples);
    s16_to_f32(mic, mic_f.data(), n_samples);
    s16_to_f32(ref, ref_f.data(), n_samples);

    int ret = deepvqe_process_f32(handle, mic_f.data(), ref_f.data(), n_samples, out_f.data());
    if (ret != 0) return ret;

    f32_to_s16(out_f.data(), out, n_samples);
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

// ── Streaming STFT/iSTFT ─────────────────────────────────────────────────

// Single-frame STFT: prev_hop (256) + new_hop (256) → window + FFT → (n_freq, 2)
static void stft_frame(const float* new_hop, const float* prev_hop,
                       const float* window, int n_fft,
                       float* out, int n_freq) {
    std::vector<float> frame(2 * n_fft, 0.0f);
    for (int i = 0; i < n_fft; i++) {
        // First half from prev_hop, second half from new_hop
        float sample = (i < n_fft / 2) ? prev_hop[i] : new_hop[i - n_fft / 2];
        frame[2 * i] = sample * window[i];
    }
    fft_radix2(frame.data(), n_fft, false);
    // Store positive frequencies: (n_freq, 2) interleaved
    for (int f = 0; f < n_freq; f++) {
        out[f * 2 + 0] = frame[2 * f];
        out[f * 2 + 1] = frame[2 * f + 1];
    }
}

// Single-frame iSTFT: IFFT + window + overlap-add → hop_length output samples
static void istft_frame(const float* enhanced_stft, float* prev_buf,
                        const float* window, int n_fft, int hop,
                        float* out) {
    int n_freq = n_fft / 2 + 1;
    std::vector<float> frame(2 * n_fft, 0.0f);

    // Reconstruct full spectrum from half spectrum
    for (int f = 0; f < n_freq; f++) {
        frame[2 * f]     = enhanced_stft[f * 2 + 0];
        frame[2 * f + 1] = enhanced_stft[f * 2 + 1];
    }
    for (int f = n_freq; f < n_fft; f++) {
        int mirror = n_fft - f;
        frame[2 * f]     =  frame[2 * mirror];
        frame[2 * f + 1] = -frame[2 * mirror + 1];
    }

    // IFFT
    fft_radix2(frame.data(), n_fft, true);

    // Synthesis window + overlap-add
    // prev_buf holds the last n_fft samples of accumulated overlap
    // Output the first hop samples (completed region)
    for (int i = 0; i < n_fft; i++)
        prev_buf[i] += frame[2 * i] * window[i];

    // sqrt-Hann with 50% overlap: COLA = 1.0, no normalization needed
    std::memcpy(out, prev_buf, hop * sizeof(float));

    // Shift: move second half to first, zero second half
    std::memmove(prev_buf, prev_buf + hop, hop * sizeof(float));
    std::memset(prev_buf + hop, 0, hop * sizeof(float));
}

// ── Streaming C API ──────────────────────────────────────────────────────

DEEPVQE_API int deepvqe_process_frame_f32(uintptr_t handle,
                                           const float* mic, const float* ref,
                                           int hop_samples, float* out) {
    if (!handle) return -1;
    auto* ctx = reinterpret_cast<deepvqe_ctx*>(handle);
    ctx->last_error.clear();

    auto& hp = ctx->model.hparams;
    if (hop_samples != hp.hop_length) {
        ctx->last_error = "hop_samples must be " + std::to_string(hp.hop_length);
        return -2;
    }

    auto& st = ctx->stream;
    int n_fft = hp.n_fft;
    int hop = hp.hop_length;
    int n_freq = n_fft / 2 + 1;

    // First call: output zeros (need 2 hops for first valid STFT frame)
    if (st.frame_count == 0) {
        std::memcpy(st.stft_prev_mic.data(), mic, hop * sizeof(float));
        std::memcpy(st.stft_prev_ref.data(), ref, hop * sizeof(float));
        std::memset(out, 0, hop * sizeof(float));
        st.frame_count = 1;
        return 0;
    }

    // Single-frame STFT for mic and ref
    std::vector<float> mic_stft(n_freq * 2), ref_stft(n_freq * 2);
    stft_frame(mic, st.stft_prev_mic.data(), ctx->window.data(),
               n_fft, mic_stft.data(), n_freq);
    stft_frame(ref, st.stft_prev_ref.data(), ctx->window.data(),
               n_fft, ref_stft.data(), n_freq);

    // Update prev hop buffers
    std::memcpy(st.stft_prev_mic.data(), mic, hop * sizeof(float));
    std::memcpy(st.stft_prev_ref.data(), ref, hop * sizeof(float));

    // Forward pass (one frame)
    std::vector<float> enhanced_stft(n_freq * 2);
    try {
        forward_frame(mic_stft.data(), ref_stft.data(),
                      ctx->model, st, enhanced_stft.data());
    } catch (const std::exception& e) {
        ctx->last_error = std::string("Forward pass failed: ") + e.what();
        return -3;
    }

    // Single-frame iSTFT → hop output samples
    istft_frame(enhanced_stft.data(), st.istft_prev.data(),
                ctx->window.data(), n_fft, hop, out);
    return 0;
}

DEEPVQE_API int deepvqe_process_frame_s16(uintptr_t handle,
                                           const int16_t* mic, const int16_t* ref,
                                           int hop_samples, int16_t* out) {
    if (!handle) return -1;

    std::vector<float> mic_f(hop_samples), ref_f(hop_samples), out_f(hop_samples);
    s16_to_f32(mic, mic_f.data(), hop_samples);
    s16_to_f32(ref, ref_f.data(), hop_samples);

    int ret = deepvqe_process_frame_f32(handle, mic_f.data(), ref_f.data(),
                                         hop_samples, out_f.data());
    if (ret != 0) return ret;

    f32_to_s16(out_f.data(), out, hop_samples);
    return 0;
}

DEEPVQE_API void deepvqe_reset(uintptr_t handle) {
    if (!handle) return;
    auto* ctx = reinterpret_cast<deepvqe_ctx*>(handle);
    reset_stream_state(ctx->stream);
}

} // extern "C"
