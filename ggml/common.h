#pragma once

/**
 * Common utilities for DeepVQE GGML inference and block tests.
 *
 * - NumPy .npy file I/O (f32, C-contiguous only)
 * - Comparison helpers (max/mean absolute error)
 * - Result reporting
 */

#include <cstdint>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ── NumPy .npy I/O ──────────────────────────────────────────────────────────

struct NpyArray {
    std::vector<float> data;
    std::vector<int64_t> shape;

    int64_t numel() const;
    int64_t dim(int i) const { return shape[i]; }
    int ndim() const { return (int)shape.size(); }
};

/// Load a .npy file (float32, C-contiguous). Throws on error.
NpyArray npy_load(const std::string& path);

/// Save a .npy file (float32, C-contiguous).
void npy_save(const std::string& path, const float* data,
              const std::vector<int64_t>& shape);

inline void npy_save(const std::string& path, const NpyArray& arr) {
    npy_save(path, arr.data.data(), arr.shape);
}

// ── Audio I/O (requires libsndfile, used by tests only) ──────────────────────

#ifdef DEEPVQE_HAS_SNDFILE
/// Load an audio file (FLAC, WAV, etc.) as mono float32 [-1,1].
/// Resamples to target_sr if the file's sample rate differs (simple decimation).
/// Returns empty vector on failure.
std::vector<float> audio_load_mono(const std::string& path,
                                    int target_sr = 16000);
#endif

// ── GGUF tensor loading ────────────────────────────────────────────────────

struct ggml_context;  // forward decl — avoid pulling in ggml.h
struct gguf_context;

/// Load tensor from ggml context, dequantizing quantized types to float32.
/// If gctx is non-null, restores original shape from GGUF metadata
/// (quantized conv weights are stored flattened to 1D).
/// Returns empty NpyArray on failure.
NpyArray load_tensor_from_ggml(struct ggml_context* ctx,
                               const std::string& name,
                               struct gguf_context* gctx = nullptr,
                               bool verbose = false);

// ── Comparison ──────────────────────────────────────────────────────────────

/// Maximum absolute difference between two arrays.
float max_abs_diff(const float* a, const float* b, int64_t n);

/// Mean absolute difference between two arrays.
float mean_abs_diff(const float* a, const float* b, int64_t n);

/// Print comparison result with OK/WARN/FAIL classification.
/// Returns true if max_err < fail_threshold (1e-2).
bool print_result(const std::string& name, float max_err, float mean_err,
                  float ok_threshold = 1e-4f, float fail_threshold = 1e-2f);

// ── FFT / STFT / iSTFT ─────────────────────────────────────────────────────


/// Build sqrt-Hann window of length n (with 1e-12 epsilon for stability).
std::vector<float> make_sqrt_hann(int n);

/// Reflect-pad index helper (mirrors at boundaries, matching torch pad_mode="reflect").
inline int reflect_idx(int idx, int len) {
    if (idx < 0) idx = -idx;
    if (idx >= len) idx = 2 * (len - 1) - idx;
    return idx;
}

/// Pre-allocated scratch buffers for STFT/iSTFT (avoids per-call allocation).
/// Pass nullptr for any field to use internal temporary allocation.
/// fwd_cfg / inv_cfg are kiss_fftr_cfg (opaque pointer, cast to void* to avoid header dep).
struct stft_buffers {
    void* fwd_cfg = nullptr;           // kiss_fftr_cfg for forward FFT
    void* inv_cfg = nullptr;           // kiss_fftr_cfg for inverse FFT
    float* scratch = nullptr;          // n_fft floats (windowed samples / real output)
    float* cpx_buf = nullptr;          // n_freq * 2 floats (interleaved complex)
    float* ola_out = nullptr;          // padded_len floats (iSTFT overlap-add output)
    float* ola_wsum = nullptr;         // padded_len floats (iSTFT window sum)
};

/// Batch STFT with center=True, reflection padding (matches torch.stft defaults).
/// Output layout: out[f * n_frames * 2 + t * 2 + {0,1}], shape (F, T, 2).
void compute_stft(const float* signal, int N,
                  int n_fft, int hop, const float* window,
                  float* out, int n_freq, int n_frames,
                  stft_buffers* bufs = nullptr);

/// Compute number of STFT frames for a signal of length N (center=True).
/// Matches torch.stft: 1 + floor(N / hop).
inline int stft_n_frames(int N, int hop) { return N / hop + 1; }

/// Batch iSTFT with overlap-add and window normalization (matches torch.istft).
/// If bufs is provided, bufs->ola_out and ola_wsum must be at least padded_len floats.
void compute_istft(const float* stft_data, int n_freq, int n_frames,
                   int n_fft, int hop, const float* window,
                   float* signal, int N,
                   stft_buffers* bufs = nullptr);

/// Extract STFT frame t from (F, T, 2) layout into F*2 interleaved [f0_re, f0_im, ...].
inline void extract_stft_frame(const float* stft, int F, int T, int t, float* frame) {
    for (int f = 0; f < F; f++) {
        frame[f * 2 + 0] = stft[f * T * 2 + t * 2 + 0];
        frame[f * 2 + 1] = stft[f * T * 2 + t * 2 + 1];
    }
}

/// Scatter F*2 interleaved frame back into (F, T, 2) layout at time t.
inline void scatter_stft_frame(const float* frame, float* stft, int F, int T, int t) {
    for (int f = 0; f < F; f++) {
        stft[f * T * 2 + t * 2 + 0] = frame[f * 2 + 0];
        stft[f * T * 2 + t * 2 + 1] = frame[f * 2 + 1];
    }
}
