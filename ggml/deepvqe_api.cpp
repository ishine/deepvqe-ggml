/**
 * DeepVQE C API implementation.
 *
 * All inference uses the GGML streaming graph (T=1 per frame).
 * Batch processing = loop over frames. No hand-rolled forward pass.
 */

#include "deepvqe_api.h"
#include "deepvqe_graph.h"

#include "kiss_fftr.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>

// ── Internal context ─────────────────────────────────────────────────────────

struct deepvqe_ctx {
    dvqe_graph_model graph_model;
    dvqe_stream_graph stream_graph;

    std::string last_error;
    std::vector<float> window;  // sqrt-Hann, length n_fft

    // KissFFT configs (persistent, avoid per-frame allocation)
    kiss_fftr_cfg fft_fwd = nullptr;
    kiss_fftr_cfg fft_inv = nullptr;

    // PCM-domain streaming state (for frame-by-frame API)
    std::vector<float> stft_prev_mic;
    std::vector<float> stft_prev_ref;
    std::vector<float> istft_prev;
    int frame_count = 0;

    // Fixed-size scratch (allocated once at startup)
    std::vector<float> fft_scratch;    // n_fft reals, for windowed samples
    std::vector<float> mic_stft_buf;   // n_freq * 2, streaming STFT output / batch cpx scratch
    std::vector<float> ref_stft_buf;   // n_freq * 2
    std::vector<float> enh_stft_buf;   // n_freq * 2
    std::vector<float> frame_discard;  // hop, for warmup frame
    std::vector<float> s16_conv_buf;   // 3 * hop, for s16 frame conversion

    // Batch working buffers (grow-only, never shrink — zero allocation after first call)
    std::vector<float> batch_stft_a;
    std::vector<float> batch_stft_b;
    std::vector<float> batch_stft_enh;
    std::vector<float> batch_istft_out;
    std::vector<float> batch_istft_wsum;
    std::vector<float> batch_s16_a;
    std::vector<float> batch_s16_b;
    std::vector<float> batch_s16_out;
};

// Grow vector to at least n elements; never shrinks.
static void ensure_size(std::vector<float>& v, size_t n) {
    if (v.size() < n) v.resize(n);
}

// ── C API ────────────────────────────────────────────────────────────────────

extern "C" {

DEEPVQE_API uintptr_t deepvqe_new(const char* model_path) {
    auto* ctx = new (std::nothrow) deepvqe_ctx;
    if (!ctx) return 0;

    int n_threads = 0;
    const char* env_threads = std::getenv("GGML_NTHREADS");
    if (env_threads) n_threads = std::atoi(env_threads);

    if (!load_graph_model(model_path, ctx->graph_model, false, n_threads)) {
        delete ctx;
        return 0;
    }

    if (!build_stream_graph(ctx->graph_model, ctx->stream_graph)) {
        free_graph_model(ctx->graph_model);
        delete ctx;
        return 0;
    }

    int n_fft = ctx->graph_model.hparams.n_fft;
    int hop = ctx->graph_model.hparams.hop_length;
    int n_freq = n_fft / 2 + 1;
    ctx->window = make_sqrt_hann(n_fft);
    ctx->stft_prev_mic.resize(hop, 0.0f);
    ctx->stft_prev_ref.resize(hop, 0.0f);
    ctx->istft_prev.resize(n_fft, 0.0f);
    ctx->frame_count = 0;
    ctx->fft_fwd = kiss_fftr_alloc(n_fft, 0, nullptr, nullptr);
    ctx->fft_inv = kiss_fftr_alloc(n_fft, 1, nullptr, nullptr);
    ctx->fft_scratch.resize(n_fft, 0.0f);
    ctx->mic_stft_buf.resize(n_freq * 2, 0.0f);
    ctx->ref_stft_buf.resize(n_freq * 2, 0.0f);
    ctx->enh_stft_buf.resize(n_freq * 2, 0.0f);
    ctx->frame_discard.resize(hop, 0.0f);
    ctx->s16_conv_buf.resize(3 * hop, 0.0f);

    return reinterpret_cast<uintptr_t>(ctx);
}

DEEPVQE_API void deepvqe_free(uintptr_t handle) {
    if (!handle) return;
    auto* ctx = reinterpret_cast<deepvqe_ctx*>(handle);
    if (ctx->fft_fwd) kiss_fftr_free(ctx->fft_fwd);
    if (ctx->fft_inv) kiss_fftr_free(ctx->fft_inv);
    free_stream_graph(ctx->stream_graph);
    free_graph_model(ctx->graph_model);
    delete ctx;
}

DEEPVQE_API int deepvqe_process_f32(uintptr_t handle,
                                     const float* mic, const float* ref,
                                     int n_samples, float* out) {
    if (!handle) return -1;
    auto* ctx = reinterpret_cast<deepvqe_ctx*>(handle);
    ctx->last_error.clear();

    auto& hp = ctx->graph_model.hparams;
    int n_fft  = hp.n_fft;
    int hop    = hp.hop_length;
    int n_freq = n_fft / 2 + 1;
    int pad    = n_fft / 2;

    if (n_samples < n_fft) {
        ctx->last_error = "Input too short: need at least " + std::to_string(n_fft) + " samples";
        return -2;
    }

    int n_frames = stft_n_frames(n_samples, hop);
    int stft_size = n_freq * n_frames * 2;
    int padded_len = (n_frames - 1) * hop + n_fft;

    // Grow batch buffers (no-op after first call of this size)
    ensure_size(ctx->batch_stft_a, stft_size);
    ensure_size(ctx->batch_stft_b, stft_size);
    ensure_size(ctx->batch_stft_enh, stft_size);
    ensure_size(ctx->batch_istft_out, padded_len);
    ensure_size(ctx->batch_istft_wsum, padded_len);

    // Shared scratch for STFT/iSTFT (persistent FFT configs + buffers)
    stft_buffers bufs = {};
    bufs.fwd_cfg  = ctx->fft_fwd;
    bufs.inv_cfg  = ctx->fft_inv;
    bufs.scratch  = ctx->fft_scratch.data();
    bufs.cpx_buf  = ctx->enh_stft_buf.data();
    bufs.ola_out  = ctx->batch_istft_out.data();
    bufs.ola_wsum = ctx->batch_istft_wsum.data();

    compute_stft(mic, n_samples, n_fft, hop, ctx->window.data(),
                 ctx->batch_stft_a.data(), n_freq, n_frames, &bufs);
    compute_stft(ref, n_samples, n_fft, hop, ctx->window.data(),
                 ctx->batch_stft_b.data(), n_freq, n_frames, &bufs);

    // Process frame-by-frame through streaming GGML graph
    reset_stream_graph(ctx->stream_graph, ctx->graph_model);

    float* enh_stft = ctx->batch_stft_enh.data();
    for (int t = 0; t < n_frames; t++) {
        extract_stft_frame(ctx->batch_stft_a.data(), n_freq, n_frames, t, ctx->mic_stft_buf.data());
        extract_stft_frame(ctx->batch_stft_b.data(), n_freq, n_frames, t, ctx->ref_stft_buf.data());

        process_frame_graph(ctx->stream_graph, ctx->graph_model,
                            ctx->mic_stft_buf.data(), ctx->ref_stft_buf.data(),
                            ctx->enh_stft_buf.data());

        scatter_stft_frame(ctx->enh_stft_buf.data(), enh_stft, n_freq, n_frames, t);
    }

    // iSTFT reuses same persistent buffers
    bufs.cpx_buf = ctx->mic_stft_buf.data();
    compute_istft(enh_stft, n_freq, n_frames,
                  n_fft, hop, ctx->window.data(), out, n_samples, &bufs);

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
    auto* ctx = reinterpret_cast<deepvqe_ctx*>(handle);

    ensure_size(ctx->batch_s16_a, n_samples);
    ensure_size(ctx->batch_s16_b, n_samples);
    ensure_size(ctx->batch_s16_out, n_samples);

    s16_to_f32(mic, ctx->batch_s16_a.data(), n_samples);
    s16_to_f32(ref, ctx->batch_s16_b.data(), n_samples);

    int ret = deepvqe_process_f32(handle, ctx->batch_s16_a.data(),
                                   ctx->batch_s16_b.data(), n_samples,
                                   ctx->batch_s16_out.data());
    if (ret != 0) return ret;

    f32_to_s16(ctx->batch_s16_out.data(), out, n_samples);
    return 0;
}

DEEPVQE_API const char* deepvqe_last_error(uintptr_t handle) {
    if (!handle) return "null context";
    return reinterpret_cast<deepvqe_ctx*>(handle)->last_error.c_str();
}

DEEPVQE_API int deepvqe_sample_rate(uintptr_t handle) {
    if (!handle) return 0;
    return reinterpret_cast<deepvqe_ctx*>(handle)->graph_model.hparams.sample_rate;
}

DEEPVQE_API int deepvqe_hop_length(uintptr_t handle) {
    if (!handle) return 0;
    return reinterpret_cast<deepvqe_ctx*>(handle)->graph_model.hparams.hop_length;
}

DEEPVQE_API int deepvqe_fft_size(uintptr_t handle) {
    if (!handle) return 0;
    return reinterpret_cast<deepvqe_ctx*>(handle)->graph_model.hparams.n_fft;
}

// ── Streaming STFT/iSTFT ────────────────────────────────────────────────

static void stft_frame(const float* new_hop, const float* prev_hop,
                       const float* window, int n_fft,
                       float* out, int n_freq,
                       kiss_fftr_cfg cfg, float* scratch) {
    for (int i = 0; i < n_fft; i++) {
        float sample = (i < n_fft / 2) ? prev_hop[i] : new_hop[i - n_fft / 2];
        scratch[i] = sample * window[i];
    }
    kiss_fftr(cfg, scratch, reinterpret_cast<kiss_fft_cpx*>(out));
}

static void istft_frame(const float* enhanced_stft, float* prev_buf,
                        const float* window, int n_fft, int hop,
                        float* out,
                        kiss_fftr_cfg icfg, float* scratch) {
    kiss_fftri(icfg, reinterpret_cast<const kiss_fft_cpx*>(enhanced_stft), scratch);

    // KissFFT does not normalize inverse — scale by 1/n_fft
    float inv_n = 1.0f / n_fft;
    for (int i = 0; i < n_fft; i++)
        prev_buf[i] += scratch[i] * inv_n * window[i];

    std::memcpy(out, prev_buf, hop * sizeof(float));

    std::memmove(prev_buf, prev_buf + hop, hop * sizeof(float));
    std::memset(prev_buf + hop, 0, hop * sizeof(float));
}

// ── Streaming C API ──────────────────────────────────────────────────────

DEEPVQE_API int deepvqe_process_frame_f32(uintptr_t handle,
                                           const float* mic, const float* ref,
                                           int hop_samples, float* out) {
    if (!handle) return -1;
    auto* ctx = reinterpret_cast<deepvqe_ctx*>(handle);

    auto& hp = ctx->graph_model.hparams;
    if (hop_samples != hp.hop_length) return -2;

    int n_fft = hp.n_fft;
    int hop = hp.hop_length;
    int n_freq = n_fft / 2 + 1;

    float* scratch = ctx->fft_scratch.data();
    stft_frame(mic, ctx->stft_prev_mic.data(), ctx->window.data(),
               n_fft, ctx->mic_stft_buf.data(), n_freq, ctx->fft_fwd, scratch);
    stft_frame(ref, ctx->stft_prev_ref.data(), ctx->window.data(),
               n_fft, ctx->ref_stft_buf.data(), n_freq, ctx->fft_fwd, scratch);

    std::memcpy(ctx->stft_prev_mic.data(), mic, hop * sizeof(float));
    std::memcpy(ctx->stft_prev_ref.data(), ref, hop * sizeof(float));

    process_frame_graph(ctx->stream_graph, ctx->graph_model,
                        ctx->mic_stft_buf.data(), ctx->ref_stft_buf.data(),
                        ctx->enh_stft_buf.data());

    // Frame 0 outputs zeros (center padding region)
    float* dst = (ctx->frame_count == 0) ? ctx->frame_discard.data() : out;
    istft_frame(ctx->enh_stft_buf.data(), ctx->istft_prev.data(),
                ctx->window.data(), n_fft, hop, dst, ctx->fft_inv, scratch);
    if (ctx->frame_count == 0) {
        std::memset(out, 0, hop * sizeof(float));
    }
    ctx->frame_count++;
    return 0;
}

DEEPVQE_API int deepvqe_process_frame_s16(uintptr_t handle,
                                           const int16_t* mic, const int16_t* ref,
                                           int hop_samples, int16_t* out) {
    if (!handle) return -1;
    auto* ctx = reinterpret_cast<deepvqe_ctx*>(handle);

    // Use pre-allocated conversion buffers (3 * hop laid out contiguously)
    float* mic_f = ctx->s16_conv_buf.data();
    float* ref_f = mic_f + hop_samples;
    float* out_f = ref_f + hop_samples;
    s16_to_f32(mic, mic_f, hop_samples);
    s16_to_f32(ref, ref_f, hop_samples);

    int ret = deepvqe_process_frame_f32(handle, mic_f, ref_f, hop_samples, out_f);
    if (ret != 0) return ret;

    f32_to_s16(out_f, out, hop_samples);
    return 0;
}

DEEPVQE_API void deepvqe_reset(uintptr_t handle) {
    if (!handle) return;
    auto* ctx = reinterpret_cast<deepvqe_ctx*>(handle);
    reset_stream_graph(ctx->stream_graph, ctx->graph_model);
    std::fill(ctx->stft_prev_mic.begin(), ctx->stft_prev_mic.end(), 0.0f);
    std::fill(ctx->stft_prev_ref.begin(), ctx->stft_prev_ref.end(), 0.0f);
    std::fill(ctx->istft_prev.begin(), ctx->istft_prev.end(), 0.0f);
    ctx->frame_count = 0;
}

} // extern "C"
