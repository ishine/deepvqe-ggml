#pragma once

/**
 * DeepVQE model inference using GGML computational graphs.
 *
 * Replaces the hand-rolled C++ loops in deepvqe_model.cpp with
 * GGML graph ops that can be dispatched to CPU (with SIMD) or GPU.
 */

#include "deepvqe_model.h"  // deepvqe_hparams, NpyArray

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <map>
#include <string>
#include <vector>

struct dvqe_graph_model {
    deepvqe_hparams hparams;

    // Backend (CPU or CUDA)
    ggml_backend_t backend = nullptr;

    // Weight storage — kept as ggml tensors (may be quantized)
    struct ggml_context* weight_ctx = nullptr;
    ggml_backend_buffer_t weight_buf = nullptr;

    // Named weight lookup
    std::map<std::string, struct ggml_tensor*> weights;

    struct ggml_tensor* w(const std::string& name) const {
        auto it = weights.find(name);
        if (it == weights.end()) return nullptr;
        return it->second;
    }
};

/// Load model into GGML graph format. Weights stay as ggml_tensor*
/// (potentially quantized). n_threads=0 means auto (nproc - 1, min 1).
bool load_graph_model(const char* path, dvqe_graph_model& model,
                      bool verbose = true, int n_threads = 0);

/// Free all resources.
void free_graph_model(dvqe_graph_model& model);

/// Per-block timing result.
struct block_timing {
    std::string name;
    double us;  // microseconds
};

// ── Streaming graph (T=1 per frame, with history buffers) ───────────────────

/// Pre-built GGML graph for frame-by-frame streaming inference.
/// Built once, reused for every frame. History persists between frames.
struct dvqe_stream_graph {
    struct ggml_context* ctx = nullptr;
    struct ggml_cgraph* graph = nullptr;
    ggml_gallocr_t galloc = nullptr;

    // Input tensors (set before each compute)
    struct ggml_tensor* mic_in = nullptr;   // (2, 1, F)
    struct ggml_tensor* ref_in = nullptr;   // (2, 1, F)

    // Per-conv-layer history: input (F_i, 3, C_i), output (F_i, 3, C_i)
    std::vector<struct ggml_tensor*> conv_hist_in;
    std::vector<struct ggml_tensor*> conv_hist_out;

    // GRU hidden state
    struct ggml_tensor* gru_h_in = nullptr;
    struct ggml_tensor* gru_h_out = nullptr;

    // AlignBlock: K and ref histories (dmax-1 frames each)
    struct ggml_tensor* align_K_hist_in = nullptr;
    struct ggml_tensor* align_K_hist_out = nullptr;
    struct ggml_tensor* align_ref_hist_in = nullptr;
    struct ggml_tensor* align_ref_hist_out = nullptr;

    // AlignBlock: smooth conv history (4 time frames)
    struct ggml_tensor* align_smooth_hist_in = nullptr;
    struct ggml_tensor* align_smooth_hist_out = nullptr;

    // CCM: mic STFT history (2 time frames)
    struct ggml_tensor* ccm_hist_in = nullptr;
    struct ggml_tensor* ccm_hist_out = nullptr;

    // Enhanced output (2, 1, F)
    struct ggml_tensor* enhanced_out = nullptr;

    // Persistent scratch for history copies (avoids per-frame heap allocation)
    std::vector<uint8_t> hist_scratch;
};

/// Build the streaming graph (T=1). Call once, then process_frame_graph() per frame.
bool build_stream_graph(dvqe_graph_model& m, dvqe_stream_graph& sg);

/// Process one STFT frame. mic/ref/out are F*2 interleaved complex floats.
/// Layout: [f0_re, f0_im, f1_re, f1_im, ...] (F = n_fft/2+1 bins).
void process_frame_graph(dvqe_stream_graph& sg, dvqe_graph_model& m,
                         const float* mic_stft_frame,
                         const float* ref_stft_frame,
                         float* enhanced_stft_frame);

/// Zero all history buffers (call before processing a new utterance).
void reset_stream_graph(dvqe_stream_graph& sg, dvqe_graph_model& m);

/// Free streaming graph resources.
void free_stream_graph(dvqe_stream_graph& sg);

/// Full forward pass using GGML graphs (batch = streaming loop).
/// Returns enhanced STFT as NpyArray (1, F, T, 2).
NpyArray forward_graph(const NpyArray& mic_stft, const NpyArray& ref_stft,
                       dvqe_graph_model& m,
                       std::vector<block_timing>* timings = nullptr,
                       bool verbose = true,
                       const std::string& dump_dir = "");
