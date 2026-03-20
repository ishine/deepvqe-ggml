#pragma once

/**
 * DeepVQE model — internal C++ structs and inference.
 * Used by both the CLI tool and the C API shared library.
 */

#include "common.h"

#include <map>
#include <string>
#include <vector>

struct deepvqe_hparams {
    int n_fft        = 512;
    int hop_length   = 256;
    int n_freq_bins  = 257;
    int sample_rate  = 16000;
    int dmax         = 32;
    int align_hidden = 32;
    float power_law_c = 0.3f;
    bool bn_folded   = true;
    std::vector<int> mic_channels;
    std::vector<int> far_channels;
};

struct deepvqe_model {
    deepvqe_hparams hparams;
    std::map<std::string, NpyArray> tensors;
};

struct Buf {
    std::vector<float> data;
    int C, T, F;

    Buf() : C(0), T(0), F(0) {}
    Buf(int c, int t, int f) : data(c * t * f, 0.0f), C(c), T(t), F(f) {}

    int64_t numel() const { return (int64_t)C * T * F; }
    float& operator()(int c, int t, int f) { return data[c * T * F + t * F + f]; }
    const float& operator()(int c, int t, int f) const { return data[c * T * F + t * F + f]; }
    float* ptr() { return data.data(); }
    const float* ptr() const { return data.data(); }
};

/// Load model from GGUF file. Returns true on success.
bool load_model(const char* path, deepvqe_model& model, bool verbose = true);

/// Full forward pass. Returns enhanced STFT (1, F, T, 2).
/// dump_dir: if non-empty, save intermediates as .npy files.
NpyArray forward(const NpyArray& mic_stft, const NpyArray& ref_stft,
                 const deepvqe_model& m,
                 const std::string& dump_dir = "", bool verbose = true);
