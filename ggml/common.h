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

// ── Comparison ──────────────────────────────────────────────────────────────

/// Maximum absolute difference between two arrays.
float max_abs_diff(const float* a, const float* b, int64_t n);

/// Mean absolute difference between two arrays.
float mean_abs_diff(const float* a, const float* b, int64_t n);

/// Print comparison result with OK/WARN/FAIL classification.
/// Returns true if max_err < fail_threshold (1e-2).
bool print_result(const std::string& name, float max_err, float mean_err,
                  float ok_threshold = 1e-4f, float fail_threshold = 1e-2f);
