/**
 * Common utilities for DeepVQE GGML inference and block tests.
 *
 * Implements:
 * - Minimal NumPy .npy reader/writer for f32 C-contiguous arrays
 * - Comparison helpers
 */

#include "common.h"

#include "ggml.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>

// ── NpyArray ────────────────────────────────────────────────────────────────

int64_t NpyArray::numel() const {
    if (shape.empty()) return 0;
    int64_t n = 1;
    for (auto s : shape) n *= s;
    return n;
}

// ── .npy format ─────────────────────────────────────────────────────────────
//
// NumPy .npy format (v1.0):
//   6 bytes: magic "\x93NUMPY"
//   1 byte:  major version (1)
//   1 byte:  minor version (0)
//   2 bytes: header length (little-endian uint16)
//   N bytes: ASCII header dict, padded with spaces + \n to 64-byte alignment
//   rest:    raw data
//
// Header dict example: "{'descr': '<f4', 'fortran_order': False, 'shape': (1, 257, 20, 2), }\n"

static std::vector<int64_t> parse_npy_shape(const std::string& header) {
    // Find 'shape': (...)
    auto pos = header.find("'shape'");
    if (pos == std::string::npos)
        throw std::runtime_error("npy: no 'shape' in header");

    auto paren_start = header.find('(', pos);
    auto paren_end = header.find(')', paren_start);
    if (paren_start == std::string::npos || paren_end == std::string::npos)
        throw std::runtime_error("npy: malformed shape");

    std::string shape_str = header.substr(paren_start + 1, paren_end - paren_start - 1);

    std::vector<int64_t> shape;
    size_t i = 0;
    while (i < shape_str.size()) {
        // Skip whitespace and commas
        while (i < shape_str.size() && (shape_str[i] == ' ' || shape_str[i] == ','))
            i++;
        if (i >= shape_str.size()) break;

        // Parse integer
        int64_t val = 0;
        while (i < shape_str.size() && shape_str[i] >= '0' && shape_str[i] <= '9') {
            val = val * 10 + (shape_str[i] - '0');
            i++;
        }
        shape.push_back(val);
    }
    return shape;
}

NpyArray npy_load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error("npy_load: cannot open " + path);

    // Read magic
    char magic[6];
    f.read(magic, 6);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0)
        throw std::runtime_error("npy_load: bad magic in " + path);

    // Read version
    uint8_t major, minor;
    f.read(reinterpret_cast<char*>(&major), 1);
    f.read(reinterpret_cast<char*>(&minor), 1);

    // Read header length
    uint32_t header_len = 0;
    if (major == 1) {
        uint16_t hl;
        f.read(reinterpret_cast<char*>(&hl), 2);
        header_len = hl;
    } else if (major == 2) {
        f.read(reinterpret_cast<char*>(&header_len), 4);
    } else {
        throw std::runtime_error("npy_load: unsupported version " +
                                 std::to_string(major) + "." + std::to_string(minor));
    }

    // Read header string
    std::string header(header_len, '\0');
    f.read(&header[0], header_len);

    // Verify dtype is float32 little-endian
    if (header.find("'<f4'") == std::string::npos &&
        header.find("'float32'") == std::string::npos) {
        throw std::runtime_error("npy_load: expected float32, got header: " + header);
    }

    // Verify C-contiguous (not Fortran order)
    if (header.find("True") != std::string::npos) {
        throw std::runtime_error("npy_load: Fortran order not supported");
    }

    // Parse shape
    auto shape = parse_npy_shape(header);

    int64_t numel = 1;
    for (auto s : shape) numel *= s;

    // Read data
    NpyArray arr;
    arr.shape = shape;
    arr.data.resize(numel);
    f.read(reinterpret_cast<char*>(arr.data.data()), numel * sizeof(float));

    if (!f)
        throw std::runtime_error("npy_load: short read in " + path);

    return arr;
}

void npy_save(const std::string& path, const float* data,
              const std::vector<int64_t>& shape) {
    // Build header dict
    std::string shape_str = "(";
    for (size_t i = 0; i < shape.size(); i++) {
        shape_str += std::to_string(shape[i]);
        if (i + 1 < shape.size()) shape_str += ", ";
        else if (shape.size() == 1) shape_str += ",";  // trailing comma for 1-d
    }
    shape_str += ")";

    std::string dict = "{'descr': '<f4', 'fortran_order': False, 'shape': " +
                       shape_str + ", }";

    // Pad header to 64-byte alignment (magic=6 + version=2 + header_len=2 + header)
    size_t preamble = 6 + 1 + 1 + 2;
    size_t total = preamble + dict.size() + 1;  // +1 for \n
    size_t padding = (64 - (total % 64)) % 64;
    dict += std::string(padding, ' ') + "\n";

    uint16_t header_len = (uint16_t)dict.size();

    int64_t numel = 1;
    for (auto s : shape) numel *= s;

    std::ofstream f(path, std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error("npy_save: cannot open " + path);

    // Write magic + version
    f.write("\x93NUMPY", 6);
    uint8_t major = 1, minor = 0;
    f.write(reinterpret_cast<char*>(&major), 1);
    f.write(reinterpret_cast<char*>(&minor), 1);
    f.write(reinterpret_cast<char*>(&header_len), 2);
    f.write(dict.data(), dict.size());
    f.write(reinterpret_cast<const char*>(data), numel * sizeof(float));
}

// ── Audio I/O ────────────────────────────────────────────────────────────────

#ifdef DEEPVQE_HAS_SNDFILE
#include <sndfile.h>

std::vector<float> audio_load_mono(const std::string& path, int target_sr) {
    SF_INFO info = {};
    SNDFILE* sf = sf_open(path.c_str(), SFM_READ, &info);
    if (!sf) {
        fprintf(stderr, "Failed to open audio: %s (%s)\n",
                path.c_str(), sf_strerror(nullptr));
        return {};
    }

    // Read all frames as float
    std::vector<float> raw(info.frames * info.channels);
    sf_count_t read = sf_readf_float(sf, raw.data(), info.frames);
    sf_close(sf);

    if (read != info.frames) {
        fprintf(stderr, "Short read: %s (%lld of %lld frames)\n",
                path.c_str(), (long long)read, (long long)info.frames);
    }

    // Mix to mono if needed
    std::vector<float> mono(read);
    if (info.channels == 1) {
        mono.assign(raw.begin(), raw.begin() + read);
    } else {
        for (sf_count_t i = 0; i < read; i++) {
            float sum = 0.0f;
            for (int c = 0; c < info.channels; c++)
                sum += raw[i * info.channels + c];
            mono[i] = sum / info.channels;
        }
    }

    // Simple integer-ratio resampling (decimation) if needed
    if (info.samplerate != target_sr) {
        if (info.samplerate % target_sr != 0) {
            fprintf(stderr, "Cannot resample %d -> %d (non-integer ratio)\n",
                    info.samplerate, target_sr);
            return {};
        }
        int ratio = info.samplerate / target_sr;
        std::vector<float> resampled(mono.size() / ratio);
        for (size_t i = 0; i < resampled.size(); i++)
            resampled[i] = mono[i * ratio];
        return resampled;
    }
    return mono;
}
#endif

// ── GGUF tensor loading ────────────────────────────────────────────────────

NpyArray load_tensor_from_ggml(struct ggml_context* ctx,
                               const std::string& name,
                               struct gguf_context* gctx,
                               bool verbose) {
    struct ggml_tensor* t = ggml_get_tensor(ctx, name.c_str());
    if (!t) {
        fprintf(stderr, "Missing tensor: %s\n", name.c_str());
        return {};
    }

    NpyArray arr;
    int nd = ggml_n_dims(t);
    for (int d = nd - 1; d >= 0; d--)
        arr.shape.push_back(t->ne[d]);

    int64_t n = ggml_nelements(t);
    arr.data.resize(n);

    if (t->type == GGML_TYPE_F32) {
        std::memcpy(arr.data.data(), t->data, n * sizeof(float));
    } else {
        const auto* traits = ggml_get_type_traits(t->type);
        if (traits && traits->to_float) {
            traits->to_float(t->data, arr.data.data(), n);
            if (verbose)
                printf("  Dequantized %s (%s -> F32, %lld elements)\n",
                       name.c_str(), ggml_type_name(t->type), (long long)n);
        } else {
            fprintf(stderr, "Unsupported tensor type for %s: %s\n",
                    name.c_str(), ggml_type_name(t->type));
            return {};
        }
    }

    // Quantized tensors may have been flattened to 1D for block-size
    // alignment.  Restore original shape from GGUF metadata if present.
    if (gctx) {
        char key[256];
        snprintf(key, sizeof(key), "deepvqe.shape.%s.ndim", name.c_str());
        int ndim_idx = gguf_find_key(gctx, key);
        if (ndim_idx >= 0) {
            int ndim = (int)gguf_get_val_u32(gctx, ndim_idx);
            arr.shape.clear();
            for (int d = 0; d < ndim; d++) {
                snprintf(key, sizeof(key), "deepvqe.shape.%s.%d",
                         name.c_str(), d);
                int d_idx = gguf_find_key(gctx, key);
                if (d_idx >= 0)
                    arr.shape.push_back((int64_t)gguf_get_val_u32(gctx, d_idx));
            }
        }
    }

    return arr;
}

// ── Comparison ──────────────────────────────────────────────────────────────

float max_abs_diff(const float* a, const float* b, int64_t n) {
    float max_err = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        float err = std::fabs(a[i] - b[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

float mean_abs_diff(const float* a, const float* b, int64_t n) {
    double sum = 0.0;
    for (int64_t i = 0; i < n; i++) {
        sum += std::fabs(a[i] - b[i]);
    }
    return (float)(sum / n);
}

bool print_result(const std::string& name, float max_err, float mean_err,
                  float ok_threshold, float fail_threshold) {
    const char* status;
    if (max_err < ok_threshold)
        status = "OK";
    else if (max_err < fail_threshold)
        status = "WARN";
    else
        status = "FAIL";

    printf("  [%s] %s: max=%.2e mean=%.2e\n", status, name.c_str(), max_err, mean_err);
    return max_err < fail_threshold;
}

// ── FFT / STFT / iSTFT (via KissFFT) ────────────────────────────────────────

#include "kiss_fftr.h"

// Verify kiss_fft_cpx layout matches our interleaved float format
static_assert(sizeof(kiss_fft_cpx) == 2 * sizeof(float),
              "kiss_fft_cpx must be packed {float r, i}");

std::vector<float> make_sqrt_hann(int n) {
    std::vector<float> w(n);
    for (int i = 0; i < n; i++) {
        float hann = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / n));
        w[i] = sqrtf(hann + 1e-12f);
    }
    return w;
}

void compute_stft(const float* signal, int N,
                  int n_fft, int hop, const float* window,
                  float* out, int n_freq, int n_frames,
                  stft_buffers* bufs) {
    int pad = n_fft / 2;

    // Use external or temporary resources
    kiss_fftr_cfg cfg = bufs && bufs->fwd_cfg
        ? (kiss_fftr_cfg)bufs->fwd_cfg
        : kiss_fftr_alloc(n_fft, 0, nullptr, nullptr);
    std::vector<float> scratch_tmp;
    std::vector<float> cpx_tmp;
    float* windowed;
    kiss_fft_cpx* cpx_out;
    if (bufs && bufs->scratch && bufs->cpx_buf) {
        windowed = bufs->scratch;
        cpx_out = reinterpret_cast<kiss_fft_cpx*>(bufs->cpx_buf);
    } else {
        scratch_tmp.resize(n_fft);
        cpx_tmp.resize(n_freq * 2);
        windowed = scratch_tmp.data();
        cpx_out = reinterpret_cast<kiss_fft_cpx*>(cpx_tmp.data());
    }

    for (int t = 0; t < n_frames; t++) {
        int center = t * hop;
        for (int i = 0; i < n_fft; i++) {
            int src = center - pad + i;
            windowed[i] = signal[reflect_idx(src, N)] * window[i];
        }
        kiss_fftr(cfg, windowed, cpx_out);
        for (int f = 0; f < n_freq; f++) {
            out[f * n_frames * 2 + t * 2 + 0] = cpx_out[f].r;
            out[f * n_frames * 2 + t * 2 + 1] = cpx_out[f].i;
        }
    }

    if (!bufs || !bufs->fwd_cfg) kiss_fftr_free(cfg);
}

void compute_istft(const float* stft_data, int n_freq, int n_frames,
                   int n_fft, int hop, const float* window,
                   float* signal, int N,
                   stft_buffers* bufs) {
    int pad = n_fft / 2;
    int padded_len = (n_frames - 1) * hop + n_fft;

    // Use external or temporary resources
    kiss_fftr_cfg icfg = bufs && bufs->inv_cfg
        ? (kiss_fftr_cfg)bufs->inv_cfg
        : kiss_fftr_alloc(n_fft, 1, nullptr, nullptr);

    std::vector<float> scratch_tmp, cpx_tmp, out_tmp, wsum_tmp;
    float* scratch;
    kiss_fft_cpx* cpx_in;
    float* output;
    float* window_sum;

    if (bufs && bufs->scratch && bufs->cpx_buf) {
        scratch = bufs->scratch;
        cpx_in = reinterpret_cast<kiss_fft_cpx*>(bufs->cpx_buf);
    } else {
        scratch_tmp.resize(n_fft);
        cpx_tmp.resize(n_freq * 2);
        scratch = scratch_tmp.data();
        cpx_in = reinterpret_cast<kiss_fft_cpx*>(cpx_tmp.data());
    }
    if (bufs && bufs->ola_out && bufs->ola_wsum) {
        output = bufs->ola_out;
        window_sum = bufs->ola_wsum;
    } else {
        out_tmp.resize(padded_len, 0.0f);
        wsum_tmp.resize(padded_len, 0.0f);
        output = out_tmp.data();
        window_sum = wsum_tmp.data();
    }
    std::fill_n(output, padded_len, 0.0f);
    std::fill_n(window_sum, padded_len, 0.0f);

    float inv_n = 1.0f / n_fft;
    for (int t = 0; t < n_frames; t++) {
        for (int f = 0; f < n_freq; f++) {
            cpx_in[f].r = stft_data[f * n_frames * 2 + t * 2 + 0];
            cpx_in[f].i = stft_data[f * n_frames * 2 + t * 2 + 1];
        }
        kiss_fftri(icfg, cpx_in, scratch);

        int start = t * hop;
        for (int i = 0; i < n_fft; i++) {
            float sample = scratch[i] * inv_n;
            output[start + i]     += sample * window[i];
            window_sum[start + i] += window[i] * window[i];
        }
    }

    if (!bufs || !bufs->inv_cfg) kiss_fftr_free(icfg);

    float threshold = 1e-11f;
    for (int i = 0; i < N; i++) {
        float ws = window_sum[i + pad];
        signal[i] = ws > threshold ? output[i + pad] / ws : 0.0f;
    }
}
