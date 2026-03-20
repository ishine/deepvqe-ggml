#ifndef DEEPVQE_API_H
#define DEEPVQE_API_H

/**
 * DeepVQE C API — purego-compatible shared library interface.
 *
 * All functions use simple C types (no structs, no variadic args)
 * for compatibility with Go's purego FFI.
 *
 * Typical usage:
 *   uintptr_t ctx = deepvqe_new("model.gguf");
 *   if (!ctx) { handle error }
 *   int ret = deepvqe_process_f32(ctx, mic, ref, n_samples, out);
 *   deepvqe_free(ctx);
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
  #ifdef DEEPVQE_BUILD
    #define DEEPVQE_API __declspec(dllexport)
  #else
    #define DEEPVQE_API __declspec(dllimport)
  #endif
#else
  #define DEEPVQE_API __attribute__((visibility("default")))
#endif

/**
 * Create a new DeepVQE context by loading a GGUF model file.
 * Returns an opaque handle, or 0 on failure.
 */
DEEPVQE_API uintptr_t deepvqe_new(const char* model_path);

/**
 * Free a DeepVQE context and all associated resources.
 */
DEEPVQE_API void deepvqe_free(uintptr_t ctx);

/**
 * Process audio through the AEC model (float32 version).
 *
 * mic:       Microphone input (mono, float32, [-1,1] range, 16kHz)
 * ref:       Far-end reference (mono, float32, [-1,1] range, 16kHz)
 * n_samples: Number of samples in both mic and ref (must be >= 512)
 * out:       Pre-allocated output buffer (n_samples floats)
 *
 * Returns 0 on success, negative on error.
 */
DEEPVQE_API int deepvqe_process_f32(uintptr_t ctx,
                                     const float* mic, const float* ref,
                                     int n_samples, float* out);

/**
 * Process audio through the AEC model (int16 PCM version).
 *
 * mic:       Microphone input (mono, int16 PCM, 16kHz)
 * ref:       Far-end reference (mono, int16 PCM, 16kHz)
 * n_samples: Number of samples in both mic and ref (must be >= 512)
 * out:       Pre-allocated output buffer (n_samples int16s)
 *
 * Returns 0 on success, negative on error.
 */
DEEPVQE_API int deepvqe_process_s16(uintptr_t ctx,
                                     const int16_t* mic, const int16_t* ref,
                                     int n_samples, int16_t* out);

/**
 * Get the last error message, or empty string if no error.
 * The returned pointer is valid until the next API call on this context.
 */
DEEPVQE_API const char* deepvqe_last_error(uintptr_t ctx);

/**
 * Get model sample rate (always 16000 currently).
 */
DEEPVQE_API int deepvqe_sample_rate(uintptr_t ctx);

/**
 * Get hop length in samples (256).
 */
DEEPVQE_API int deepvqe_hop_length(uintptr_t ctx);

/**
 * Get FFT size (512).
 */
DEEPVQE_API int deepvqe_fft_size(uintptr_t ctx);

#ifdef __cplusplus
}
#endif

#endif /* DEEPVQE_API_H */
