/**
 * DeepVQE AEC inference CLI.
 *
 * Build:
 *   cd ggml && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
 *
 * Usage:
 *   ./build/deepvqe model.gguf --input-npy mic.npy ref.npy [--dump-intermediates]
 */

#include "deepvqe_model.h"

#include <cstdio>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr,
            "Usage: deepvqe <model.gguf> --input-npy <mic.npy> <ref.npy> [--dump-intermediates] [--output-dir <dir>]\n");
        return 1;
    }

    const char* model_path = nullptr;
    const char* mic_path = nullptr;
    const char* ref_path = nullptr;
    std::string output_dir;
    bool dump = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--input-npy" && i + 2 < argc) { mic_path = argv[++i]; ref_path = argv[++i]; }
        else if (a == "--dump-intermediates") dump = true;
        else if (a == "--output-dir" && i + 1 < argc) output_dir = argv[++i];
        else if (a == "-h" || a == "--help") {
            fprintf(stderr, "Usage: deepvqe <model.gguf> --input-npy <mic.npy> <ref.npy> "
                    "[--dump-intermediates] [--output-dir <dir>]\n");
            return 0;
        }
        else if (!model_path) model_path = argv[i];
        else { fprintf(stderr, "Unknown arg: %s\n", argv[i]); return 1; }
    }

    if (!model_path || !mic_path || !ref_path) {
        fprintf(stderr, "Error: model path and --input-npy required\n");
        return 1;
    }

    if (dump && output_dir.empty()) output_dir = "intermediates/ggml";

    deepvqe_model model;
    if (!load_model(model_path, model)) return 1;

    NpyArray mic_stft = npy_load(mic_path);
    NpyArray ref_stft = npy_load(ref_path);
    printf("Input: mic=(%lld,%lld,%lld,%lld) ref=(%lld,%lld,%lld,%lld)\n",
           (long long)mic_stft.dim(0), (long long)mic_stft.dim(1),
           (long long)mic_stft.dim(2), (long long)mic_stft.dim(3),
           (long long)ref_stft.dim(0), (long long)ref_stft.dim(1),
           (long long)ref_stft.dim(2), (long long)ref_stft.dim(3));

    NpyArray enhanced = forward(mic_stft, ref_stft, model, dump ? output_dir : "");

    // Always save output
    std::string out_path = output_dir.empty() ? "enhanced.npy" : output_dir + "/output.npy";
    npy_save(out_path, enhanced);
    printf("Saved enhanced to: %s\n", out_path.c_str());

    return 0;
}
