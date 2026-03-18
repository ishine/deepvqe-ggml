/**
 * DeepVQE AEC inference using GGML.
 *
 * Full forward pass: FE -> Encoder -> AlignBlock -> Bottleneck -> Decoder -> CCM.
 * All ops are manual C++ (verified block-by-block against PyTorch).
 *
 * Build:
 *   cd ggml && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
 *
 * Usage:
 *   ./build/deepvqe model.gguf --input-npy mic.npy ref.npy [--dump-intermediates]
 */

#include "common.h"

#include "ggml.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

// ── Structs ────────────────────────────────────────────────────────────────

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

// ── Tensor helpers ─────────────────────────────────────────────────────────

// Flat buffer representing (C, T, F) — all block ops work on this layout.
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

static const NpyArray& W(const deepvqe_model& m, const std::string& name) {
    auto it = m.tensors.find(name);
    if (it == m.tensors.end()) {
        fprintf(stderr, "Missing tensor: %s\n", name.c_str());
        static NpyArray empty;
        return empty;
    }
    return it->second;
}

// ── Primitive ops ──────────────────────────────────────────────────────────

static inline float elu_f(float x) {
    return x > 0.0f ? x : std::exp(x) - 1.0f;
}

static inline float sigmoid_f(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

static void conv2d(
    const float* input, int C_in, int T_in, int F_in,
    const float* weight, const float* bias,
    int C_out, int kT, int kF, int sT, int sF,
    float* output, int T_out, int F_out
) {
    for (int co = 0; co < C_out; co++) {
        for (int t = 0; t < T_out; t++) {
            for (int f = 0; f < F_out; f++) {
                float sum = bias ? bias[co] : 0.0f;
                for (int ci = 0; ci < C_in; ci++) {
                    for (int kt = 0; kt < kT; kt++) {
                        for (int kf = 0; kf < kF; kf++) {
                            sum += weight[co * C_in * kT * kF + ci * kT * kF + kt * kF + kf] *
                                   input[ci * T_in * F_in + (t * sT + kt) * F_in + (f * sF + kf)];
                        }
                    }
                }
                output[co * T_out * F_out + t * F_out + f] = sum;
            }
        }
    }
}

static Buf zero_pad(const Buf& x, int pl, int pr, int pt, int pb) {
    Buf out(x.C, x.T + pt + pb, x.F + pl + pr);
    for (int c = 0; c < x.C; c++)
        for (int t = 0; t < x.T; t++)
            for (int f = 0; f < x.F; f++)
                out(c, t + pt, f + pl) = x(c, t, f);
    return out;
}

// Causal Conv2d with ZeroPad2d([1,1,3,0]) + kernel (4,3)
static Buf causal_conv(const Buf& x, const float* w, const float* b, int C_out, int sT, int sF) {
    Buf padded = zero_pad(x, 1, 1, 3, 0);
    int T_out = (padded.T - 4) / sT + 1;
    int F_out = (padded.F - 3) / sF + 1;
    Buf out(C_out, T_out, F_out);
    conv2d(padded.ptr(), x.C, padded.T, padded.F, w, b, C_out, 4, 3, sT, sF, out.ptr(), T_out, F_out);
    return out;
}

// ── Block ops (verified against PyTorch) ───────────────────────────────────

static void fe(const float* stft, float* out, int F, int T, float c) {
    const float eps = 1e-12f;
    for (int f = 0; f < F; f++) {
        for (int t = 0; t < T; t++) {
            int idx = f * T * 2 + t * 2;
            float re = stft[idx], im = stft[idx + 1];
            float mag = std::sqrt(re * re + im * im + eps);
            float s = std::pow(mag, c - 1.0f) / (1.0f + eps);
            out[0 * T * F + t * F + f] = re * s;
            out[1 * T * F + t * F + f] = im * s;
        }
    }
}

static Buf encoder_block(const Buf& x, const std::string& prefix, const deepvqe_model& m) {
    auto& cw = W(m, prefix + ".conv.weight");
    auto& cb = W(m, prefix + ".conv.bias");
    auto& rw = W(m, prefix + ".resblock.conv.weight");
    auto& rb = W(m, prefix + ".resblock.conv.bias");
    int C_out = (int)cw.dim(0);

    // Main: pad -> conv(stride 1,2) -> elu
    Buf y = causal_conv(x, cw.data.data(), cb.data.data(), C_out, 1, 2);
    for (auto& v : y.data) v = elu_f(v);

    // Residual: pad -> conv(stride 1,1) -> elu + skip
    Buf res = causal_conv(y, rw.data.data(), rb.data.data(), C_out, 1, 1);
    for (int64_t i = 0; i < res.numel(); i++)
        res.data[i] = elu_f(res.data[i]) + y.data[i];
    return res;
}

static Buf bottleneck(const Buf& x, const deepvqe_model& m) {
    int C = x.C, T = x.T, F = x.F;
    int input_size = C * F;

    auto& wih = W(m, "bottleneck.gru.weight_ih_l0");
    auto& whh = W(m, "bottleneck.gru.weight_hh_l0");
    auto& bih = W(m, "bottleneck.gru.bias_ih_l0");
    auto& bhh = W(m, "bottleneck.gru.bias_hh_l0");
    auto& fc_w = W(m, "bottleneck.fc.weight");
    auto& fc_b = W(m, "bottleneck.fc.bias");
    int hidden_size = (int)whh.dim(1);

    // Reshape: (C,T,F) -> (T, C*F)
    std::vector<float> flat(T * input_size);
    for (int t = 0; t < T; t++)
        for (int c = 0; c < C; c++)
            for (int f = 0; f < F; f++)
                flat[t * input_size + c * F + f] = x(c, t, f);

    // GRU forward
    std::vector<float> hidden(hidden_size, 0.0f);
    std::vector<float> gru_out(T * hidden_size);
    std::vector<float> gih(3 * hidden_size), ghh(3 * hidden_size);

    for (int t = 0; t < T; t++) {
        const float* xt = flat.data() + t * input_size;
        for (int i = 0; i < 3 * hidden_size; i++) {
            float s = bih.data[i];
            for (int j = 0; j < input_size; j++) s += wih.data[i * input_size + j] * xt[j];
            gih[i] = s;
        }
        for (int i = 0; i < 3 * hidden_size; i++) {
            float s = bhh.data[i];
            for (int j = 0; j < hidden_size; j++) s += whh.data[i * hidden_size + j] * hidden[j];
            ghh[i] = s;
        }
        for (int i = 0; i < hidden_size; i++) {
            float r = sigmoid_f(gih[i] + ghh[i]);
            float z = sigmoid_f(gih[hidden_size + i] + ghh[hidden_size + i]);
            float n = std::tanh(gih[2 * hidden_size + i] + r * ghh[2 * hidden_size + i]);
            hidden[i] = (1.0f - z) * n + z * hidden[i];
        }
        std::memcpy(gru_out.data() + t * hidden_size, hidden.data(), hidden_size * sizeof(float));
    }

    // Linear: (T, H) -> (T, C*F)
    int out_features = (int)fc_w.dim(0);
    std::vector<float> fc_out(T * out_features);
    for (int t = 0; t < T; t++)
        for (int o = 0; o < out_features; o++) {
            float s = fc_b.data[o];
            for (int j = 0; j < hidden_size; j++) s += fc_w.data[o * hidden_size + j] * gru_out[t * hidden_size + j];
            fc_out[t * out_features + o] = s;
        }

    // Reshape back: (T, C*F) -> (C,T,F)
    Buf out(C, T, F);
    for (int t = 0; t < T; t++)
        for (int c = 0; c < C; c++)
            for (int f = 0; f < F; f++)
                out(c, t, f) = fc_out[t * out_features + c * F + f];
    return out;
}

static Buf align_block(const Buf& x_mic, const Buf& x_ref, const deepvqe_model& m) {
    auto& hp = m.hparams;
    int C = x_mic.C, T = x_mic.T, F = x_mic.F;
    int H = hp.align_hidden, dmax = hp.dmax;
    float temperature = 1.0f;

    auto& pmw = W(m, "align.pconv_mic.weight");
    auto& pmb = W(m, "align.pconv_mic.bias");
    auto& prw = W(m, "align.pconv_ref.weight");
    auto& prb = W(m, "align.pconv_ref.bias");
    auto& sw = W(m, "align.conv.1.weight");
    auto& sb = W(m, "align.conv.1.bias");

    // Q = pconv_mic(x_mic), K = pconv_ref(x_ref) — 1x1 convs
    Buf Q(H, T, F), K(H, T, F);
    for (int h = 0; h < H; h++)
        for (int t = 0; t < T; t++)
            for (int f = 0; f < F; f++) {
                float sq = pmb.data[h], sk = prb.data[h];
                for (int ci = 0; ci < C; ci++) {
                    sq += pmw.data[h * C + ci] * x_mic(ci, t, f);
                    sk += prw.data[h * C + ci] * x_ref(ci, t, f);
                }
                Q(h, t, f) = sq;
                K(h, t, f) = sk;
            }

    // Pad K: (H, T, F) -> (H, T+dmax-1, F)
    int Tp = T + dmax - 1;
    Buf Kp(H, Tp, F);
    for (int h = 0; h < H; h++)
        for (int t = 0; t < T; t++)
            for (int f = 0; f < F; f++)
                Kp(h, t + dmax - 1, f) = K(h, t, f);

    // Similarity: V[h,t,d] = sum_f Q[h,t,f] * Kp[h,t+d,f] / sqrt(F)
    float scale = 1.0f / std::sqrt((float)F);
    std::vector<float> V(H * T * dmax, 0.0f);
    for (int h = 0; h < H; h++)
        for (int t = 0; t < T; t++)
            for (int d = 0; d < dmax; d++) {
                float s = 0.0f;
                for (int f = 0; f < F; f++)
                    s += Q(h, t, f) * Kp(h, t + d, f);
                V[(h * T + t) * dmax + d] = s * scale;
            }

    // Smooth: pad(1,1,4,0) + Conv2d(H,1,(5,3)) -> (1,T,dmax)
    int Tv = T + 4, Dv = dmax + 2;
    std::vector<float> Vp(H * Tv * Dv, 0.0f);
    for (int h = 0; h < H; h++)
        for (int t = 0; t < T; t++)
            for (int d = 0; d < dmax; d++)
                Vp[h * Tv * Dv + (t + 4) * Dv + (d + 1)] = V[(h * T + t) * dmax + d];

    std::vector<float> Vc(T * dmax);
    conv2d(Vp.data(), H, Tv, Dv, sw.data.data(), sb.data.data(), 1, 5, 3, 1, 1, Vc.data(), T, dmax);

    // Softmax over delay
    for (auto& v : Vc) v /= temperature;
    for (int t = 0; t < T; t++) {
        float* row = Vc.data() + t * dmax;
        float mx = row[0];
        for (int d = 1; d < dmax; d++) if (row[d] > mx) mx = row[d];
        float sm = 0.0f;
        for (int d = 0; d < dmax; d++) { row[d] = std::exp(row[d] - mx); sm += row[d]; }
        for (int d = 0; d < dmax; d++) row[d] /= sm;
    }

    // Pad x_ref and weighted sum
    Buf rp(C, Tp, F);
    for (int c = 0; c < C; c++)
        for (int t = 0; t < T; t++)
            for (int f = 0; f < F; f++)
                rp(c, t + dmax - 1, f) = x_ref(c, t, f);

    Buf aligned(C, T, F);
    for (int c = 0; c < C; c++)
        for (int t = 0; t < T; t++)
            for (int d = 0; d < dmax; d++) {
                float a = Vc[t * dmax + d];
                for (int f = 0; f < F; f++)
                    aligned(c, t, f) += a * rp(c, t + d, f);
            }
    return aligned;
}

static Buf decoder_block(const Buf& x, const Buf& x_en,
                         const std::string& prefix, const deepvqe_model& m, bool is_last) {
    int C = x.C, T = x.T, F = x.F;

    auto& skw = W(m, prefix + ".skip_conv.weight");
    auto& skb = W(m, prefix + ".skip_conv.bias");
    auto& rw = W(m, prefix + ".resblock.conv.weight");
    auto& rb = W(m, prefix + ".resblock.conv.bias");
    auto& dw = W(m, prefix + ".deconv.conv.weight");
    auto& db = W(m, prefix + ".deconv.conv.bias");
    int C_out = (int)dw.dim(0) / 2;

    // y = x + skip_conv(x_en) — 1x1 conv
    Buf y(C, T, F);
    for (int co = 0; co < C; co++)
        for (int t = 0; t < T; t++)
            for (int f = 0; f < F; f++) {
                float s = skb.data[co];
                for (int ci = 0; ci < C; ci++)
                    s += skw.data[co * C + ci] * x_en(ci, t, f);
                y(co, t, f) = x(co, t, f) + s;
            }

    // ResidualBlock: pad -> conv -> elu + skip
    Buf res = causal_conv(y, rw.data.data(), rb.data.data(), C, 1, 1);
    for (int64_t i = 0; i < res.numel(); i++)
        res.data[i] = elu_f(res.data[i]) + y.data[i];

    // SubpixelConv2d: pad -> conv(C_out*2) -> pixel shuffle
    Buf padded = zero_pad(res, 1, 1, 3, 0);
    int Tc = padded.T - 4 + 1;
    int Fc = padded.F - 3 + 1;
    int C_conv = C_out * 2;
    Buf conv_out(C_conv, Tc, Fc);
    conv2d(padded.ptr(), C, padded.T, padded.F,
           dw.data.data(), db.data.data(), C_conv, 4, 3, 1, 1,
           conv_out.ptr(), Tc, Fc);

    // Pixel shuffle: rearrange("(r c) t f -> c t (r f)", r=2)
    int F_out = 2 * Fc;
    Buf deconv(C_out, Tc, F_out);
    for (int c = 0; c < C_out; c++)
        for (int t = 0; t < Tc; t++)
            for (int r = 0; r < 2; r++)
                for (int f = 0; f < Fc; f++)
                    deconv(c, t, r * Fc + f) = conv_out(r * C_out + c, t, f);

    // ChannelAffine + ELU (if not is_last)
    if (!is_last) {
        auto& bns = W(m, prefix + ".bn.scale");
        auto& bnb = W(m, prefix + ".bn.bias");
        for (int c = 0; c < C_out; c++)
            for (int t = 0; t < Tc; t++)
                for (int f = 0; f < F_out; f++)
                    deconv(c, t, f) = elu_f(deconv(c, t, f) * bns.data[c] + bnb.data[c]);
    }
    return deconv;
}

// Trim frequency dimension to target
static Buf freq_trim(const Buf& x, int target_F) {
    if (x.F <= target_F) return x;
    Buf out(x.C, x.T, target_F);
    for (int c = 0; c < x.C; c++)
        for (int t = 0; t < x.T; t++)
            std::memcpy(&out(c, t, 0), &x(c, t, 0), target_F * sizeof(float));
    return out;
}

static Buf concat_channels(const Buf& a, const Buf& b) {
    Buf out(a.C + b.C, a.T, a.F);
    for (int c = 0; c < a.C; c++)
        std::memcpy(out.data.data() + c * a.T * a.F, a.data.data() + c * a.T * a.F, a.T * a.F * sizeof(float));
    for (int c = 0; c < b.C; c++)
        std::memcpy(out.data.data() + (a.C + c) * a.T * a.F, b.data.data() + c * b.T * b.F, b.T * b.F * sizeof(float));
    return out;
}

static NpyArray ccm(const Buf& mask, const float* stft, int F_stft, int T, const deepvqe_model& /*m*/) {
    int F = mask.F;
    static const float VR[3] = {1.0f, -0.5f, -0.5f};
    static const float VI[3] = {0.0f, 0.86602540378f, -0.86602540378f};

    // H_real, H_imag: (9, T, F)
    std::vector<float> Hr(9 * T * F, 0.0f), Hi(9 * T * F, 0.0f);
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 9; c++) {
            int ch = r * 9 + c;
            for (int t = 0; t < T; t++)
                for (int f = 0; f < F; f++) {
                    float val = mask(ch, t, f);
                    Hr[c * T * F + t * F + f] += VR[r] * val;
                    Hi[c * T * F + t * F + f] += VI[r] * val;
                }
        }

    // Pad input: stft (F,T,2) -> x_perm (2,T,F) -> padded (2, T+2, F+2)
    int Tp = T + 2, Fp = F + 2;
    std::vector<float> xp(2 * Tp * Fp, 0.0f);
    for (int c = 0; c < 2; c++)
        for (int t = 0; t < T; t++)
            for (int f = 0; f < F; f++)
                xp[c * Tp * Fp + (t + 2) * Fp + (f + 1)] = stft[f * T * 2 + t * 2 + c];

    // Complex multiply and sum
    std::vector<float> er(T * F, 0.0f), ei(T * F, 0.0f);
    for (int mn = 0; mn < 3; mn++)
        for (int nn = 0; nn < 3; nn++) {
            int ki = mn * 3 + nn;
            for (int t = 0; t < T; t++)
                for (int f = 0; f < F; f++) {
                    int tf = t * F + f;
                    float mr = Hr[ki * T * F + tf], mi = Hi[ki * T * F + tf];
                    float xr = xp[0 * Tp * Fp + (t + mn) * Fp + (f + nn)];
                    float xi = xp[1 * Tp * Fp + (t + mn) * Fp + (f + nn)];
                    er[tf] += mr * xr - mi * xi;
                    ei[tf] += mr * xi + mi * xr;
                }
        }

    // Output: (1, F, T, 2)
    NpyArray out;
    out.shape = {1, (int64_t)F, (int64_t)T, 2};
    out.data.resize(F * T * 2);
    for (int f = 0; f < F; f++)
        for (int t = 0; t < T; t++) {
            out.data[f * T * 2 + t * 2 + 0] = er[t * F + f];
            out.data[f * T * 2 + t * 2 + 1] = ei[t * F + f];
        }
    return out;
}

// ── Full forward pass ──────────────────────────────────────────────────────

static NpyArray forward(const NpyArray& mic_stft, const NpyArray& ref_stft,
                        const deepvqe_model& m, const std::string& dump_dir) {
    auto& hp = m.hparams;
    int F = (int)mic_stft.dim(1);
    int T = (int)mic_stft.dim(2);
    bool dump = !dump_dir.empty();

    auto save = [&](const std::string& name, const Buf& b) {
        if (!dump) return;
        NpyArray a;
        a.shape = {1, (int64_t)b.C, (int64_t)b.T, (int64_t)b.F};
        a.data = b.data;
        npy_save(dump_dir + "/" + name + ".npy", a);
    };

    printf("Forward: F=%d T=%d\n", F, T);

    // 1. Feature extraction
    Buf mic_fe(2, T, F), ref_fe(2, T, F);
    fe(mic_stft.data.data(), mic_fe.ptr(), F, T, hp.power_law_c);
    fe(ref_stft.data.data(), ref_fe.ptr(), F, T, hp.power_law_c);
    save("fe_mic", mic_fe);
    save("fe_ref", ref_fe);

    // 2. Mic encoder 1-2
    Buf mic_e1 = encoder_block(mic_fe, "mic_enc1", m);
    Buf mic_e2 = encoder_block(mic_e1, "mic_enc2", m);
    save("mic_enc1", mic_e1);
    save("mic_enc2", mic_e2);

    // 3. Far-end encoder 1-2
    Buf far_e1 = encoder_block(ref_fe, "far_enc1", m);
    Buf far_e2 = encoder_block(far_e1, "far_enc2", m);
    save("far_enc1", far_e1);
    save("far_enc2", far_e2);

    // 4. Alignment
    Buf aligned = align_block(mic_e2, far_e2, m);
    save("align", aligned);

    // 5. Concat + encoder 3-5
    Buf cat = concat_channels(mic_e2, aligned);
    Buf mic_e3 = encoder_block(cat, "mic_enc3", m);
    Buf mic_e4 = encoder_block(mic_e3, "mic_enc4", m);
    Buf mic_e5 = encoder_block(mic_e4, "mic_enc5", m);
    save("mic_enc3", mic_e3);
    save("mic_enc4", mic_e4);
    save("mic_enc5", mic_e5);

    // 6. Bottleneck
    Buf bn = bottleneck(mic_e5, m);
    save("bottleneck", bn);

    // 7. Decoder with skip connections + frequency trimming
    Buf d5_raw = decoder_block(bn, mic_e5, "dec5", m, false);
    save("dec5", d5_raw);
    Buf d5 = freq_trim(d5_raw, mic_e4.F);

    Buf d4_raw = decoder_block(d5, mic_e4, "dec4", m, false);
    save("dec4", d4_raw);
    Buf d4 = freq_trim(d4_raw, mic_e3.F);

    Buf d3_raw = decoder_block(d4, mic_e3, "dec3", m, false);
    save("dec3", d3_raw);
    Buf d3 = freq_trim(d3_raw, mic_e2.F);

    Buf d2_raw = decoder_block(d3, mic_e2, "dec2", m, false);
    save("dec2", d2_raw);
    Buf d2 = freq_trim(d2_raw, mic_e1.F);

    Buf d1_raw = decoder_block(d2, mic_e1, "dec1", m, true);
    save("dec1", d1_raw);
    Buf d1 = freq_trim(d1_raw, mic_fe.F);

    printf("Decoder output: C=%d T=%d F=%d\n", d1.C, d1.T, d1.F);

    // 8. CCM
    NpyArray enhanced = ccm(d1, mic_stft.data.data(), F, T, m);
    if (dump)
        npy_save(dump_dir + "/output.npy", enhanced);

    printf("Enhanced: (%lld, %lld, %lld, %lld)\n",
           (long long)enhanced.dim(0), (long long)enhanced.dim(1),
           (long long)enhanced.dim(2), (long long)enhanced.dim(3));
    return enhanced;
}

// ── GGUF loading ───────────────────────────────────────────────────────────

static uint32_t gguf_u32(struct gguf_context* ctx, const char* key) {
    int idx = gguf_find_key(ctx, key);
    return idx >= 0 ? gguf_get_val_u32(ctx, idx) : 0;
}

static bool load_model(const char* path, deepvqe_model& model) {
    struct ggml_context* ggml_ctx = nullptr;
    struct gguf_init_params params;
    params.no_alloc = false;
    params.ctx = &ggml_ctx;

    struct gguf_context* gctx = gguf_init_from_file(path, params);
    if (!gctx) { fprintf(stderr, "Failed to load: %s\n", path); return false; }

    auto& hp = model.hparams;
    hp.n_fft        = (int)gguf_u32(gctx, "deepvqe.n_fft");
    hp.hop_length   = (int)gguf_u32(gctx, "deepvqe.hop_length");
    hp.n_freq_bins  = (int)gguf_u32(gctx, "deepvqe.n_freq_bins");
    hp.sample_rate  = (int)gguf_u32(gctx, "deepvqe.sample_rate");
    hp.dmax         = (int)gguf_u32(gctx, "deepvqe.dmax");
    hp.align_hidden = (int)gguf_u32(gctx, "deepvqe.align_hidden");
    int idx = gguf_find_key(gctx, "deepvqe.power_law_c");
    hp.power_law_c = idx >= 0 ? gguf_get_val_f32(gctx, idx) : 0.3f;
    idx = gguf_find_key(gctx, "deepvqe.bn_folded");
    hp.bn_folded = idx >= 0 ? gguf_get_val_bool(gctx, idx) : true;

    int mic_n = (int)gguf_u32(gctx, "deepvqe.mic_channels.count");
    hp.mic_channels.resize(mic_n);
    for (int i = 0; i < mic_n; i++) {
        char k[64]; snprintf(k, sizeof(k), "deepvqe.mic_channels.%d", i);
        hp.mic_channels[i] = (int)gguf_u32(gctx, k);
    }
    int far_n = (int)gguf_u32(gctx, "deepvqe.far_channels.count");
    hp.far_channels.resize(far_n);
    for (int i = 0; i < far_n; i++) {
        char k[64]; snprintf(k, sizeof(k), "deepvqe.far_channels.%d", i);
        hp.far_channels[i] = (int)gguf_u32(gctx, k);
    }

    printf("Config: n_fft=%d hop=%d dmax=%d c=%.2f\n", hp.n_fft, hp.hop_length, hp.dmax, hp.power_law_c);

    int n_tensors = gguf_get_n_tensors(gctx);
    for (int i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(gctx, i);
        struct ggml_tensor* t = ggml_get_tensor(ggml_ctx, name);
        if (!t) continue;
        NpyArray arr;
        int nd = ggml_n_dims(t);
        for (int d = nd - 1; d >= 0; d--) arr.shape.push_back(t->ne[d]);
        arr.data.resize(arr.numel());
        std::memcpy(arr.data.data(), t->data, arr.numel() * sizeof(float));
        model.tensors[name] = std::move(arr);
    }
    printf("Loaded %zu tensors\n", model.tensors.size());

    gguf_free(gctx);
    if (ggml_ctx) ggml_free(ggml_ctx);
    return true;
}

// ── Main ───────────────────────────────────────────────────────────────────

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
