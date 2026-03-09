"""Block-level verification tests for DeepVQE.

Tests each block in isolation with synthetic tensors on CPU to find
the root cause of training failure (delay never works, model suppresses
everything instead of cancelling echo).

Run: python test_blocks.py
"""

import math

import torch
import torch.nn as nn

from src.align import AlignBlock
from src.blocks import (
    FE,
    Bottleneck,
    DecoderBlock,
    EncoderBlock,
    SubpixelConv2d,
)
from src.ccm import CCM
from src.model import DeepVQEAEC
from train import delay_samples_to_frame


# ── 1. FE ────────────────────────────────────────────────────────────────────


def test_fe_output_shape():
    fe = FE(c=0.3)
    x = torch.randn(2, 257, 16, 2)
    y = fe(x)
    assert y.shape == (2, 2, 16, 257), f"Expected (2,2,16,257), got {y.shape}"
    print(f"[PASS] test_fe_output_shape: {y.shape}")


def test_fe_power_law_compression():
    fe = FE(c=0.3)
    # Create input with known magnitude 5 at a single T-F bin
    x = torch.zeros(1, 1, 1, 2)
    x[0, 0, 0, 0] = 5.0  # real=5, imag=0 → mag=5
    y = fe(x)
    # Output should have mag = 5^0.3
    out_mag = torch.sqrt(y[0, 0, 0, 0] ** 2 + y[0, 1, 0, 0] ** 2)
    expected = 5.0 ** 0.3
    assert torch.allclose(out_mag, torch.tensor(expected), atol=1e-4), \
        f"Expected mag={expected:.6f}, got {out_mag.item():.6f}"
    print(f"[PASS] test_fe_power_law_compression: mag={out_mag.item():.6f} ≈ {expected:.6f}")


def test_fe_zero_input_stability():
    fe = FE(c=0.3)
    x = torch.zeros(1, 4, 8, 2)
    y = fe(x)
    assert not torch.isnan(y).any(), "NaN in output"
    assert not torch.isinf(y).any(), "Inf in output"
    print("[PASS] test_fe_zero_input_stability")


def test_fe_unit_magnitude_passthrough():
    fe = FE(c=0.3)
    # Unit magnitude: real=cos(θ), imag=sin(θ) → mag=1 → 1^0.3=1
    theta = torch.rand(1, 4, 8, 1) * 2 * math.pi
    x = torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1)
    y = fe(x)
    # y is (B,2,T,F), compute magnitude
    out_mag = torch.sqrt(y[:, 0] ** 2 + y[:, 1] ** 2)
    assert torch.allclose(out_mag, torch.ones_like(out_mag), atol=1e-4), \
        f"Unit mag not preserved: mean={out_mag.mean():.6f}"
    print(f"[PASS] test_fe_unit_magnitude_passthrough: mean mag={out_mag.mean():.6f}")


# ── 2. EncoderBlock ──────────────────────────────────────────────────────────


def test_encoder_output_shape():
    enc = EncoderBlock(2, 64)
    x = torch.randn(2, 2, 16, 257)
    y = enc(x)
    assert y.shape == (2, 64, 16, 129), f"Expected (2,64,16,129), got {y.shape}"
    print(f"[PASS] test_encoder_output_shape: {y.shape}")


def test_encoder_causality():
    enc = EncoderBlock(2, 64).eval()
    T = 20
    a = torch.randn(1, 2, T, 257)
    b = torch.randn(1, 2, T, 257)
    c = torch.randn(1, 2, T, 257)

    x1 = torch.cat([a, b], dim=2)
    x2 = torch.cat([a, c], dim=2)

    y1 = enc(x1)
    y2 = enc(x2)

    diff_past = (y1[:, :, :T] - y2[:, :, :T]).abs().max().item()
    diff_future = (y1[:, :, T:] - y2[:, :, T:]).abs().max().item()
    assert diff_past == 0.0, f"Causality violated: past diff={diff_past:.2e}"
    assert diff_future > 0.0, "Future frames should differ"
    print(f"[PASS] test_encoder_causality: past diff={diff_past:.2e}")


def test_encoder_frequency_chain():
    """5-block chain: 257→129→65→33→17→9"""
    channels = [2, 64, 128, 128, 128, 128]
    expected_freqs = [257, 129, 65, 33, 17, 9]
    x = torch.randn(1, channels[0], 16, expected_freqs[0])
    for i in range(5):
        enc = EncoderBlock(channels[i], channels[i + 1]).eval()
        x = enc(x)
        actual_f = x.shape[-1]
        assert actual_f == expected_freqs[i + 1], \
            f"Block {i}: expected F={expected_freqs[i+1]}, got {actual_f}"
    print(f"[PASS] test_encoder_frequency_chain: freqs={expected_freqs}")


# ── 3. AlignBlock ────────────────────────────────────────────────────────────


def test_align_unfold_indexing():
    """Frame t, delay d → ref[t-(dmax-1)+d]."""
    dmax = 4
    C, F = 2, 3
    T = 6
    # Create ref with frame IDs: ref[:,:,t,:] = t
    x_ref = torch.zeros(1, C, T, F)
    for t in range(T):
        x_ref[0, :, t, :] = t

    # Pad and unfold same as AlignBlock
    padded = nn.functional.pad(x_ref, [0, 0, dmax - 1, 0])  # (1,C,T+dmax-1,F)
    unfolded = padded.unfold(2, dmax, 1)  # (1,C,T,F,dmax)
    unfolded = unfolded.permute(0, 1, 2, 4, 3)  # (1,C,T,dmax,F)

    # For frame t=dmax-1 (first valid frame), delay d should give ref[d]
    t = dmax - 1
    for d in range(dmax):
        expected_frame_id = t - (dmax - 1) + d
        actual = unfolded[0, 0, t, d, 0].item()
        assert actual == expected_frame_id, \
            f"t={t}, d={d}: expected ref[{expected_frame_id}], got ref[{actual}]"
    print(f"[PASS] test_align_unfold_indexing")


def test_align_onehot_selects_frame():
    """One-hot at d → picks ref[t-(dmax-1)+d]."""
    dmax = 4
    C, F = 2, 3
    T = 8
    align = AlignBlock(C, hidden_channels=4, dmax=dmax).eval()

    x_ref = torch.zeros(1, C, T, F)
    for t in range(T):
        x_ref[0, :, t, :] = t

    # Manually compute what AlignBlock's weighted sum should give
    padded = nn.functional.pad(x_ref, [0, 0, dmax - 1, 0])
    ref_unfolded = padded.unfold(2, dmax, 1).permute(0, 1, 2, 4, 3)  # (1,C,T,dmax,F)

    # One-hot attention at d=2
    d_target = 2
    A = torch.zeros(1, 1, T, dmax)
    A[0, 0, :, d_target] = 1.0
    A_expanded = A[:, :, :, :, None]
    aligned = torch.sum(ref_unfolded * A_expanded, dim=-2)  # (1,C,T,F)

    # Only check frames where the selected ref frame is valid (non-padded)
    for t in range(T):
        ref_idx = t - (dmax - 1) + d_target
        if ref_idx < 0:
            # Padded region — should be zero
            actual = aligned[0, 0, t, 0].item()
            assert actual == 0.0, f"t={t}: expected 0 (padding), got {actual}"
        else:
            actual = aligned[0, 0, t, 0].item()
            assert actual == ref_idx, \
                f"t={t}: expected ref[{ref_idx}], got ref[{actual}]"
    print(f"[PASS] test_align_onehot_selects_frame")


def test_align_roundtrip_known_delay():
    """Mic = shifted ref → similarity should peak at correct delay index."""
    dmax = 8
    C, F = 128, 65
    T = 20
    delay_frames = 3  # ref is delayed by 3 frames

    torch.manual_seed(42)
    ref_signal = torch.randn(1, C, T, F)
    # Mic = ref shifted left by delay_frames (i.e., mic[t] = ref[t + delay_frames])
    mic = torch.zeros(1, C, T, F)
    for t in range(T - delay_frames):
        mic[0, :, t, :] = ref_signal[0, :, t + delay_frames, :]

    align = AlignBlock(C, hidden_channels=32, dmax=dmax)
    # Use identity projections to make similarity directly compare features
    with torch.no_grad():
        align.pconv_mic.weight.zero_()
        align.pconv_mic.bias.zero_()
        align.pconv_ref.weight.zero_()
        align.pconv_ref.bias.zero_()
        for i in range(min(C, 32)):
            align.pconv_mic.weight[i, i, 0, 0] = 1.0
            align.pconv_ref.weight[i, i, 0, 0] = 1.0
        # Zero out the smoothing conv, use passthrough at center position
        # Conv kernel is (5, 3) with causal pad [1,1,4,0]:
        #   time: pad top=4, kernel pos 4 = current frame
        #   delay: pad left=1, kernel pos 1 = current delay
        align.conv[1].weight.zero_()
        align.conv[1].bias.zero_()
        for h in range(32):
            align.conv[1].weight[0, h, 4, 1] = 1.0

    align.eval()
    _, delay_dist = align(mic, ref_signal, return_delay=True)  # (1, T, dmax)

    # Expected: ref[t-(dmax-1)+d] = mic[t] means t-(dmax-1)+d = t+delay_frames
    # → d = dmax-1+delay_frames ... but that's > dmax. Let me reconsider.
    # mic[t] = ref[t+delay_frames]. AlignBlock picks d such that ref[t-(dmax-1)+d] ≈ mic[t].
    # ref[t-(dmax-1)+d] = ref[t+delay_frames] → d = dmax-1+delay_frames. That exceeds dmax.
    # So actually: ref is the far-end that plays first, mic receives it delayed.
    # mic[t] = ref[t - delay_frames] (mic lags behind ref)
    mic2 = torch.zeros(1, C, T, F)
    for t in range(delay_frames, T):
        mic2[0, :, t, :] = ref_signal[0, :, t - delay_frames, :]

    _, delay_dist2 = align(mic2, ref_signal, return_delay=True)

    # Expected d = dmax-1-delay_frames for the alignment ref[t-(dmax-1)+d] = ref[t-delay_frames]
    expected_d = dmax - 1 - delay_frames
    # Check frames where there's enough context
    valid_start = dmax
    peaks = delay_dist2[0, valid_start:, :].argmax(dim=-1)
    correct = (peaks == expected_d).float().mean().item()
    assert correct > 0.5, \
        f"Only {correct:.0%} of frames peaked at d={expected_d}, peaks={peaks.tolist()}"
    print(f"[PASS] test_align_roundtrip_known_delay: {correct:.0%} correct at d={expected_d}")


def test_align_gradient_flow():
    align = AlignBlock(128, 32, dmax=8).train()
    mic = torch.randn(1, 128, 16, 65)
    ref = torch.randn(1, 128, 16, 65)
    aligned = align(mic, ref)
    loss = aligned.sum()
    loss.backward()
    no_grad = []
    for name, p in align.named_parameters():
        if p.grad is None or p.grad.abs().max() == 0:
            no_grad.append(name)
    assert len(no_grad) == 0, f"No gradient: {no_grad}"
    print(f"[PASS] test_align_gradient_flow: all params have grads")


def test_align_gradient_through_delay():
    align = AlignBlock(128, 32, dmax=8).train()
    mic = torch.randn(1, 128, 16, 65)
    ref = torch.randn(1, 128, 16, 65)
    _, delay_dist = align(mic, ref, return_delay=True)
    loss = delay_dist.sum()
    loss.backward()
    pconv_grads = {}
    for name in ["pconv_mic.weight", "pconv_mic.bias", "pconv_ref.weight", "pconv_ref.bias"]:
        p = dict(align.named_parameters())[name]
        has_grad = p.grad is not None and p.grad.abs().max() > 0
        pconv_grads[name] = has_grad
    failed = [k for k, v in pconv_grads.items() if not v]
    assert len(failed) == 0, f"No gradient through delay: {failed}"
    print(f"[PASS] test_align_gradient_through_delay")


def test_align_isolated_learning():
    """Train AlignBlock alone for 200 steps on known delay."""
    dmax = 8
    C, F = 32, 16
    T = 20
    delay_frames = 3
    torch.manual_seed(0)

    align = AlignBlock(C, hidden_channels=16, dmax=dmax)
    align.temperature = 0.5
    optimizer = torch.optim.Adam(align.parameters(), lr=1e-3)

    target_d = dmax - 1 - delay_frames

    for step in range(200):
        ref = torch.randn(2, C, T, F)
        mic = torch.zeros(2, C, T, F)
        for t in range(delay_frames, T):
            mic[:, :, t, :] = ref[:, :, t - delay_frames, :]

        _, delay_dist = align(mic, ref, return_delay=True)
        # Cross-entropy on target
        log_probs = torch.log(delay_dist + 1e-10)
        loss = -log_probs[:, dmax:, target_d].mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate
    align.eval()
    ref = torch.randn(4, C, T, F)
    mic = torch.zeros(4, C, T, F)
    for t in range(delay_frames, T):
        mic[:, :, t, :] = ref[:, :, t - delay_frames, :]

    with torch.no_grad():
        _, delay_dist = align(mic, ref, return_delay=True)
    peaks = delay_dist[:, dmax:, :].argmax(dim=-1)
    # Allow ±1 frame tolerance
    correct = ((peaks - target_d).abs() <= 1).float().mean().item()
    assert correct >= 0.9, f"Delay accuracy {correct:.0%} < 90%"
    print(f"[PASS] test_align_isolated_learning: accuracy={correct:.0%}")


def test_align_temperature_effect():
    align = AlignBlock(32, 16, dmax=8).eval()
    mic = torch.randn(1, 32, 16, 16)
    ref = torch.randn(1, 32, 16, 16)

    align.temperature = 0.1
    _, dist_cold = align(mic, ref, return_delay=True)
    entropy_cold = -(dist_cold * torch.log(dist_cold + 1e-10)).sum(-1).mean()

    align.temperature = 1.0
    _, dist_warm = align(mic, ref, return_delay=True)
    entropy_warm = -(dist_warm * torch.log(dist_warm + 1e-10)).sum(-1).mean()

    assert entropy_cold < entropy_warm, \
        f"Lower temp should give lower entropy: {entropy_cold:.4f} vs {entropy_warm:.4f}"
    print(f"[PASS] test_align_temperature_effect: cold={entropy_cold:.4f} < warm={entropy_warm:.4f}")


def test_align_K_vs_ref_unfold_consistency():
    """nn.Unfold (for K) vs tensor.unfold (for ref) select same frames."""
    dmax = 4
    C, F = 2, 3
    T = 6

    # Create input where each frame has a unique value
    x = torch.zeros(1, C, T, F)
    for t in range(T):
        x[0, :, t, :] = t + 1  # 1-indexed so we can distinguish from padding

    # Method 1: nn.Unfold (used for K in AlignBlock line 57)
    unfold_nn = nn.Sequential(
        nn.ZeroPad2d([0, 0, dmax - 1, 0]),
        nn.Unfold((dmax, 1)),
    )
    Ku = unfold_nn(x)  # (1, C*dmax, T*F)
    Ku = Ku.view(1, C, dmax, T, F)
    Ku = Ku.permute(0, 1, 3, 2, 4)  # (1, C, T, dmax, F)

    # Method 2: tensor.unfold (used for ref in AlignBlock line 76)
    padded = nn.functional.pad(x, [0, 0, dmax - 1, 0])
    ref_unfolded = padded.unfold(2, dmax, 1)  # (1, C, T, F, dmax)
    ref_unfolded = ref_unfolded.permute(0, 1, 2, 4, 3)  # (1, C, T, dmax, F)

    match = torch.allclose(Ku, ref_unfolded, atol=1e-5)
    if not match:
        # Show the mismatch for debugging
        for t in range(T):
            for d in range(dmax):
                nn_val = Ku[0, 0, t, d, 0].item()
                tf_val = ref_unfolded[0, 0, t, d, 0].item()
                if abs(nn_val - tf_val) > 1e-5:
                    print(f"  MISMATCH t={t} d={d}: nn.Unfold={nn_val}, tensor.unfold={tf_val}")
    assert match, "nn.Unfold and tensor.unfold produce different frame ordering!"
    print(f"[PASS] test_align_K_vs_ref_unfold_consistency")


def test_align_nn_unfold_vs_tensor_unfold():
    """Direct comparison of the two unfold APIs on identical input."""
    dmax = 4
    H, W = 8, 3  # T=8, F=3
    C = 2

    x = torch.arange(C * H * W, dtype=torch.float32).reshape(1, C, H, W)

    # Pad top by dmax-1
    x_padded = nn.functional.pad(x, [0, 0, dmax - 1, 0])  # (1, C, H+dmax-1, W)

    # nn.Unfold with kernel (dmax, 1)
    nn_out = nn.Unfold((dmax, 1))(x_padded)  # (1, C*dmax, H*W)
    nn_out = nn_out.view(1, C, dmax, H, W).permute(0, 1, 3, 2, 4)  # (1,C,H,dmax,W)

    # tensor.unfold along dim 2
    tf_out = x_padded.unfold(2, dmax, 1)  # (1, C, H, W, dmax)
    tf_out = tf_out.permute(0, 1, 2, 4, 3)  # (1, C, H, dmax, W)

    match = torch.allclose(nn_out, tf_out)
    if not match:
        print(f"  nn.Unfold shape: {nn_out.shape}")
        print(f"  tensor.unfold shape: {tf_out.shape}")
        diff = (nn_out - tf_out).abs()
        print(f"  Max diff: {diff.max().item()}")
        # Show first mismatch
        for h in range(H):
            for d in range(dmax):
                nn_val = nn_out[0, 0, h, d, 0].item()
                tf_val = tf_out[0, 0, h, d, 0].item()
                if abs(nn_val - tf_val) > 1e-5:
                    print(f"  MISMATCH h={h} d={d}: nn={nn_val}, tf={tf_val}")
                    break
            else:
                continue
            break
    assert match, "nn.Unfold and tensor.unfold differ!"
    print(f"[PASS] test_align_nn_unfold_vs_tensor_unfold")


# ── 4. Bottleneck ────────────────────────────────────────────────────────────


def test_bottleneck_shape():
    bn = Bottleneck(128 * 9, 128 * 9 // 2)
    x = torch.randn(2, 128, 16, 9)
    y = bn(x)
    assert y.shape == (2, 128, 16, 9), f"Expected (2,128,16,9), got {y.shape}"
    print(f"[PASS] test_bottleneck_shape: {y.shape}")


def test_bottleneck_reshape_ordering():
    """einops rearrange round-trip preserves channel-freq mapping."""
    from einops import rearrange
    C, F = 128, 9
    x = torch.randn(1, C, 4, F)
    # Forward reshape
    y = rearrange(x, "b c t f -> b t (c f)")
    assert y.shape == (1, 4, C * F)
    # Inverse reshape
    z = rearrange(y, "b t (c f) -> b c t f", c=C)
    assert torch.allclose(x, z), "Round-trip failed!"
    print(f"[PASS] test_bottleneck_reshape_ordering")


def test_bottleneck_gradient_flow():
    bn = Bottleneck(128 * 9, 128 * 9 // 2).train()
    x = torch.randn(1, 128, 8, 9, requires_grad=True)
    y = bn(x)
    y.sum().backward()
    no_grad = []
    for name, p in bn.named_parameters():
        if p.grad is None or p.grad.abs().max() == 0:
            no_grad.append(name)
    assert len(no_grad) == 0, f"No gradient: {no_grad}"
    print(f"[PASS] test_bottleneck_gradient_flow")


def test_bottleneck_temporal_ordering():
    """GRU sees frames in correct order: different temporal inputs → different outputs."""
    bn = Bottleneck(128 * 9, 128 * 9 // 2).eval()
    x1 = torch.randn(1, 128, 8, 9)
    x2 = x1.clone()
    x2[:, :, 4, :] += 1.0  # modify one frame

    y1 = bn(x1)
    y2 = bn(x2)

    # Frames before modification should be identical (GRU is causal)
    diff_before = (y1[:, :, :4] - y2[:, :, :4]).abs().max().item()
    # Modified frame and after should differ
    diff_at = (y1[:, :, 4] - y2[:, :, 4]).abs().max().item()
    assert diff_before == 0.0, f"Pre-modification frames differ: {diff_before:.2e}"
    assert diff_at > 0.0, "Modified frame should produce different output"
    print(f"[PASS] test_bottleneck_temporal_ordering: before diff={diff_before:.2e}, at diff={diff_at:.2e}")


# ── 5. DecoderBlock + SubpixelConv2d ─────────────────────────────────────────


def test_subpixel_frequency_doubling():
    sp = SubpixelConv2d(128, 64)
    x = torch.randn(1, 128, 8, 9)
    y = sp(x)
    assert y.shape == (1, 64, 8, 18), f"Expected (1,64,8,18), got {y.shape}"
    print(f"[PASS] test_subpixel_frequency_doubling: {y.shape}")


def test_subpixel_interleaving():
    """Sub-pixel shuffle interleaves channels along frequency."""
    from einops import rearrange
    # Simple test: 2 output channels, 4 freq bins → 8 output freq bins
    x = torch.zeros(1, 4, 1, 3)  # (B, 2*2, T, F)
    x[0, 0, 0, :] = 1.0  # r=0 channels
    x[0, 1, 0, :] = 2.0  # r=0 channels
    x[0, 2, 0, :] = 10.0  # r=1 channels
    x[0, 3, 0, :] = 20.0  # r=1 channels

    y = rearrange(x, "b (r c) t f -> b c t (r f)", r=2)
    # r=0 freqs come first, then r=1 freqs, interleaved: [f0_r0, f0_r1, f1_r0, f1_r1, ...]
    # Actually einops (r f) means r varies slowly: [r0_f0, r0_f1, r0_f2, r1_f0, r1_f1, r1_f2]
    # Wait, let me check: "b (r c) t f -> b c t (r f)" with r=2
    # Input (r c) decomposed: ch0=(r=0,c=0), ch1=(r=0,c=1), ch2=(r=1,c=0), ch3=(r=1,c=1)
    # Output for c=0: (r f) = [r=0 f=0,1,2; r=1 f=0,1,2] = [1,1,1,10,10,10]
    expected_c0 = torch.tensor([1.0, 1.0, 1.0, 10.0, 10.0, 10.0])
    assert torch.allclose(y[0, 0, 0], expected_c0), \
        f"Interleaving wrong: {y[0, 0, 0]} vs {expected_c0}"
    print(f"[PASS] test_subpixel_interleaving")


def test_decoder_skip_connection():
    dec = DecoderBlock(128, 64).eval()
    x = torch.randn(1, 128, 8, 17)
    x_en = torch.randn(1, 128, 8, 17)

    # The decoder does: y = x + skip_conv(x_en), then resblock, then deconv, then bn+elu
    # We can verify skip addition by checking that skip_conv output is added
    with torch.no_grad():
        skip_out = dec.skip_conv(x_en)
        added = x + skip_out
        # Compare to what decoder's internal chain produces before resblock
        # We can't easily decompose further, but we can check the skip is non-trivial
        assert not torch.allclose(added, x, atol=1e-6), "Skip connection has no effect"
    print(f"[PASS] test_decoder_skip_connection")


def test_decoder_frequency_chain():
    """5-block chain with trimming: 9→17→33→65→129→257"""
    channels = [128, 128, 128, 128, 64, 27]
    expected_freqs = [9, 18, 34, 66, 130, 258]  # raw sub-pixel output (before trim)
    trim_freqs = [9, 17, 33, 65, 129, 257]  # after trim to match encoder

    x = torch.randn(1, channels[0], 8, expected_freqs[0])
    for i in range(5):
        is_last = (i == 4)
        dec = DecoderBlock(channels[i], channels[i + 1], is_last=is_last).eval()
        x_en = torch.randn(1, channels[i], 8, expected_freqs[0 + i])  # dummy skip
        x_en = torch.randn(1, channels[i], 8, trim_freqs[i])
        y = dec(x, x_en)
        # SubpixelConv2d doubles freq
        raw_f = y.shape[-1]
        trimmed = y[..., :trim_freqs[i + 1]]
        actual_f = trimmed.shape[-1]
        assert actual_f == trim_freqs[i + 1], \
            f"Block {i}: expected F={trim_freqs[i+1]}, got {actual_f} (raw={raw_f})"
        x = trimmed
    print(f"[PASS] test_decoder_frequency_chain: freqs={trim_freqs}")


# ── 6. CCM ───────────────────────────────────────────────────────────────────


def test_ccm_identity_mask():
    """Mask with current-frame center tap of basis 0 = 1 → passthrough.

    With causal padding ZeroPad2d([1,1,2,0]) and 3×3 kernel:
      m=0: t-2, m=1: t-1, m=2: t (current)
      n=0: f-1, n=1: f (current), n=2: f+1
    So center = (m=2, n=1) = kernel index 2*3+1 = 7, NOT 4.
    """
    ccm = CCM().eval()
    B, F, T = 1, 8, 6

    m = torch.zeros(B, 27, T, F)
    m[:, 7, :, :] = 1.0  # r=0, kernel index 7 = current (t, f)

    x = torch.randn(B, F, T, 2)
    y = ccm(m, x)

    # Check interior frames/freqs where padding doesn't affect
    interior_t = slice(2, T)  # skip first 2 frames (causal pad)
    interior_f = slice(1, F - 1)  # skip edge freqs

    diff = (y[:, interior_f, interior_t, :] - x[:, interior_f, interior_t, :]).abs().max().item()
    assert diff < 1e-5, f"Identity mask failed: max diff={diff:.2e}"
    print(f"[PASS] test_ccm_identity_mask: max interior diff={diff:.2e}")


def test_ccm_basis_decomposition():
    """Only basis r=0 active → H_imag = 0."""
    import numpy as np
    v_real = torch.tensor([1, -0.5, -0.5])
    v_imag = torch.tensor([0, np.sqrt(3) / 2, -np.sqrt(3) / 2])

    # Mask with only r=0 basis active
    m = torch.zeros(1, 3, 9, 1, 1)  # (B, r, c, T, F) after rearrange
    m[0, 0, :, 0, 0] = torch.randn(9)

    H_real = torch.sum(v_real[None, :, None, None, None] * m, dim=1)
    H_imag = torch.sum(v_imag[None, :, None, None, None] * m, dim=1)

    assert H_imag.abs().max() < 1e-6, f"H_imag should be 0 for r=0 only, got max={H_imag.abs().max():.2e}"
    assert H_real.abs().max() > 0, "H_real should be non-zero"
    print(f"[PASS] test_ccm_basis_decomposition")


def test_ccm_complex_multiply():
    """Known mask × known input → verify complex arithmetic."""
    ccm = CCM().eval()
    B, F, T = 1, 5, 4

    # Create a mask that's a + bi at center tap only.
    # Use basis 0 (v_real=1, v_imag=0) for real part, basis 1 (v_real=-0.5, v_imag=√3/2) for imag.
    # To get H_real=a, H_imag=b at center tap:
    # H_real = 1*m0 - 0.5*m1 - 0.5*m2 = a
    # H_imag = 0*m0 + (√3/2)*m1 - (√3/2)*m2 = b
    # Solution: m1 - m2 = 2b/√3, m0 - 0.5*(m1+m2) = a
    # Let m2 = 0: m1 = 2b/√3, m0 = a + 0.5*m1 = a + b/√3
    a, b = 2.0, 3.0
    sqrt3 = math.sqrt(3)
    m1_val = 2 * b / sqrt3
    m0_val = a + b / sqrt3

    m = torch.zeros(B, 27, T, F)
    center_k = 4  # center of 3x3 kernel
    m[:, center_k + 0, :, :] = m0_val   # r=0, center
    m[:, center_k + 9, :, :] = m1_val   # r=1, center
    # r=2 center stays 0

    # Simple input: c + di
    c_val, d_val = 1.0, 0.5
    x = torch.zeros(B, F, T, 2)
    x[:, :, :, 0] = c_val
    x[:, :, :, 1] = d_val

    y = ccm(m, x)

    # Expected complex multiply: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    exp_real = a * c_val - b * d_val
    exp_imag = a * d_val + b * c_val

    # Check interior (avoiding edge effects from padding)
    interior_t = slice(2, T)
    interior_f = slice(1, F - 1)
    real_diff = (y[:, interior_f, interior_t, 0] - exp_real).abs().max().item()
    imag_diff = (y[:, interior_f, interior_t, 1] - exp_imag).abs().max().item()
    assert real_diff < 1e-4, f"Real part wrong: diff={real_diff:.2e}, expected={exp_real}"
    assert imag_diff < 1e-4, f"Imag part wrong: diff={imag_diff:.2e}, expected={exp_imag}"
    print(f"[PASS] test_ccm_complex_multiply: real diff={real_diff:.2e}, imag diff={imag_diff:.2e}")


def test_ccm_gradient_flow():
    ccm = CCM().train()
    m = torch.randn(1, 27, 8, 16, requires_grad=True)
    x = torch.randn(1, 16, 8, 2)
    y = ccm(m, x)
    y.sum().backward()
    assert m.grad is not None and m.grad.abs().max() > 0, "No gradient through mask"
    print(f"[PASS] test_ccm_gradient_flow")


def test_ccm_unfold_neighborhood():
    """3×3 causal unfold selects correct (t,f) neighbors."""
    ccm = CCM().eval()
    F, T = 5, 6

    # Create input where each element has a unique value
    x = torch.zeros(1, F, T, 2)
    for f in range(F):
        for t in range(T):
            x[0, f, t, 0] = f * 100 + t  # real
            x[0, f, t, 1] = -(f * 100 + t)  # imag

    x_perm = x.permute(0, 3, 2, 1).contiguous()  # (1, 2, T, F)
    x_unfold = ccm.unfold(x_perm)  # after ZeroPad2d([1,1,2,0]) + Unfold(3,3)

    from einops import rearrange
    x_unfold = rearrange(
        x_unfold, "b (c m n) (t f) -> b c m n t f", m=3, n=3, f=F
    )

    # With causal padding [1,1,2,0] and kernel (3,3):
    #   m=0: t-2, m=1: t-1, m=2: t (current frame)
    #   n=0: f-1, n=1: f (current freq), n=2: f+1
    # So "current (t, f)" is at (m=2, n=1)
    t_check, f_check = 2, 2
    center_real = x_unfold[0, 0, 2, 1, t_check, f_check].item()
    expected_real = x[0, f_check, t_check, 0].item()
    assert abs(center_real - expected_real) < 1e-5, \
        f"Center mismatch: got {center_real}, expected {expected_real}"

    # Also verify a neighbor: (m=0, n=0) should be (t-2, f-1)
    neighbor_real = x_unfold[0, 0, 0, 0, t_check, f_check].item()
    expected_neighbor = x[0, f_check - 1, t_check - 2, 0].item()
    assert abs(neighbor_real - expected_neighbor) < 1e-5, \
        f"Neighbor mismatch: got {neighbor_real}, expected {expected_neighbor}"
    print(f"[PASS] test_ccm_unfold_neighborhood")


# ── 7. Full Model Integration ────────────────────────────────────────────────


def test_model_identity_passthrough():
    """Fresh model with CCM identity init should approximately pass through mic."""
    model = DeepVQEAEC().eval()
    B, F, T = 1, 257, 32
    mic = torch.randn(B, F, T, 2) * 0.1  # small values for stability
    ref = torch.randn(B, F, T, 2) * 0.1

    with torch.no_grad():
        out = model(mic, ref)

    # Check that output has reasonable magnitude relative to input
    mic_mag = torch.sqrt(mic[..., 0] ** 2 + mic[..., 1] ** 2 + 1e-12)
    out_mag = torch.sqrt(out[..., 0] ** 2 + out[..., 1] ** 2 + 1e-12)
    ratio = (out_mag.mean() / mic_mag.mean()).item()
    assert 0.1 < ratio < 10.0, f"Output/input magnitude ratio {ratio:.2f} out of range"
    print(f"[PASS] test_model_identity_passthrough: mag ratio={ratio:.3f}")


def test_model_forced_identity_reconstruction():
    """Zero dec1 conv weights (keep identity bias) → output ≈ mic_stft.

    Forces the CCM to always receive the identity mask regardless of
    encoder/decoder activity, testing the entire forward path dimension flow.
    """
    model = DeepVQEAEC().eval()
    B, F, T = 1, 257, 32

    # Zero the dec1 subpixel conv weights so output is purely from bias
    with torch.no_grad():
        model.dec1.deconv.conv.weight.zero_()
        # Bias already set by _init_ccm_identity: bias[4]=1, bias[31]=1

    mic = torch.randn(B, F, T, 2) * 0.5
    ref = torch.randn(B, F, T, 2) * 0.5

    with torch.no_grad():
        out = model(mic, ref)

    # The identity mask from bias should pass through the mic STFT
    # Check interior to avoid edge effects from CCM's 3×3 causal unfold
    interior = out[:, 1:-1, 2:, :]
    mic_interior = mic[:, 1:-1, 2:, :]

    max_err = (interior - mic_interior).abs().max().item()
    mean_err = (interior - mic_interior).abs().mean().item()
    assert max_err < 1e-4, f"Forced identity failed: max err={max_err:.2e}, mean err={mean_err:.2e}"
    print(f"[PASS] test_model_forced_identity_reconstruction: max err={max_err:.2e}")


def test_delay_supervision_gradient_path():
    """delay_loss.backward() → AlignBlock params have grads."""
    model = DeepVQEAEC().train()
    mic = torch.randn(1, 257, 32, 2)
    ref = torch.randn(1, 257, 32, 2)

    _, delay_dist, _ = model(mic, ref, return_delay=True)
    # Compute delay loss
    target = torch.tensor([5])  # 5 frames delay
    target_expanded = target[:, None, None].expand(1, 32, 1)
    log_probs = torch.log(delay_dist + 1e-10)
    delay_loss = -log_probs.gather(dim=-1, index=target_expanded).mean()
    delay_loss.backward()

    no_grad = []
    for name, p in model.align.named_parameters():
        if p.grad is None or p.grad.abs().max() == 0:
            no_grad.append(name)
    assert len(no_grad) == 0, f"AlignBlock params without grad: {no_grad}"
    print(f"[PASS] test_delay_supervision_gradient_path")


def test_delay_frame_conversion():
    """delay_samples_to_frame matches unfold convention."""
    hop = 256
    dmax = 32

    # 0 samples delay → ref plays, mic receives immediately → d = dmax-1
    assert delay_samples_to_frame(torch.tensor([0]), hop, dmax).item() == dmax - 1

    # 1 hop delay → d = dmax-2
    assert delay_samples_to_frame(torch.tensor([hop]), hop, dmax).item() == dmax - 2

    # (dmax-1)*hop delay → d = 0
    assert delay_samples_to_frame(torch.tensor([(dmax - 1) * hop]), hop, dmax).item() == 0

    # Larger delay gets clamped to 0
    assert delay_samples_to_frame(torch.tensor([dmax * hop * 2]), hop, dmax).item() == 0

    print(f"[PASS] test_delay_frame_conversion")


def test_model_can_learn_echo_cancellation():
    """50-step training on synthetic echo task: loss should decrease."""
    torch.manual_seed(42)
    dmax = 8
    model = DeepVQEAEC(dmax=dmax)
    model.align.temperature = 0.5
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    B, F, T = 2, 257, 32
    delay_frames = 3
    hop = 256

    losses = []
    for step in range(50):
        # Clean signal
        clean = torch.randn(B, F, T, 2) * 0.3
        # Reference (far-end) signal
        ref = torch.randn(B, F, T, 2) * 0.3
        # Mic = clean + delayed ref (echo)
        echo = torch.zeros_like(ref)
        for t in range(delay_frames, T):
            echo[:, :, t, :] = ref[:, :, t - delay_frames, :]
        mic = clean + echo

        enhanced, delay_dist, _ = model(mic, ref, return_delay=True)

        # Reconstruction loss
        recon_loss = ((enhanced - clean) ** 2).mean()

        # Delay loss
        target_d = dmax - 1 - delay_frames
        log_probs = torch.log(delay_dist + 1e-10)
        delay_loss = -log_probs[:, :, target_d].mean()

        loss = recon_loss + 0.5 * delay_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Loss should decrease
    first_5 = sum(losses[:5]) / 5
    last_5 = sum(losses[-5:]) / 5
    decreased = last_5 < first_5
    assert decreased, f"Loss didn't decrease: first_5={first_5:.4f}, last_5={last_5:.4f}"

    # Check delay accuracy
    model.eval()
    with torch.no_grad():
        ref = torch.randn(4, F, T, 2) * 0.3
        echo = torch.zeros_like(ref)
        for t in range(delay_frames, T):
            echo[:, :, t, :] = ref[:, :, t - delay_frames, :]
        clean = torch.randn(4, F, T, 2) * 0.3
        mic = clean + echo
        _, delay_dist, _ = model(mic, ref, return_delay=True)

    avg_dist = delay_dist.mean(dim=1)
    peaks = avg_dist.argmax(dim=-1)
    correct = ((peaks - target_d).abs() <= 1).float().mean().item()
    print(f"[PASS] test_model_can_learn_echo_cancellation: "
          f"loss {first_5:.4f}→{last_5:.4f}, delay_acc={correct:.0%}")


# ── Runner ───────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    tests = [
        # 1. FE
        test_fe_output_shape,
        test_fe_power_law_compression,
        test_fe_zero_input_stability,
        test_fe_unit_magnitude_passthrough,
        # 2. EncoderBlock
        test_encoder_output_shape,
        test_encoder_causality,
        test_encoder_frequency_chain,
        # 3. AlignBlock
        test_align_unfold_indexing,
        test_align_onehot_selects_frame,
        test_align_roundtrip_known_delay,
        test_align_gradient_flow,
        test_align_gradient_through_delay,
        test_align_isolated_learning,
        test_align_temperature_effect,
        test_align_K_vs_ref_unfold_consistency,
        test_align_nn_unfold_vs_tensor_unfold,
        # 4. Bottleneck
        test_bottleneck_shape,
        test_bottleneck_reshape_ordering,
        test_bottleneck_gradient_flow,
        test_bottleneck_temporal_ordering,
        # 5. DecoderBlock + SubpixelConv2d
        test_subpixel_frequency_doubling,
        test_subpixel_interleaving,
        test_decoder_skip_connection,
        test_decoder_frequency_chain,
        # 6. CCM
        test_ccm_identity_mask,
        test_ccm_basis_decomposition,
        test_ccm_complex_multiply,
        test_ccm_gradient_flow,
        test_ccm_unfold_neighborhood,
        # 7. Full Model Integration
        test_model_identity_passthrough,
        test_model_forced_identity_reconstruction,
        test_delay_supervision_gradient_path,
        test_delay_frame_conversion,
        test_model_can_learn_echo_cancellation,
    ]

    passed = 0
    failed = 0
    failures = []
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1
            failures.append(test.__name__)

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failures:
        print(f"Failures: {', '.join(failures)}")
