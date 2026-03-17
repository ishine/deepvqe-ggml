"""Component isolation tests for DeepVQE mask learning.

Tests each network component in isolation to find where echo cancellation
learning breaks down. Complements test_blocks.py (which tests correctness)
by testing *learnability* — can each component learn the right output
via gradient descent?

Run: python test_ccm_learning.py
"""

import math

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from src.blocks import DecoderBlock
from src.ccm import CCM
from src.losses import mask_mag_from_raw
from src.model import DeepVQEAEC


# ── Helpers ──────────────────────────────────────────────────────────────────


def compute_ideal_mask_27ch(clean_stft, mic_stft):
    """Compute the oracle 27-channel mask for CCM (single center tap).

    Solves for m[r=0,1,2] at kernel index 7 such that CCM output = clean.
    Uses minimum-norm solution of the underdetermined basis system.

    Args:
        clean_stft: (B, F, T, 2) target STFT
        mic_stft: (B, F, T, 2) microphone STFT

    Returns:
        mask: (B, 27, T, F) oracle 27-channel mask
    """
    B, F, T, _ = clean_stft.shape
    eps = 1e-8

    # Complex division: H = clean / mic
    mic_r, mic_i = mic_stft[..., 0], mic_stft[..., 1]
    cln_r, cln_i = clean_stft[..., 0], clean_stft[..., 1]
    denom = mic_r ** 2 + mic_i ** 2 + eps
    H_real = (cln_r * mic_r + cln_i * mic_i) / denom  # (B, F, T)
    H_imag = (cln_i * mic_r - cln_r * mic_i) / denom  # (B, F, T)

    # Transpose to (B, T, F) to match mask layout
    H_real = H_real.transpose(1, 2)  # (B, T, F)
    H_imag = H_imag.transpose(1, 2)  # (B, T, F)

    # Solve: v_real . m = H_real, v_imag . m = H_imag
    # v_real = [1, -0.5, -0.5], v_imag = [0, sqrt(3)/2, -sqrt(3)/2]
    # Minimum-norm solution:
    sqrt3 = math.sqrt(3)
    m0 = (2.0 / 3.0) * H_real
    m1 = (-1.0 / 3.0) * H_real + (1.0 / sqrt3) * H_imag
    m2 = (-1.0 / 3.0) * H_real - (1.0 / sqrt3) * H_imag

    # Place at kernel index 7, all other taps zero
    mask = torch.zeros(B, 27, T, F)
    mask[:, 7, :, :] = m0       # basis 0, kernel 7
    mask[:, 7 + 9, :, :] = m1   # basis 1, kernel 7
    mask[:, 7 + 18, :, :] = m2  # basis 2, kernel 7
    return mask


def basis_group_stats(mask_27ch):
    """Compute per-basis-group statistics from raw 27-channel mask.

    Returns dict with basis means at center tap, H_real/H_imag, |H|, spread.
    """
    m = rearrange(mask_27ch.detach(), "b (r c) t f -> b r c t f", r=3)
    center = 7  # kernel index for current (t, f)
    basis_means = [m[:, r, center].mean().item() for r in range(3)]

    v_r = [1.0, -0.5, -0.5]
    v_i = [0.0, math.sqrt(3) / 2, -math.sqrt(3) / 2]

    H_real = sum(v_r[r] * basis_means[r] for r in range(3))
    H_imag = sum(v_i[r] * basis_means[r] for r in range(3))
    H_mag = math.sqrt(H_real ** 2 + H_imag ** 2)
    max_diff = max(
        abs(basis_means[i] - basis_means[j])
        for i in range(3)
        for j in range(i + 1, 3)
    )
    return {
        "basis_means": basis_means,
        "H_real": H_real,
        "H_imag": H_imag,
        "H_mag": H_mag,
        "max_basis_diff": max_diff,
    }


# ── Test 1: CCM Oracle Mask ─────────────────────────────────────────────────


def test_ccm_oracle_mask():
    """Feed mathematically ideal 27ch mask to CCM → output should equal clean."""
    torch.manual_seed(42)
    B, F, T = 2, 257, 32

    clean_stft = torch.randn(B, F, T, 2) * 0.3
    mic_stft = torch.randn(B, F, T, 2) * 0.5

    mask = compute_ideal_mask_27ch(clean_stft, mic_stft)
    ccm = CCM().eval()
    with torch.no_grad():
        output = ccm(mask, mic_stft)

    # Check interior (skip t<2 for causal pad, skip edge freqs)
    interior = output[:, 1:-1, 2:, :] - clean_stft[:, 1:-1, 2:, :]
    max_err = interior.abs().max().item()
    mean_err = interior.abs().mean().item()

    print(f"  Oracle mask: max_err={max_err:.2e}, mean_err={mean_err:.2e}")
    assert max_err < 1e-4, f"Oracle mask failed: max_err={max_err:.2e}"
    print(f"[PASS] test_ccm_oracle_mask")


# ── Test 2: CCM Mask Learnability ────────────────────────────────────────────


def test_ccm_mask_learnability():
    """Directly optimize a mask parameter through CCM to match target."""
    torch.manual_seed(42)
    B, F, T = 4, 257, 32

    # Target = lowpass-filtered mic (same phase, frequency-dependent gain)
    mic_stft = torch.randn(B, F, T, 2) * 0.5
    gain = torch.exp(-torch.arange(F, dtype=torch.float32) / 100)
    clean_stft = mic_stft * gain[None, :, None, None]

    ccm = CCM().eval()

    # Learnable mask, identity-initialized
    mask = nn.Parameter(torch.zeros(B, 27, T, F))
    with torch.no_grad():
        mask[:, 7, :, :] = 1.0  # basis 0, center tap = passthrough

    optimizer = torch.optim.Adam([mask], lr=0.01)

    # Interior slice for loss (avoid causal padding + edge freq artifacts)
    t_sl = slice(2, T)
    f_sl = slice(1, F - 1)

    initial_loss = None
    for step in range(200):
        optimizer.zero_grad()
        output = ccm(mask, mic_stft)
        loss = (output[:, f_sl, t_sl, :] - clean_stft[:, f_sl, t_sl, :]).pow(2).mean()
        if step == 0:
            initial_loss = loss.item()
        loss.backward()
        optimizer.step()

    final_loss = loss.item()
    ratio = final_loss / (initial_loss + 1e-12)

    # Output SNR on interior
    with torch.no_grad():
        output = ccm(mask, mic_stft)
        signal_pow = clean_stft[:, f_sl, t_sl, :].pow(2).mean()
        error_pow = (output[:, f_sl, t_sl, :] - clean_stft[:, f_sl, t_sl, :]).pow(2).mean()
        snr = 10 * torch.log10(signal_pow / (error_pow + 1e-12)).item()

    # Mask stats
    mag = mask_mag_from_raw(mask.data)
    center_mag = mag[:, 7].mean().item()

    print(f"  loss: {initial_loss:.4e} → {final_loss:.4e} ({ratio:.4f}x)")
    print(f"  output SNR: {snr:.1f} dB")
    print(f"  center tap |H|: {center_mag:.4f}")
    assert ratio < 0.01, f"Loss didn't decrease enough: ratio={ratio:.4f}"
    assert snr > 30, f"Output SNR too low: {snr:.1f} dB"
    print(f"[PASS] test_ccm_mask_learnability")


# ── Test 3: dec1 + CCM Learnability ──────────────────────────────────────────


def test_dec1_mask_learnability():
    """Train just dec1 (DecoderBlock) + CCM to produce correct mask."""
    torch.manual_seed(42)
    B, F, T = 4, 257, 32

    mic_stft = torch.randn(B, F, T, 2) * 0.3
    clean_stft = mic_stft * 0.5  # simple 6dB attenuation target

    # Fixed inputs to dec1 (simulating dec2 output and enc1 skip)
    x_main = torch.randn(B, 64, T, 129)  # from dec2
    x_skip = torch.randn(B, 64, T, 129)  # from enc1

    dec1 = DecoderBlock(64, 27, is_last=True)
    ccm = CCM().eval()

    # Identity init on dec1 deconv bias
    with torch.no_grad():
        dec1.deconv.conv.bias.zero_()
        dec1.deconv.conv.bias[7] = 1.0
        dec1.deconv.conv.bias[34] = 1.0

    params = list(dec1.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)

    t_sl = slice(2, T)
    f_sl = slice(1, F - 1)

    initial_loss = None
    for step in range(300):
        optimizer.zero_grad()
        d1 = dec1(x_main, x_skip)[..., :F]
        output = ccm(d1, mic_stft)
        loss = (output[:, f_sl, t_sl, :] - clean_stft[:, f_sl, t_sl, :]).pow(2).mean()
        if step == 0:
            initial_loss = loss.item()
        loss.backward()
        optimizer.step()

    final_loss = loss.item()
    ratio = final_loss / (initial_loss + 1e-12)

    # Mask stats
    with torch.no_grad():
        d1 = dec1(x_main, x_skip)[..., :F]
    mag = mask_mag_from_raw(d1)
    center_mag = mag[:, 7].mean().item()
    stats = basis_group_stats(d1)

    print(f"  loss: {initial_loss:.4e} → {final_loss:.4e} ({ratio:.4f}x)")
    print(f"  center tap |H|: {center_mag:.4f}")
    print(f"  basis means: {[f'{m:.4f}' for m in stats['basis_means']]}")
    print(f"  basis diff: {stats['max_basis_diff']:.4f}, H_mag: {stats['H_mag']:.4f}")
    assert ratio < 0.5, f"Loss didn't decrease enough: ratio={ratio:.4f}"
    assert center_mag > 0.3, f"Center tap |H| too low: {center_mag:.4f}"
    print(f"[PASS] test_dec1_mask_learnability")


# ── Test 4: Full Decoder Chain + CCM ─────────────────────────────────────────


def test_decoder_chain_learnability():
    """Train all 5 decoder blocks to produce correct mask from fixed features."""
    torch.manual_seed(42)
    B, F, T = 2, 257, 32

    mic_stft = torch.randn(B, F, T, 2) * 0.3
    clean_stft = mic_stft * 0.5  # 6dB attenuation

    # Fixed encoder outputs (skip connections)
    enc5_out = torch.randn(B, 128, T, 9)
    enc4_out = torch.randn(B, 128, T, 17)
    enc3_out = torch.randn(B, 128, T, 33)
    enc2_out = torch.randn(B, 128, T, 65)
    enc1_out = torch.randn(B, 64, T, 129)
    bn_out = torch.randn(B, 128, T, 9)

    # Create decoder chain + mask head
    dec5 = DecoderBlock(128, 128)
    dec4 = DecoderBlock(128, 128)
    dec3 = DecoderBlock(128, 128)
    dec2 = DecoderBlock(128, 64)
    dec1 = DecoderBlock(64, 27, is_last=True)
    ccm = CCM().eval()

    # Identity init on dec1 deconv bias
    with torch.no_grad():
        dec1.deconv.conv.bias.zero_()
        dec1.deconv.conv.bias[7] = 1.0
        dec1.deconv.conv.bias[34] = 1.0

    all_params = []
    for dec in [dec5, dec4, dec3, dec2, dec1]:
        all_params.extend(dec.parameters())
    optimizer = torch.optim.Adam(all_params, lr=1e-3)

    t_sl = slice(2, T)
    f_sl = slice(1, F - 1)

    initial_loss = None
    for step in range(500):
        optimizer.zero_grad()
        d5 = dec5(bn_out, enc5_out)[..., :17]
        d4 = dec4(d5, enc4_out)[..., :33]
        d3 = dec3(d4, enc3_out)[..., :65]
        d2 = dec2(d3, enc2_out)[..., :129]
        d1 = dec1(d2, enc1_out)[..., :F]
        output = ccm(d1, mic_stft)
        loss = (output[:, f_sl, t_sl, :] - clean_stft[:, f_sl, t_sl, :]).pow(2).mean()
        if step == 0:
            initial_loss = loss.item()
        loss.backward()
        optimizer.step()

    final_loss = loss.item()
    ratio = final_loss / (initial_loss + 1e-12)

    with torch.no_grad():
        d5 = dec5(bn_out, enc5_out)[..., :17]
        d4 = dec4(d5, enc4_out)[..., :33]
        d3 = dec3(d4, enc3_out)[..., :65]
        d2 = dec2(d3, enc2_out)[..., :129]
        d1_final = dec1(d2, enc1_out)[..., :F]

    mag = mask_mag_from_raw(d1_final)
    center_mag = mag[:, 7].mean().item()

    print(f"  loss: {initial_loss:.4e} → {final_loss:.4e} ({ratio:.4f}x)")
    print(f"  d1 std: {d1_final.std():.4f}")
    print(f"  center tap |H|: {center_mag:.4f}")
    assert ratio < 0.3, f"Loss didn't decrease enough: ratio={ratio:.4f}"
    assert d1_final.std() > 0.1, f"d1 std collapsed: {d1_final.std():.4f}"
    print(f"[PASS] test_decoder_chain_learnability")


# ── Test 5: Encoder Information Probe ────────────────────────────────────────


def test_encoder_information_probe():
    """Linear probe on enc5 output to predict ideal magnitude mask."""
    torch.manual_seed(42)
    B, F, T = 8, 257, 32

    # Synthetic echo cancellation data
    clean_stft = torch.randn(B, F, T, 2) * 0.3
    echo_stft = torch.randn(B, F, T, 2) * 0.3
    mic_stft = clean_stft + echo_stft
    ref_stft = torch.randn(B, F, T, 2) * 0.3  # far-end

    # Ideal magnitude mask
    mic_mag = torch.sqrt(mic_stft[..., 0] ** 2 + mic_stft[..., 1] ** 2 + 1e-12)
    cln_mag = torch.sqrt(clean_stft[..., 0] ** 2 + clean_stft[..., 1] ** 2 + 1e-12)
    target_mask = cln_mag / (mic_mag + 1e-8)  # (B, F, T)
    target_mask = target_mask.transpose(1, 2)  # (B, T, F)

    # Frozen model — extract encoder features
    model = DeepVQEAEC(dmax=8)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        mic_fe = model.fe_mic(mic_stft)
        ref_fe = model.fe_ref(ref_stft)
        mic_e1 = model.mic_enc1(mic_fe)
        mic_e2 = model.mic_enc2(mic_e1)
        far_e1 = model.far_enc1(ref_fe)
        far_e2 = model.far_enc2(far_e1)
        aligned, _ = model.align(mic_e2, far_e2, return_delay=True)
        concat = torch.cat([mic_e2, aligned], dim=1)
        mic_e3 = model.mic_enc3(concat)
        mic_e4 = model.mic_enc4(mic_e3)
        mic_e5 = model.mic_enc5(mic_e4)
        # enc5 features: (B, 128, T, 9) → flatten to (B, T, 1152)
        enc5_flat = rearrange(mic_e5, "b c t f -> b t (c f)")

    # Linear probe
    probe = nn.Linear(1152, F)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)

    initial_loss = None
    for step in range(300):
        optimizer.zero_grad()
        pred = probe(enc5_flat)  # (B, T, F)
        loss = (pred - target_mask).pow(2).mean()
        if step == 0:
            initial_loss = loss.item()
        loss.backward()
        optimizer.step()

    final_loss = loss.item()

    # R² score
    with torch.no_grad():
        pred = probe(enc5_flat)
        ss_res = (target_mask - pred).pow(2).sum()
        ss_tot = (target_mask - target_mask.mean()).pow(2).sum()
        r2 = 1 - (ss_res / (ss_tot + 1e-12))
        r2 = r2.item()

    print(f"  probe loss: {initial_loss:.4e} → {final_loss:.4e}")
    print(f"  R²: {r2:.4f}")
    assert r2 > 0.1, f"Encoder probe R² too low: {r2:.4f}"
    print(f"[PASS] test_encoder_information_probe")


# ── Test 6: Bottleneck Information Probe ─────────────────────────────────────


def test_bottleneck_information_probe():
    """Compare linear probes: enc5 alone vs bottleneck alone vs bottleneck + skips.

    The GRU bottleneck is designed to compress — it only needs to carry
    global temporal context. The skip connections from encoder stages
    provide the local spectral detail. So we test three conditions:
    1. enc5 alone (pre-bottleneck, full info)
    2. bottleneck alone (post-GRU, compressed)
    3. bottleneck + enc5 skip (what the decoder actually sees at dec5)
    """
    torch.manual_seed(42)
    B, F, T = 8, 257, 32

    clean_stft = torch.randn(B, F, T, 2) * 0.3
    echo_stft = torch.randn(B, F, T, 2) * 0.3
    mic_stft = clean_stft + echo_stft
    ref_stft = torch.randn(B, F, T, 2) * 0.3

    mic_mag = torch.sqrt(mic_stft[..., 0] ** 2 + mic_stft[..., 1] ** 2 + 1e-12)
    cln_mag = torch.sqrt(clean_stft[..., 0] ** 2 + clean_stft[..., 1] ** 2 + 1e-12)
    target_mask = (cln_mag / (mic_mag + 1e-8)).transpose(1, 2)  # (B, T, F)

    model = DeepVQEAEC(dmax=8)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        mic_fe = model.fe_mic(mic_stft)
        ref_fe = model.fe_ref(ref_stft)
        mic_e1 = model.mic_enc1(mic_fe)
        mic_e2 = model.mic_enc2(mic_e1)
        far_e1 = model.far_enc1(ref_fe)
        far_e2 = model.far_enc2(far_e1)
        aligned, _ = model.align(mic_e2, far_e2, return_delay=True)
        concat = torch.cat([mic_e2, aligned], dim=1)
        mic_e3 = model.mic_enc3(concat)
        mic_e4 = model.mic_enc4(mic_e3)
        mic_e5 = model.mic_enc5(mic_e4)
        bn_out = model.bottleneck(mic_e5)

        enc5_flat = rearrange(mic_e5, "b c t f -> b t (c f)")    # (B, T, 1152)
        bn_flat = rearrange(bn_out, "b c t f -> b t (c f)")      # (B, T, 1152)
        # Bottleneck + enc5 skip (what dec5 actually receives)
        bn_skip_flat = torch.cat([bn_flat, enc5_flat], dim=-1)    # (B, T, 2304)

    def train_probe(features, target, name):
        torch.manual_seed(123)
        probe = nn.Linear(features.shape[-1], F)
        opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
        for step in range(300):
            opt.zero_grad()
            pred = probe(features)
            loss = (pred - target).pow(2).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred = probe(features)
            ss_res = (target - pred).pow(2).sum()
            ss_tot = (target - target.mean()).pow(2).sum()
            r2 = (1 - ss_res / (ss_tot + 1e-12)).item()
        return r2, loss.item()

    r2_enc, loss_enc = train_probe(enc5_flat, target_mask, "enc5")
    r2_bn, loss_bn = train_probe(bn_flat, target_mask, "bottleneck")
    r2_combo, loss_combo = train_probe(bn_skip_flat, target_mask, "bn+skip")

    print(f"  enc5 alone:       R²={r2_enc:.4f}, loss={loss_enc:.4e}")
    print(f"  bottleneck alone: R²={r2_bn:.4f}, loss={loss_bn:.4e}")
    print(f"  bn + enc5 skip:   R²={r2_combo:.4f}, loss={loss_combo:.4e}")
    # The combined features (what decoder actually sees) should be at least
    # as good as enc5 alone — the skip restores what the GRU compressed away
    assert r2_combo >= r2_enc * 0.8, (
        f"Skip connections don't recover info: enc R²={r2_enc:.4f}, bn+skip R²={r2_combo:.4f}"
    )
    print(f"[PASS] test_bottleneck_information_probe")


# ── Test 7: Gradient Magnitude Map ───────────────────────────────────────────


def test_gradient_magnitude_map():
    """Map gradient norms per component on a single forward+backward pass."""
    torch.manual_seed(42)
    B, F, T = 2, 257, 32

    model = DeepVQEAEC(dmax=8)
    model.train()

    mic_stft = torch.randn(B, F, T, 2) * 0.3
    ref_stft = torch.randn(B, F, T, 2) * 0.3
    clean_stft = torch.randn(B, F, T, 2) * 0.3

    # Average over a few random seeds for stability
    grad_norms = {}
    n_trials = 3
    for trial in range(n_trials):
        torch.manual_seed(42 + trial)
        mic_stft = torch.randn(B, F, T, 2) * 0.3
        ref_stft = torch.randn(B, F, T, 2) * 0.3
        clean_stft = torch.randn(B, F, T, 2) * 0.3

        model.zero_grad()
        enhanced = model(mic_stft, ref_stft)
        loss = (enhanced - clean_stft).pow(2).mean()
        loss.backward()

        components = [
            ("fe_mic", model.fe_mic),
            ("fe_ref", model.fe_ref),
            ("mic_enc1", model.mic_enc1),
            ("mic_enc2", model.mic_enc2),
            ("far_enc1", model.far_enc1),
            ("far_enc2", model.far_enc2),
            ("align", model.align),
            ("mic_enc3", model.mic_enc3),
            ("mic_enc4", model.mic_enc4),
            ("mic_enc5", model.mic_enc5),
            ("bottleneck", model.bottleneck),
            ("dec5", model.dec5),
            ("dec4", model.dec4),
            ("dec3", model.dec3),
            ("dec2", model.dec2),
            ("dec1", model.dec1),
        ]

        for name, mod in components:
            total = 0.0
            count = 0
            for p in mod.parameters():
                if p.grad is not None:
                    total += p.grad.norm().item() ** 2
                    count += 1
            norm = math.sqrt(total) if total > 0 else 0.0
            if name not in grad_norms:
                grad_norms[name] = []
            grad_norms[name].append(norm)

    # Print table
    print(f"  {'Component':<14} {'Grad Norm':>12} {'Params':>8}")
    print(f"  {'-'*14} {'-'*12} {'-'*8}")
    dec1_norm = np.mean(grad_norms.get("dec1", [0]))
    any_zero = False
    for name, mod in components:
        avg_norm = np.mean(grad_norms[name])
        n_params = sum(p.numel() for p in mod.parameters())
        ratio = avg_norm / (dec1_norm + 1e-12)
        print(f"  {name:<14} {avg_norm:12.4e} {n_params:8d}  ({ratio:.2f}x dec1)")
        if avg_norm == 0 and n_params > 0:
            any_zero = True

    # Check dec1 to enc1 ratio
    enc1_norm = np.mean(grad_norms.get("mic_enc1", [0]))
    ratio = dec1_norm / (enc1_norm + 1e-12) if enc1_norm > 0 else float("inf")

    assert not any_zero, "Some component has zero gradient!"
    assert ratio < 1000, f"Gradient vanishing: dec1/enc1 ratio={ratio:.1f}"
    print(f"[PASS] test_gradient_magnitude_map")


# ── Runner ───────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    tests = [
        test_ccm_oracle_mask,
        test_ccm_mask_learnability,
        test_dec1_mask_learnability,
        test_decoder_chain_learnability,
        test_encoder_information_probe,
        test_bottleneck_information_probe,
        test_gradient_magnitude_map,
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

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failures:
        print(f"Failures: {', '.join(failures)}")
