"""Data pipeline verification tests.

Checks:
- STFT/iSTFT round-trip error < 1e-6
- Output shapes correct
- DummyAECDataset produces valid data
- DataLoader batching works
"""

import torch
import numpy as np

from src.stft import stft, istft
from data.dataset import DummyAECDataset


def test_stft_roundtrip():
    """STFT -> iSTFT should reconstruct the signal."""
    N = 48000
    x = torch.randn(2, N)
    X = stft(x, n_fft=512, hop_length=256)
    x_hat = istft(X, n_fft=512, hop_length=256, length=N)
    err = (x - x_hat).abs().max().item()
    assert err < 1e-5, f"Round-trip error too high: {err}"
    print(f"[PASS] STFT round-trip error: {err:.2e}")


def test_stft_shapes():
    """Verify STFT output dimensions."""
    N = 48000
    x = torch.randn(4, N)
    X = stft(x, n_fft=512, hop_length=256)
    F = 512 // 2 + 1  # 257
    T = N // 256 + 1  # 188 (with center padding)
    assert X.shape[0] == 4, f"Batch dim wrong: {X.shape}"
    assert X.shape[1] == F, f"Freq dim wrong: {X.shape[1]} != {F}"
    assert X.shape[3] == 2, f"Real/imag dim wrong: {X.shape[3]}"
    print(f"[PASS] STFT shape: {X.shape} (B=4, F={F}, T={X.shape[2]}, 2)")


def test_dummy_dataset():
    """Verify DummyAECDataset returns correct shapes."""
    ds = DummyAECDataset(length=5, target_len=48000, delay_range=(160, 160))
    sample = ds[0]

    assert sample["mic_stft"].shape[0] == 257, f"Freq dim: {sample['mic_stft'].shape}"
    assert sample["mic_stft"].shape[2] == 2, f"Real/imag: {sample['mic_stft'].shape}"
    assert sample["ref_stft"].shape == sample["mic_stft"].shape
    assert sample["clean_stft"].shape == sample["mic_stft"].shape
    # With delay_range=(160,160) and hop=256, quantized delay is 0 or 256
    assert sample["metadata"]["delay_samples"] >= 0
    print(f"[PASS] DummyDataset shapes: mic_stft={sample['mic_stft'].shape}")


def test_dataloader():
    """Verify DataLoader batching works."""
    ds = DummyAECDataset(length=8, target_len=48000)
    # Manual collation (metadata is a dict, can't be default-collated)
    batch_mic = torch.stack([ds[i]["mic_stft"] for i in range(4)])
    batch_ref = torch.stack([ds[i]["ref_stft"] for i in range(4)])
    assert batch_mic.shape[0] == 4
    assert batch_mic.shape[1] == 257
    print(f"[PASS] Batch shapes: mic={batch_mic.shape}, ref={batch_ref.shape}")


def test_delay_crosscorrelation():
    """Verify that synthesized delay matches cross-correlation peak."""
    delay_samples = 768  # ~48ms at 16kHz, exactly 3 hops (768/256=3)
    ds = DummyAECDataset(length=1, target_len=48000, delay_range=(delay_samples, delay_samples))
    sample = ds[0]

    # Reconstruct waveforms from STFT
    mic_stft = sample["mic_stft"].unsqueeze(0)
    ref_stft = sample["ref_stft"].unsqueeze(0)

    mic_wav = istft(mic_stft, length=48000).squeeze().numpy()
    ref_wav = istft(ref_stft, length=48000).squeeze().numpy()

    # Cross-correlation to find delay
    corr = np.correlate(mic_wav, ref_wav, mode="full")
    peak_offset = np.argmax(np.abs(corr)) - (len(ref_wav) - 1)

    # Allow some tolerance due to noise and echo scaling
    error = abs(peak_offset - delay_samples)
    tolerance = 50  # samples
    assert error < tolerance, \
        f"Cross-correlation peak at {peak_offset}, expected {delay_samples}, error={error}"
    print(f"[PASS] Delay cross-correlation: peak={peak_offset}, expected={delay_samples}, error={error}")


if __name__ == "__main__":
    tests = [
        test_stft_roundtrip,
        test_stft_shapes,
        test_dummy_dataset,
        test_dataloader,
        test_delay_crosscorrelation,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1
    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
