"""Evaluation metrics for DeepVQE AEC.

Metrics:
- ERLE: Echo Return Loss Enhancement
- PESQ: Perceptual Evaluation of Speech Quality (wideband)
- STOI: Short-Time Objective Intelligibility (extended)
- Segmental SNR improvement
- DNSMOS P.835: SIG, BAK, OVRL (requires onnxruntime)
- AECMOS: echo_mos, deg_mos (requires onnxruntime)
"""

import hashlib
import urllib.request
from pathlib import Path

import numpy as np


def erle(mic_wav, enhanced_wav, clean_wav, frame_len=512, eps=1e-10):
    """Compute Echo Return Loss Enhancement (ERLE) in dB.

    ERLE = 10 * log10(E[echo^2] / E[residual_echo^2])
    where echo = mic - clean, residual_echo = enhanced - clean.

    Args:
        mic_wav: (N,) numpy array, microphone signal
        enhanced_wav: (N,) numpy array, enhanced signal
        clean_wav: (N,) numpy array, clean near-end signal
        frame_len: frame length for segmental computation

    Returns:
        erle_db: scalar, overall ERLE in dB
        erle_frames: (num_frames,) per-frame ERLE in dB
    """
    echo = mic_wav - clean_wav
    residual = enhanced_wav - clean_wav

    # Overall ERLE
    echo_power = np.mean(echo**2) + eps
    residual_power = np.mean(residual**2) + eps
    erle_db = 10 * np.log10(echo_power / residual_power)

    # Per-frame ERLE
    n_frames = len(echo) // frame_len
    erle_frames = np.zeros(n_frames)
    for i in range(n_frames):
        s = i * frame_len
        e = s + frame_len
        ep = np.mean(echo[s:e] ** 2) + eps
        rp = np.mean(residual[s:e] ** 2) + eps
        erle_frames[i] = 10 * np.log10(ep / rp)

    return erle_db, erle_frames


def segmental_snr(clean_wav, enhanced_wav, frame_len=512, eps=1e-10):
    """Compute segmental SNR improvement in dB.

    Args:
        clean_wav: (N,) numpy array
        enhanced_wav: (N,) numpy array
        frame_len: frame length

    Returns:
        seg_snr: scalar, average segmental SNR in dB
    """
    noise = enhanced_wav - clean_wav
    n_frames = len(clean_wav) // frame_len
    snrs = np.zeros(n_frames)
    for i in range(n_frames):
        s = i * frame_len
        e = s + frame_len
        sig_pow = np.mean(clean_wav[s:e] ** 2) + eps
        noise_pow = np.mean(noise[s:e] ** 2) + eps
        snrs[i] = 10 * np.log10(sig_pow / noise_pow)
    # Clamp to [-10, 35] dB as is standard
    snrs = np.clip(snrs, -10, 35)
    return float(np.mean(snrs))


def compute_pesq(clean_wav, enhanced_wav, sr=16000):
    """Compute wideband PESQ score.

    Returns None if pesq library is not installed.
    """
    try:
        from pesq import pesq as pesq_fn

        score = pesq_fn(sr, clean_wav, enhanced_wav, "wb")
        return float(score)
    except ImportError:
        return None
    except Exception:
        return None


def compute_stoi(clean_wav, enhanced_wav, sr=16000):
    """Compute extended STOI score.

    Returns None if pystoi library is not installed.
    """
    try:
        from pystoi import stoi

        score = stoi(clean_wav, enhanced_wav, sr, extended=True)
        return float(score)
    except ImportError:
        return None
    except Exception:
        return None


# ── DNSMOS P.835 ─────────────────────────────────────────────────────────────

_DNSMOS_MODELS_DIR = Path(".cache/dnsmos/models")
_DNSMOS_MODELS = {
    "sig_bak_ovr.onnx": {
        "url": "https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx",
        "sha256": "269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd",
    },
    "model_v8.onnx": {
        "url": "https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/model_v8.onnx",
        "sha256": "9246480c58567bc6affd4200938e77eef49468c8bc7ed3776d109c07456f6e91",
    },
}
_DNSMOS_COEFS_SIG = [-0.08397278, 1.22083953, 0.0052439]
_DNSMOS_COEFS_BAK = [-0.13166888, 1.60915514, -0.39604546]
_DNSMOS_COEFS_OVRL = [-0.06766283, 1.11546468, 0.04602535]
_DNSMOS_SR = 16000
_DNSMOS_INPUT_LENGTH = 9.01
_DNSMOS_INPUT_SAMPLES = int(_DNSMOS_INPUT_LENGTH * _DNSMOS_SR)

_dnsmos_sessions = None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_model(path: Path, url: str, sha256: str):
    """Download a model file if not present, verify SHA256."""
    if path.exists():
        if _sha256(path) == sha256:
            return
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {path.name}...")
    urllib.request.urlretrieve(url, str(path))
    digest = _sha256(path)
    if digest != sha256:
        path.unlink()
        raise RuntimeError(f"{path.name}: SHA256 mismatch — expected {sha256}, got {digest}")


def _get_dnsmos_sessions():
    """Lazy-load DNSMOS ONNX sessions (cached)."""
    global _dnsmos_sessions
    if _dnsmos_sessions is not None:
        return _dnsmos_sessions
    import onnxruntime as ort
    for name, info in _DNSMOS_MODELS.items():
        _ensure_model(_DNSMOS_MODELS_DIR / name, info["url"], info["sha256"])
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    primary = ort.InferenceSession(
        str(_DNSMOS_MODELS_DIR / "sig_bak_ovr.onnx"), opts,
        providers=["CPUExecutionProvider"],
    )
    p808 = ort.InferenceSession(
        str(_DNSMOS_MODELS_DIR / "model_v8.onnx"), opts,
        providers=["CPUExecutionProvider"],
    )
    _dnsmos_sessions = (primary, p808)
    return _dnsmos_sessions


def _melspec(audio, sr, n_fft, hop_length, n_mels):
    """Normalized mel spectrogram: (power_to_db + 40) / 40, transposed to (T, n_mels)."""
    import librosa
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
    )
    return ((librosa.power_to_db(mel, ref=np.max) + 40) / 40).T


def compute_dnsmos(enhanced_wav):
    """Compute DNSMOS P.835 scores for a single waveform.

    Args:
        enhanced_wav: (N,) numpy array, 16 kHz float32
    Returns:
        dict with SIG, BAK, OVRL, P808_MOS, or None if onnxruntime unavailable
    """
    try:
        primary, p808 = _get_dnsmos_sessions()
    except ImportError:
        return None

    audio = enhanced_wav.astype(np.float32)
    if len(audio) < _DNSMOS_INPUT_SAMPLES:
        audio = np.pad(audio, (0, _DNSMOS_INPUT_SAMPLES - len(audio)))

    num_hops = max(1, int(np.floor(len(audio) / _DNSMOS_SR) - _DNSMOS_INPUT_LENGTH) + 1)
    sig_scores, bak_scores, ovrl_scores, p808_scores = [], [], [], []

    for idx in range(num_hops):
        start = idx * _DNSMOS_SR
        segment = audio[start:start + _DNSMOS_INPUT_SAMPLES]
        # Primary model
        out = primary.run(None, {"input_1": segment.reshape(1, -1)})[0][0]
        sig_scores.append(out[0])
        bak_scores.append(out[1])
        ovrl_scores.append(out[2])
        # P808 model (expects 900 time frames)
        mel = _melspec(segment, _DNSMOS_SR, n_fft=321, hop_length=160, n_mels=120).astype(np.float32)[:900]
        p808_out = p808.run(None, {"input_1": mel.reshape(1, *mel.shape)})[0][0]
        p808_scores.append(p808_out)

    return {
        "SIG": float(np.polyval(_DNSMOS_COEFS_SIG, np.mean(sig_scores))),
        "BAK": float(np.polyval(_DNSMOS_COEFS_BAK, np.mean(bak_scores))),
        "OVRL": float(np.polyval(_DNSMOS_COEFS_OVRL, np.mean(ovrl_scores))),
        "P808_MOS": float(np.mean(p808_scores)),
    }


# ── AECMOS ───────────────────────────────────────────────────────────────────

_AECMOS_MODELS_DIR = Path(".cache/aecmos/models")
_AECMOS_MODEL_URL = "https://github.com/microsoft/AEC-Challenge/raw/main/AECMOS/AECMOS_local/Run_1663829550_Stage_0.onnx"
_AECMOS_SR = 16000
_AECMOS_MAX_LEN = 20  # seconds

_aecmos_session = None


def _get_aecmos_session():
    """Lazy-load AECMOS ONNX session (cached)."""
    global _aecmos_session
    if _aecmos_session is not None:
        return _aecmos_session
    import onnxruntime as ort
    model_path = _AECMOS_MODELS_DIR / "Run_1663829550_Stage_0.onnx"
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {model_path.name}...")
        urllib.request.urlretrieve(_AECMOS_MODEL_URL, str(model_path))
        print(f"  -> {model_path} (SHA256: {_sha256(model_path)})")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    _aecmos_session = ort.InferenceSession(
        str(model_path), opts, providers=["CPUExecutionProvider"],
    )
    return _aecmos_session


def _aecmos_melspec(audio, sr=_AECMOS_SR):
    return _melspec(audio, sr, n_fft=513, hop_length=256, n_mels=160)


def compute_aecmos(lpb_wav, mic_wav, enhanced_wav):
    """Compute AECMOS scores for a single sample (16 kHz).

    Args:
        lpb_wav: (N,) numpy array, far-end loopback signal
        mic_wav: (N,) numpy array, microphone signal
        enhanced_wav: (N,) numpy array, model output
    Returns:
        dict with echo_mos, deg_mos (1-5 scale), or None if unavailable
    """
    try:
        session = _get_aecmos_session()
    except ImportError:
        return None

    # Align lengths and cap at max duration
    min_len = min(len(lpb_wav), len(mic_wav), len(enhanced_wav))
    max_samples = _AECMOS_MAX_LEN * _AECMOS_SR
    n = min(min_len, max_samples)
    lpb = lpb_wav[:n].astype(np.float32)
    mic = mic_wav[:n].astype(np.float32)
    enh = enhanced_wav[:n].astype(np.float32)

    # Compute mel spectrograms and stack
    lpb_mel = _aecmos_melspec(lpb)
    mic_mel = _aecmos_melspec(mic)
    enh_mel = _aecmos_melspec(enh)
    feats = np.stack([lpb_mel, mic_mel, enh_mel]).astype(np.float32)
    feats = np.expand_dims(feats, axis=0)  # (1, 3, T, 160)

    h0 = np.zeros((4, 1, 64), dtype=np.float32)
    input_name = session.get_inputs()[0].name
    result = session.run(None, {input_name: feats, "h0": h0})[0]

    return {
        "echo_mos": float(result[0]),
        "deg_mos": float(result[1]),
    }


# ── Combined evaluation ──────────────────────────────────────────────────────


def evaluate_sample(mic_wav, enhanced_wav, clean_wav, sr=16000, ref_wav=None):
    """Compute all metrics for a single sample.

    Args:
        mic_wav, enhanced_wav, clean_wav: (N,) numpy arrays
        ref_wav: (N,) numpy array, far-end reference (optional, enables AECMOS)

    Returns:
        dict of metric name -> value
    """
    results = {}

    erle_db, erle_frames = erle(mic_wav, enhanced_wav, clean_wav)
    results["erle_db"] = erle_db
    results["erle_frames"] = erle_frames

    results["seg_snr"] = segmental_snr(clean_wav, enhanced_wav)

    pesq_score = compute_pesq(clean_wav, enhanced_wav, sr)
    if pesq_score is not None:
        results["pesq"] = pesq_score

    stoi_score = compute_stoi(clean_wav, enhanced_wav, sr)
    if stoi_score is not None:
        results["stoi"] = stoi_score

    dnsmos = compute_dnsmos(enhanced_wav)
    if dnsmos is not None:
        results["dnsmos_sig"] = dnsmos["SIG"]
        results["dnsmos_bak"] = dnsmos["BAK"]
        results["dnsmos_ovrl"] = dnsmos["OVRL"]
        results["dnsmos_p808"] = dnsmos["P808_MOS"]

    if ref_wav is not None:
        aecmos = compute_aecmos(ref_wav, mic_wav, enhanced_wav)
        if aecmos is not None:
            results["echo_mos"] = aecmos["echo_mos"]
            results["deg_mos"] = aecmos["deg_mos"]

    return results
