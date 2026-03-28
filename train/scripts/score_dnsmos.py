#!/usr/bin/env python3
"""Score clean speech files with DNSMOS P.835.

Runs Microsoft's DNSMOS P.835 model on all clean speech files and saves
per-file quality scores (SIG, BAK, OVRL, P808_MOS) to a JSON cache file.

The scores file integrates with AECDataset's quality filtering: set
data.dnsmos_ovrl_min in the training config to filter low-quality files.

ONNX models are auto-downloaded from the DNS-Challenge GitHub repo on first run.

Usage:
    python scripts/score_dnsmos.py
    python scripts/score_dnsmos.py --clean-dir /workspace/deepvqe/datasets_fullband/clean
    python scripts/score_dnsmos.py --limit 100
    python scripts/score_dnsmos.py --workers 8 --no-gpu
"""

import argparse
import hashlib
import json
import urllib.request
from multiprocessing import Pool
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

# ── File discovery (self-contained, no torch dependency) ─────────────────────

_CACHE_DIR = Path(".cache/file_lists")
_DNSMOS_DIR = Path(".cache/dnsmos")


def _dir_cache_key(directory: str) -> str:
    return hashlib.md5(str(Path(directory).resolve()).encode()).hexdigest()


def dnsmos_scores_path(clean_dir: str) -> Path:
    return _DNSMOS_DIR / f"{_dir_cache_key(clean_dir)}.json"


def collect_audio_files(directory):
    """Recursively collect .wav and .flac files, with disk caching."""
    if not directory:
        return []
    d = Path(directory)
    if not d.exists():
        return []
    abs_path = str(d.resolve())
    cache_file = _CACHE_DIR / f"{_dir_cache_key(directory)}.json"
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
            if cached.get("path") == abs_path:
                files = cached["files"]
                if files and Path(files[0]).exists() and Path(files[-1]).exists():
                    return files
        except (json.JSONDecodeError, KeyError):
            pass
    files = []
    for ext in ("*.wav", "*.flac"):
        files.extend(str(f) for f in d.rglob(ext))
    files.sort()
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps({
        "path": abs_path, "count": len(files), "files": files,
    }))
    return files

# ── Constants ────────────────────────────────────────────────────────────────

SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01  # seconds

# ONNX model URLs and SHA256 hashes from Microsoft DNS-Challenge repo
MODELS = {
    "sig_bak_ovr.onnx": {
        "url": "https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx",
        "sha256": "269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd",
    },
    "model_v8.onnx": {
        "url": "https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/model_v8.onnx",
        "sha256": "9246480c58567bc6affd4200938e77eef49468c8bc7ed3776d109c07456f6e91",
    },
}

# Polynomial fitting coefficients (non-personalized)
COEFS_SIG = [-0.08397278, 1.22083953, 0.0052439]
COEFS_BAK = [-0.13166888, 1.60915514, -0.39604546]
COEFS_OVRL = [-0.06766283, 1.11546468, 0.04602535]

INPUT_SAMPLES = int(INPUT_LENGTH * SAMPLING_RATE)

MODELS_DIR = Path(".cache/dnsmos/models")

# ── Model download ───────────────────────────────────────────────────────────


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_models(models_dir: Path):
    """Download ONNX models if not present, verify SHA256."""
    models_dir.mkdir(parents=True, exist_ok=True)
    for name, info in MODELS.items():
        path = models_dir / name
        if path.exists():
            digest = _sha256(path)
            if digest == info["sha256"]:
                continue
            print(f"WARNING: {name} hash mismatch (got {digest}), re-downloading")
            path.unlink()
        print(f"Downloading {name}...")
        urllib.request.urlretrieve(info["url"], str(path))
        digest = _sha256(path)
        if digest != info["sha256"]:
            path.unlink()
            raise RuntimeError(
                f"{name}: SHA256 mismatch — expected {info['sha256']}, got {digest}")
        print(f"  -> {path} (SHA256 verified)")


# ── Scoring logic ────────────────────────────────────────────────────────────


def audio_melspec(audio, sr=SAMPLING_RATE, n_mels=120, frame_size=320, hop_length=160):
    """Compute normalized mel spectrogram."""
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=frame_size + 1, hop_length=hop_length,
        n_mels=n_mels,
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return ((mel_spec_db + 40) / 40).T  # (time, n_mels)


def score_audio(audio, primary_session, p808_session):
    """Score a single audio array. Returns dict with SIG, BAK, OVRL, P808_MOS."""
    # Pad if too short
    if len(audio) < INPUT_SAMPLES:
        audio = np.pad(audio, (0, INPUT_SAMPLES - len(audio)))

    # Process overlapping windows (1-second hop)
    num_hops = int(np.floor(len(audio) / SAMPLING_RATE) - INPUT_LENGTH) + 1
    num_hops = max(1, num_hops)

    ovrl_scores, sig_scores, bak_scores, p808_scores = [], [], [], []

    for idx in range(num_hops):
        start = idx * SAMPLING_RATE
        end = start + INPUT_SAMPLES
        segment = audio[start:end]

        # Primary model (P.835: SIG, BAK, OVRL)
        input_features = segment.reshape(1, -1)
        onnx_out = primary_session.run(None, {"input_1": input_features})[0][0]
        sig_scores.append(onnx_out[0])
        bak_scores.append(onnx_out[1])
        ovrl_scores.append(onnx_out[2])

        # P808 model (expects exactly 900 time frames)
        mel = audio_melspec(segment).astype(np.float32)[:900]
        mel_input = mel.reshape(1, *mel.shape)
        p808_out = p808_session.run(None, {"input_1": mel_input})[0][0]
        p808_scores.append(p808_out)

    # Average across windows, apply polynomial fitting
    sig_raw = np.mean(sig_scores)
    bak_raw = np.mean(bak_scores)
    ovrl_raw = np.mean(ovrl_scores)
    p808_raw = np.mean(p808_scores)

    return {
        "SIG": round(float(np.polyval(COEFS_SIG, sig_raw)), 4),
        "BAK": round(float(np.polyval(COEFS_BAK, bak_raw)), 4),
        "OVRL": round(float(np.polyval(COEFS_OVRL, ovrl_raw)), 4),
        "P808_MOS": round(float(p808_raw), 4),
    }


def load_and_score(path, primary_session, p808_session):
    """Load audio file and score it."""
    audio, sr = sf.read(path, dtype="float32")
    if sr != SAMPLING_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLING_RATE)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return score_audio(audio, primary_session, p808_session)


# ── Multiprocessing worker ───────────────────────────────────────────────────

_worker_primary = None
_worker_p808 = None


def _worker_init(models_dir, providers):
    """Initialize ONNX sessions in each worker process."""
    global _worker_primary, _worker_p808
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    _worker_primary = ort.InferenceSession(
        str(models_dir / "sig_bak_ovr.onnx"), sess_options=opts, providers=providers)
    _worker_p808 = ort.InferenceSession(
        str(models_dir / "model_v8.onnx"), sess_options=opts, providers=providers)


def _worker_score(path):
    """Score a single file in the worker process."""
    try:
        return path, load_and_score(path, _worker_primary, _worker_p808)
    except Exception as e:
        return path, {"error": str(e)}


# ── Persistence ──────────────────────────────────────────────────────────────


def load_scores(path: Path) -> dict:
    """Load existing scores or return empty structure."""
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, KeyError):
            pass
    return {"dir": "", "version": 1, "scores": {}}


def save_scores(data: dict, path: Path):
    """Atomically save scores to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, separators=(",", ":")))
    tmp.rename(path)


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Score clean speech with DNSMOS P.835")
    parser.add_argument("--clean-dir", default="/workspace/deepvqe/datasets_fullband/clean",
                        help="Directory of clean speech files")
    parser.add_argument("--models-dir", default=str(MODELS_DIR),
                        help="Directory for ONNX models")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers (CPU mode)")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Force CPU-only mode")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit to first N files (0 = all)")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    ensure_models(models_dir)

    # Determine ONNX providers
    import onnxruntime as ort
    if not args.no_gpu and "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        use_gpu = True
        print("Using CUDA for ONNX inference")
    else:
        providers = ["CPUExecutionProvider"]
        use_gpu = False
        print("Using CPU for ONNX inference")

    # Collect files
    print(f"Scanning {args.clean_dir}...")
    all_files = collect_audio_files(args.clean_dir)
    print(f"Found {len(all_files)} audio files")

    # Load existing scores
    sp = dnsmos_scores_path(args.clean_dir)
    data = load_scores(sp)
    data["dir"] = args.clean_dir

    existing = set(data["scores"].keys())
    to_score = [f for f in all_files if f not in existing]

    if args.limit > 0:
        to_score = to_score[:args.limit]

    print(f"Already scored: {len(existing)}, remaining: {len(to_score)}")

    if not to_score:
        print("Nothing to do.")
        return

    save_interval = 10000

    if use_gpu:
        # GPU mode: single process, sequential inference
        primary = ort.InferenceSession(
            str(models_dir / "sig_bak_ovr.onnx"), providers=providers)
        p808 = ort.InferenceSession(
            str(models_dir / "model_v8.onnx"), providers=providers)

        for i, path in enumerate(tqdm(to_score, desc="Scoring (GPU)")):
            try:
                result = load_and_score(path, primary, p808)
            except Exception as e:
                result = {"error": str(e)}
            data["scores"][path] = result

            if (i + 1) % save_interval == 0:
                save_scores(data, sp)
    else:
        # CPU mode: multiprocessing
        with Pool(args.workers, initializer=_worker_init,
                  initargs=(models_dir, providers)) as pool:
            for i, (path, result) in enumerate(
                tqdm(pool.imap_unordered(_worker_score, to_score),
                     total=len(to_score), desc="Scoring (CPU)")):
                data["scores"][path] = result

                if (i + 1) % save_interval == 0:
                    save_scores(data, sp)

    save_scores(data, sp)

    # Summary
    scores = [s for s in data["scores"].values() if "error" not in s]
    errors = sum(1 for s in data["scores"].values() if "error" in s)
    if scores:
        ovrl_vals = [s["OVRL"] for s in scores]
        print(f"\nScored {len(scores)} files ({errors} errors)")
        print(f"OVRL: mean={np.mean(ovrl_vals):.2f}, "
              f"median={np.median(ovrl_vals):.2f}, "
              f"min={np.min(ovrl_vals):.2f}, max={np.max(ovrl_vals):.2f}")
        for threshold in [2.5, 3.0, 3.5, 4.0]:
            n = sum(1 for v in ovrl_vals if v >= threshold)
            print(f"  OVRL >= {threshold}: {n}/{len(ovrl_vals)} ({100*n/len(ovrl_vals):.1f}%)")
    print(f"\nScores saved to {sp}")


if __name__ == "__main__":
    main()
