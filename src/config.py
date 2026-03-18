from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import yaml


@dataclass
class ModelConfig:
    mic_channels: List[int] = field(default_factory=lambda: [2, 64, 128, 128, 128, 128])
    far_channels: List[int] = field(default_factory=lambda: [2, 32, 128])
    align_hidden: int = 32
    dmax: int = 32
    align_temp_start: float = 1.0
    align_temp_end: float = 0.1
    align_temp_epochs: int = 20
    power_law_c: float = 0.3


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    n_fft: int = 512
    hop_length: int = 256
    n_freqs: int = 257


@dataclass
class TrainingConfig:
    # NOTE: YAML keys must match fields here — add new fields to BOTH places.
    batch_size: int = 8
    grad_accum_steps: int = 12
    num_workers: int = 4
    lr: float = 1.2e-3
    weight_decay: float = 5e-7
    epochs: int = 250
    clip_length_sec: float = 3.0
    amp: bool = True
    grad_clip: float = 5.0
    warmup_epochs: int = 5
    checkpoint_every: int = 5
    keep_checkpoints: int = 5
    early_stop_patience: int = 10
    early_stop_min_delta: float = 1e-3
    delay_acc_min: float = 0.7
    erle_min_db: float = 3.0
    lr_scheduler: str = "plateau"  # "plateau" or "cosine_restarts"
    lr_patience: int = 5
    lr_factor: float = 0.5
    lr_min: float = 1e-6
    lr_cosine_t0: int = 100
    lr_cosine_tmult: int = 2


@dataclass
class LossConfig:
    plcmse_weight: float = 1.0
    mag_l1_weight: float = 0.5
    time_l1_weight: float = 0.5
    sisdr_weight: float = 0.5
    smooth_l1_weight: float = 0.0
    smooth_l1_beta: float = 1.0
    energy_preservation_weight: float = 0.0
    energy_pres_mode: str = "relative"  # "absolute" or "relative"
    delay_weight: float = 1.0
    entropy_weight: float = 0.01
    mask_reg_weight: float = 0.1
    power_law_c: float = 0.5


@dataclass
class DataConfig:
    clean_dir: str = ""
    noise_dir: str = ""
    rir_dir: str = ""
    farend_dir: str = ""
    snr_range: Tuple[float, float] = (5, 40)
    ser_range: Tuple[float, float] = (-10, 10)
    delay_range: Tuple[float, float] = (0, 320)
    max_delay_frames: int = 32
    single_talk_prob: float = 0.2
    max_rir_length_ms: float = 500.0
    drr_range: Tuple[float, float] = (0, 20)  # DRR dB, uniform
    num_train: int = 10000  # only used for DummyAECDataset
    num_val: int = 1000
    # FixedSynthDataset settings (used with --overfit-real)
    overfit_delays_ms: List[float] = field(default_factory=lambda: [0, 40, 80, 120, 160, 200, 240, 300])
    overfit_snr_db: float = 20.0
    overfit_ser_db: float = 0.0
    overfit_repeat: int = 1


@dataclass
class EvalConfig:
    pesq_subset: int = 200
    audio_samples: int = 10


@dataclass
class PathsConfig:
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"


@dataclass
class HubConfig:
    push_to_hub: bool = False
    hub_model_id: str = ""           # e.g. "richdrummer33/deepvqe-aec"
    push_logs_every: int = 5         # upload TB logs every N epochs
    push_checkpoints: bool = True


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    hub: HubConfig = field(default_factory=HubConfig)


def _apply_dict(dc, d):
    """Recursively apply dict values to a dataclass instance."""
    for k, v in d.items():
        if not hasattr(dc, k):
            raise ValueError(f"Unknown config key: {k}")
        current = getattr(dc, k)
        if hasattr(current, "__dataclass_fields__") and isinstance(v, dict):
            _apply_dict(current, v)
        else:
            # Convert lists to tuples for tuple-typed fields
            if isinstance(v, list) and isinstance(current, tuple):
                v = tuple(v)
            setattr(dc, k, v)


def load_config(path: str | Path = "configs/default.yaml") -> Config:
    cfg = Config()
    with open(path) as f:
        d = yaml.safe_load(f)
    if d:
        _apply_dict(cfg, d)
    return cfg


if __name__ == "__main__":
    cfg = load_config()
    print(cfg)
