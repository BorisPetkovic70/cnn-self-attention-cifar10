"""Experiment configuration: the single source of truth for every run setting."""

from dataclasses import asdict, dataclass


@dataclass
class ExperimentConfig:
    """All settings for one training run.

    An empty ``run_name`` is filled in from the model, attention position and seed,
    e.g. ``hybrid_stage1-stage2_s42``.
    """

    run_name: str = ""
    # --- structure ---
    model: str = "baseline"          # "baseline" | "hybrid"
    attn_pos: str = "none"           # "none" | "after_stage1" | "after_stage1,after_stage2" | ...
    attn_reduction: int = 8          # Q/K bottleneck C//reduction
    dropout: float = 0.3
    # --- optimization ---
    epochs: int = 30
    batch_size: int = 128
    optimizer: str = "sgd"           # "sgd" | "adam" | "adamw"
    lr: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 5e-4       # L2 regularization
    scheduler: str = "cosine"        # "none" | "cosine" | "step"
    label_smoothing: float = 0.0
    # --- data ---
    augment: bool = True
    val_split: float = 0.1
    # --- infra ---
    seed: int = 42
    num_workers: int = 2
    amp: bool = True                 # mixed precision, CUDA only
    fast_dev_run: int = 0            # >0 → use only N batches per epoch
    data_dir: str = "./data"
    out_dir: str = "./results"

    def __post_init__(self) -> None:
        if not self.run_name:
            parts = [self.model]
            if self.attn_pos != "none":
                parts.append(self.attn_pos.replace("after_", "").replace(",", "-"))
            parts.append(f"s{self.seed}")
            self.run_name = "_".join(parts)

    def to_dict(self) -> dict:
        """Return the config as a plain dict for JSON logs and CSV rows."""
        return asdict(self)
