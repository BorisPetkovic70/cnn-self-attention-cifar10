"""Model registry: maps ``cfg.model`` to a constructed model."""

from collections.abc import Callable

from torch import nn

from src.config import ExperimentConfig
from src.models.baseline import BaselineVGG

# Model name -> builder taking the config; each entry passes only the settings its model uses
MODEL_BUILDERS: dict[str, Callable[[ExperimentConfig], nn.Module]] = {
    "baseline": lambda cfg: BaselineVGG(dropout=cfg.dropout),
}


def build_model(cfg: ExperimentConfig) -> nn.Module:
    """Build the model named by ``cfg.model``.

    Raises:
        ValueError: If ``cfg.model`` is not a key of ``MODEL_BUILDERS``.
    """
    if cfg.model not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model {cfg.model!r}, expected one of {sorted(MODEL_BUILDERS)}")
    return MODEL_BUILDERS[cfg.model](cfg)
