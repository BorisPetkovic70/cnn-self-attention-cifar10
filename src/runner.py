"""One call per experiment: train, test on the best-validation checkpoint, write the result files."""

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch import nn

from src.config import ExperimentConfig
from src.data import get_dataloaders
from src.metrics import predict_all
from src.models.factory import build_model
from src.trainer import Trainer
from src.utils import count_parameters, load_checkpoint, seed_everything

RESULT_COLUMNS = ("best_val_acc", "test_acc", "test_loss", "n_params", "train_time_s")


@dataclass
class RunResult:
    """Everything one run produced, handed to the metrics and plotting code."""

    config: ExperimentConfig
    history: dict[str, list[float]]  # Trainer.fit() output
    best_val_acc: float
    test_acc: float                  # on the best-validation checkpoint
    test_loss: float
    y_true: np.ndarray               # (10000,) int64, test labels
    y_pred: np.ndarray               # (10000,) int64, test predictions
    n_params: int                    # trainable parameters
    train_time_s: float
    model: nn.Module                 # best-validation weights loaded


def run_experiment(cfg: ExperimentConfig) -> RunResult:
    """Seed, train, reload the best-validation checkpoint, evaluate the test set once and log.

    Writes ``<out_dir>/logs/<run_name>.json`` and appends one row to ``<out_dir>/results.csv``.

    Raises:
        FileExistsError: If a log for ``cfg.run_name`` already exists. Checked before training,
            so a sweep never overwrites an earlier run.
    """
    out_dir = Path(cfg.out_dir)
    log_path = out_dir / "logs" / f"{cfg.run_name}.json"
    if log_path.exists():
        raise FileExistsError(f"{log_path} already exists; pass a new run_name or delete the log")

    seed_everything(cfg.seed)
    train_loader, val_loader, test_loader = get_dataloaders(cfg)
    model = build_model(cfg)
    n_params = count_parameters(model)

    trainer = Trainer(model, train_loader, val_loader, cfg)
    start = time.perf_counter()
    history = trainer.fit()
    train_time_s = time.perf_counter() - start

    # Test only the best-validation weights, never the last-epoch ones
    load_checkpoint(trainer.ckpt_path, model, map_location=trainer.device)
    test_loss, test_acc = trainer.evaluate(test_loader)
    y_true, y_pred = predict_all(model, test_loader, trainer.device)

    result = RunResult(
        config=cfg,
        history=history,
        best_val_acc=trainer.best_val_acc,
        test_acc=test_acc,
        test_loss=test_loss,
        y_true=y_true,
        y_pred=y_pred,
        n_params=n_params,
        train_time_s=train_time_s,
        model=model,
    )
    _write_log(result, log_path)
    _append_csv(result, out_dir / "results.csv")
    return result


def _write_log(result: RunResult, path: Path) -> None:
    """Write the config, history and headline numbers of one run as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    log = {
        "config": result.config.to_dict(),
        "history": result.history,
        **{col: getattr(result, col) for col in RESULT_COLUMNS},
    }
    path.write_text(json.dumps(log, indent=2))


def _append_csv(result: RunResult, path: Path) -> None:
    """Append one row (config fields, then ``RESULT_COLUMNS``), writing the header for a new file."""
    row = {**result.config.to_dict(), **{col: getattr(result, col) for col in RESULT_COLUMNS}}
    is_new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if is_new:
            writer.writeheader()
        writer.writerow(row)
