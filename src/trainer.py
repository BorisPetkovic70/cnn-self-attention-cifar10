"""Training loop: optimizer and scheduler builders, train/eval epochs, history and checkpoints."""

from collections.abc import Iterable
import time
from itertools import islice
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torch.utils.data import DataLoader

from src.config import ExperimentConfig
from src.utils import AverageMeter, get_device, save_checkpoint


def build_optimizer(params: Iterable[nn.Parameter], cfg: ExperimentConfig) -> Optimizer:
    """Build the optimizer named by ``cfg.optimizer``; ``weight_decay`` is the L2 strength."""
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum,
                               weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(f"Unknown optimizer {cfg.optimizer!r}, expected 'sgd', 'adam' or 'adamw'")


def build_scheduler(optimizer: Optimizer, cfg: ExperimentConfig) -> LRScheduler | None:
    """Build the per-epoch LR scheduler named by ``cfg.scheduler``; ``"none"`` keeps the LR constant."""
    if cfg.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    if cfg.scheduler == "none":
        return None
    raise ValueError(f"Unknown scheduler {cfg.scheduler!r}, expected 'cosine' or 'none'")


class Trainer:
    """Train ``model`` on ``train_loader``, validate each epoch, keep the best checkpoint.

    Mixed precision is used only when ``cfg.amp`` is set and the device is CUDA.

    Attributes:
        device: Device the model and batches live on.
        best_val_acc: Best validation accuracy seen so far, in [0, 1].
        ckpt_path: Where the best-validation weights are saved.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: ExperimentConfig,
    ) -> None:
        self.cfg = cfg
        self.device = get_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
        # Frozen parameters are left out so the optimizer keeps no state for them
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = build_optimizer(trainable, cfg)
        self.scheduler = build_scheduler(self.optimizer, cfg)

        self.use_amp = cfg.amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.best_val_acc = float("-inf")
        self.ckpt_path = Path(cfg.out_dir) / "checkpoints" / f"{cfg.run_name}.pth"

    def _batches(self, loader: DataLoader) -> Iterable:
        """Iterate ``loader``, limited to ``cfg.fast_dev_run`` batches when that is > 0."""
        n = self.cfg.fast_dev_run
        return islice(loader, n) if n > 0 else loader

    def train_epoch(self) -> tuple[float, float]:
        """Run one training epoch and return its sample-weighted (loss, accuracy)."""
        self.model.train()
        loss_meter, acc_meter = AverageMeter(), AverageMeter()
        for x, y in self._batches(self.train_loader):
            x = x.to(self.device, non_blocking=True)  # (B, 3, 32, 32)
            y = y.to(self.device, non_blocking=True)  # (B,)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.use_amp):
                logits = self.model(x)  # (B, num_classes)
                loss = self.criterion(logits, y)
            # The scaler multiplies the loss so float16 gradients don't underflow,
            # then unscales them before the optimizer step
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Detached tensors keep the running sums on the device: no sync per batch
            n = y.size(0)
            loss_meter.update(loss.detach(), n)
            acc_meter.update((logits.argmax(dim=1) == y).float().mean(), n)
        return float(loss_meter.avg), float(acc_meter.avg)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple[float, float]:
        """Return the sample-weighted (loss, accuracy) of the model on ``loader`` in eval mode."""
        self.model.eval()
        loss_meter, acc_meter = AverageMeter(), AverageMeter()
        for x, y in self._batches(loader):
            x = x.to(self.device, non_blocking=True)  # (B, 3, 32, 32)
            y = y.to(self.device, non_blocking=True)  # (B,)
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.use_amp):
                logits = self.model(x)  # (B, num_classes)
                loss = self.criterion(logits, y)
            n = y.size(0)
            loss_meter.update(loss, n)
            acc_meter.update((logits.argmax(dim=1) == y).float().mean(), n)
        return float(loss_meter.avg), float(acc_meter.avg)

    def fit(self) -> dict[str, list[float]]:
        """Train for ``cfg.epochs`` epochs, saving the weights whenever validation accuracy improves.

        Returns:
            History with one value per epoch under ``train_loss``, ``train_acc``, ``val_loss``,
            ``val_acc``, ``lr`` (the LR used in that epoch) and ``epoch_time`` (seconds,
            training plus validation).
        """
        history: dict[str, list[float]] = {
            "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": [], "epoch_time": [],
        }
        for epoch in range(1, self.cfg.epochs + 1):
            # Read before scheduler.step(), so it is the LR this epoch trains with
            lr = self.optimizer.param_groups[0]["lr"]
            start = time.perf_counter()
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.evaluate(self.val_loader)
            epoch_time = time.perf_counter() - start
            if self.scheduler is not None:
                self.scheduler.step()

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["lr"].append(lr)
            history["epoch_time"].append(epoch_time)

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                save_checkpoint(self.ckpt_path, self.model, epoch=epoch, val_acc=val_acc)

            print(f"epoch {epoch:3d}/{self.cfg.epochs} | "
                  f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
                  f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
                  f"lr {lr:.5f} | {epoch_time:.1f}s")
        return history
