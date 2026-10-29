# Module Interfaces

The contract between the two halves of the codebase. Every signature, attribute and file format
below is relied on by the other student's code, so it is **frozen**: changing one needs both
students' agreement, and the change to this file goes in the same feature branch as the code
change.

## Ownership

| Area | Files | Owner |
|---|---|---|
| Configuration, data, training | `config.py`, `data.py`, `utils.py`, `trainer.py`, `runner.py` | A |
| Backbone and baseline | `models/blocks.py`, `models/baseline.py` | A |
| Attention and hybrid | `models/attention.py`, `models/hybrid.py` | B |
| Model registry | `models/factory.py` (B adds the `"hybrid"` entry) | A + B |
| Evaluation and plots | `metrics.py`, `viz.py` | B |
| CLI | `scripts/run_experiment.py` | A |
| Tests, notebooks, docs | `tests/`, `notebooks/`, `docs/` | B |
| README | Goal / Setup / Structure / Reproduction | A |
| README | Results / Analysis / Conclusions | B |

## Conventions

- **Tensors:** images `(B, 3, 32, 32)` float32, normalized with `CIFAR10_MEAN` / `CIFAR10_STD`.
  Labels `(B,)` int64 in `0..9`. Model output: logits `(B, num_classes)`.
- **Accuracies** are fractions in `[0, 1]` everywhere (history, `RunResult`, JSON, CSV).
- **Stage names** are the entries of `STAGE_ORDER`: `stem`, `stage1`, `stage2`, `stage3`, `stage4`.
- **`attn_pos`** is `"none"` or a comma-separated list of `after_<stage>` values, e.g.
  `"after_stage1"` or `"after_stage1,after_stage3"`. An unknown stage raises `ValueError`.

---

## `src/config.py` (A)

```python
@dataclass
class ExperimentConfig:
    run_name: str = ""               # empty → filled as <model>[_<stages>]_s<seed>, e.g. hybrid_stage1-stage2_s42
    model: str = "baseline"          # "baseline" | "hybrid"
    attn_pos: str = "none"
    attn_reduction: int = 8          # Q/K bottleneck C // reduction
    dropout: float = 0.3
    epochs: int = 30
    batch_size: int = 128
    optimizer: str = "sgd"           # "sgd" | "adam" | "adamw"
    lr: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 5e-4
    scheduler: str = "cosine"        # "none" | "cosine" | "step"
    label_smoothing: float = 0.0
    augment: bool = True
    val_split: float = 0.1
    seed: int = 42
    num_workers: int = 2
    amp: bool = True
    fast_dev_run: int = 0            # >0 → only N batches per epoch
    data_dir: str = "./data"
    out_dir: str = "./results"

    def to_dict(self) -> dict
```

## `src/data.py` (A)

```python
CIFAR10_CLASSES: list[str]                 # index = label
CIFAR10_MEAN: tuple[float, float, float]   # for un-normalizing images in plots
CIFAR10_STD: tuple[float, float, float]

def get_dataloaders(cfg: ExperimentConfig) -> tuple[DataLoader, DataLoader, DataLoader]  # train, val, test
```

Train is 45k images (augmented), val 5k (fixed split, not augmented), test 10k.

## `src/utils.py` (A)

```python
def seed_everything(seed: int) -> None
def get_device() -> torch.device
def count_parameters(model: nn.Module, trainable_only: bool = True) -> int
```

## `src/models/blocks.py` (A)

```python
STAGE_ORDER: tuple[str, ...] = ("stem", "stage1", "stage2", "stage3", "stage4")
STAGE_CHANNELS: dict[str, int]       # output channels per stage, see table

def build_backbone() -> nn.ModuleDict                                     # keys = STAGE_ORDER
def build_head(num_classes: int = 10, dropout: float = 0.3) -> nn.Sequential  # GAP → Dropout → Linear(512, num_classes)
```

| Stage | Output |
|---|---|
| `stem` | `(B, 64, 32, 32)` |
| `stage1` | `(B, 64, 16, 16)` |
| `stage2` | `(B, 128, 8, 8)` |
| `stage3` | `(B, 256, 4, 4)` |
| `stage4` | `(B, 512, 4, 4)` |

## `src/models/baseline.py` (A)

```python
class BaselineVGG(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.3) -> None
    def forward(self, x: Tensor) -> Tensor   # (B, 3, 32, 32) → (B, num_classes)
    stages: nn.ModuleDict                    # build_backbone()
    head: nn.Sequential                      # build_head(num_classes, dropout)
```

## `src/models/attention.py` (B)

```python
class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 8) -> None
    def forward(self, x: Tensor) -> Tensor   # (B, C, H, W) → (B, C, H, W)
    gamma: nn.Parameter                      # scalar, initialized to 0 → block starts as identity
    store_attn: bool                         # default False; when True, forward caches last_attn
    last_attn: Tensor | None                 # (B, N, N), N = H*W, each row sums to 1
```

## `src/models/hybrid.py` (B)

```python
class AttentionVGG(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        attn_pos: str = "after_stage1",
        reduction: int = 8,
        dropout: float = 0.3,
    ) -> None
    def forward(self, x: Tensor) -> Tensor   # (B, 3, 32, 32) → (B, num_classes)
    stages: nn.ModuleDict                    # build_backbone(), identical to BaselineVGG
    attn: nn.ModuleDict                      # stage name → SpatialSelfAttention(STAGE_CHANNELS[name], reduction)
    head: nn.Sequential                      # build_head(num_classes, dropout)
```

`attn` is keyed by stage name (`"stage1"`), not by `attn_pos` value. The forward pass applies
`attn[name]` right after `stages[name]` for every stage in `attn`.

## `src/models/factory.py` (A + B)

```python
def build_model(cfg: ExperimentConfig) -> nn.Module
```

| `cfg.model` | Returns |
|---|---|
| `"baseline"` | `BaselineVGG(dropout=cfg.dropout)` |
| `"hybrid"` | `AttentionVGG(attn_pos=cfg.attn_pos, reduction=cfg.attn_reduction, dropout=cfg.dropout)` |
| anything else | raises `ValueError` |

## `src/trainer.py` (A)

```python
class Trainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, cfg: ExperimentConfig) -> None
    def fit(self) -> dict[str, list[float]]
```

`fit()` returns the history: one entry per epoch under the keys `train_loss`, `train_acc`,
`val_loss`, `val_acc`, `lr` (learning rate used in that epoch) and `epoch_time` (seconds).

## `src/runner.py` (A)

```python
@dataclass
class RunResult:
    config: ExperimentConfig
    history: dict[str, list[float]]   # Trainer.fit() output
    best_val_acc: float
    test_acc: float                   # on the best-validation checkpoint
    test_loss: float
    y_true: np.ndarray                # (10000,) int64, test labels
    y_pred: np.ndarray                # (10000,) int64, test predictions
    n_params: int                     # trainable parameters
    train_time_s: float
    model: nn.Module                  # best-validation weights loaded

def run_experiment(cfg: ExperimentConfig) -> RunResult
```

`run_experiment` seeds, trains, reloads the best-validation checkpoint, evaluates the test set
once and writes the result files below.

## `src/metrics.py` (B)

```python
def predict_all(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]  # y_true, y_pred, (N,) int64
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float
def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 10) -> np.ndarray  # (n_classes, n_classes), cm[true, pred]
def per_class_accuracy(cm: np.ndarray) -> np.ndarray  # (n_classes,)
```

## `src/viz.py` (B)

```python
def plot_history(histories: dict[str, dict], metric: str = "acc", save_path: str | None = None)  # run label → history; metric "acc" | "loss"
def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], normalize: bool = True, save_path: str | None = None)
def plot_attention_map(model: nn.Module, image: Tensor, stage: str = "after_stage1", save_path: str | None = None)  # image (3, 32, 32), normalized
def plot_comparison_bars(results_csv: str, save_path: str | None = None)
```

Each function draws one matplotlib figure and saves it to `save_path` when given.

---

## Result files (written by `runner.py`, read by `viz.py` and the notebooks)

**`results/logs/<run_name>.json`**, one file per run:

```json
{
  "config": {"run_name": "...", "model": "...", "...": "..."},
  "history": {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": [], "epoch_time": []},
  "best_val_acc": 0.0,
  "test_acc": 0.0,
  "test_loss": 0.0,
  "n_params": 0,
  "train_time_s": 0.0
}
```

**`results/results.csv`**, one appended row per run. Columns: every `ExperimentConfig` field in
declaration order, then `best_val_acc`, `test_acc`, `test_loss`, `n_params`, `train_time_s`.
The header is written when the file is created.

Checkpoints go to `results/checkpoints/<run_name>.pth` and are not committed.
