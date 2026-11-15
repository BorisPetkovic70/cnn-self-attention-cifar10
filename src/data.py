"""CIFAR-10 dataloaders with train-only augmentation and a fixed train/val split."""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.config import ExperimentConfig

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Per-channel statistics of the CIFAR-10 training images
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# Fixed so every run, whatever its cfg.seed, is validated on the same 5k images
SPLIT_SEED = 42


def get_dataloaders(cfg: ExperimentConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, validation and test loaders.

    The validation set is carved out of the 50k training images, never from test.
    Augmentation (random crop + horizontal flip) applies to the training subset only.
    """
    normalize = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    eval_tf = transforms.Compose([transforms.ToTensor(), normalize])
    if cfg.augment:
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_tf = eval_tf

    # Two views of the same 50k images: augmented for training, plain for validation
    train_full = datasets.CIFAR10(cfg.data_dir, train=True, download=True, transform=train_tf)
    val_full = datasets.CIFAR10(cfg.data_dir, train=True, download=False, transform=eval_tf)
    test_set = datasets.CIFAR10(cfg.data_dir, train=False, download=True, transform=eval_tf)

    n_val = int(len(train_full) * cfg.val_split)
    perm = torch.randperm(len(train_full), generator=torch.Generator().manual_seed(SPLIT_SEED))
    train_set = Subset(train_full, perm[n_val:].tolist())
    val_set = Subset(val_full, perm[:n_val].tolist())

    loader_kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
    )
    return (
        DataLoader(train_set, shuffle=True, **loader_kwargs),
        DataLoader(val_set, shuffle=False, **loader_kwargs),
        DataLoader(test_set, shuffle=False, **loader_kwargs),
    )
