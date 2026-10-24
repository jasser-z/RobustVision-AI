from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

from robustvision.training.reproducibility import worker_init_fn

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.247, 0.243, 0.261)


def cifar_transform():
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def cifar_loaders(
    data_dir: str | Path,
    batch_size: int,
    seed: int,
    validation_fraction: float = 0.1,
    quick_samples: int | None = None,
    download: bool = False,
    workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """Create deterministic train/validation split strictly from CIFAR-10 train data."""
    root = Path(data_dir)
    train_all = datasets.CIFAR10(root, train=True, download=download, transform=cifar_transform())
    test = datasets.CIFAR10(root, train=False, download=download, transform=cifar_transform())
    if quick_samples:
        limit = min(quick_samples, len(train_all))
        indices = torch.randperm(len(train_all), generator=torch.Generator().manual_seed(seed))[:limit]
        train_all = Subset(train_all, indices.tolist())
        test = Subset(test, list(range(min(quick_samples, len(test)))))
    validation_size = max(1, int(len(train_all) * validation_fraction))
    train_size = len(train_all) - validation_size
    train, validation = random_split(
        train_all, [train_size, validation_size], generator=torch.Generator().manual_seed(seed)
    )
    generator = torch.Generator().manual_seed(seed)
    kwargs = {"num_workers": workers, "worker_init_fn": worker_init_fn, "generator": generator}
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, **kwargs),
        DataLoader(validation, batch_size=batch_size, **kwargs),
        DataLoader(test, batch_size=batch_size, **kwargs),
        {"train": len(train), "validation": len(validation), "test": len(test), "seed": seed},
    )
