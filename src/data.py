from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms


@dataclass
class DatasetBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_names: list[str]


class TransformSubset(Dataset):
    def __init__(self, base_dataset: datasets.ImageFolder, indices: list[int], transform: transforms.Compose) -> None:
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform
        self.classes = base_dataset.classes
        self.class_to_idx = base_dataset.class_to_idx

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image, label = self.base_dataset.samples[self.indices[idx]]
        sample = self.base_dataset.loader(image)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label


def _limit_classes(dataset: datasets.ImageFolder, max_classes: int | None) -> Dataset | Subset:
    if max_classes is None or max_classes <= 0 or max_classes >= len(dataset.classes):
        return dataset

    selected_classes = set(range(max_classes))
    indices = [idx for idx, (_, label) in enumerate(dataset.samples) if label in selected_classes]
    limited = Subset(dataset, indices)
    limited.classes = dataset.classes[:max_classes]  # type: ignore[attr-defined]
    limited.class_to_idx = {name: idx for idx, name in enumerate(limited.classes)}  # type: ignore[attr-defined]
    return limited


def _limit_samples_per_class(dataset: datasets.ImageFolder | Subset, max_samples_per_class: int | None) -> Dataset | Subset:
    if max_samples_per_class is None or max_samples_per_class <= 0:
        return dataset

    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        candidate_indices = dataset.indices
        classes = dataset.classes
    else:
        base_dataset = dataset
        candidate_indices = list(range(len(dataset.samples)))
        classes = dataset.classes

    per_class_counts: dict[int, int] = {}
    kept_indices: list[int] = []
    for sample_idx in candidate_indices:
        _path, label = base_dataset.samples[sample_idx]
        current = per_class_counts.get(label, 0)
        if current >= max_samples_per_class:
            continue
        per_class_counts[label] = current + 1
        kept_indices.append(sample_idx)

    limited = Subset(base_dataset, kept_indices)
    limited.classes = classes  # type: ignore[attr-defined]
    limited.class_to_idx = {name: idx for idx, name in enumerate(classes)}  # type: ignore[attr-defined]
    return limited


def build_transforms(image_size: int, inception: bool = False) -> tuple[transforms.Compose, transforms.Compose]:
    size = 299 if inception else image_size
    train_tfms = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_tfms, eval_tfms


def _resolve_split_dir(data_dir: Path, *candidates: str) -> Path | None:
    for candidate in candidates:
        split_dir = data_dir / candidate
        if split_dir.exists():
            return split_dir
    return None


def _train_val_lengths(total_size: int, val_fraction: float) -> tuple[int, int]:
    if total_size < 2:
        msg = "Training data must contain at least 2 images to create train/validation splits."
        raise ValueError(msg)

    val_size = max(1, int(total_size * val_fraction))
    if val_size >= total_size:
        val_size = total_size - 1
    train_size = total_size - val_size
    return train_size, val_size


def _train_val_test_lengths(total_size: int, train_split: float, val_split: float) -> tuple[int, int, int]:
    if total_size < 3:
        msg = "Dataset must contain at least 3 images to create train/validation/test splits."
        raise ValueError(msg)

    train_size = max(1, int(total_size * train_split))
    val_size = max(1, int(total_size * val_split))
    test_size = total_size - train_size - val_size

    if test_size <= 0:
        test_size = 1
        if train_size >= val_size and train_size > 1:
            train_size -= 1
        elif val_size > 1:
            val_size -= 1
        else:
            msg = "Dataset is too small for the configured train/validation/test splits."
            raise ValueError(msg)

    return train_size, val_size, test_size


def build_dataloaders(
    data_dir: str | Path,
    image_size: int,
    batch_size: int,
    train_split: float,
    val_split: float,
    test_split: float,
    seed: int,
    num_workers: int,
    inception: bool = False,
    max_classes: int | None = None,
    max_samples_per_class: int | None = None,
) -> DatasetBundle:
    data_dir = Path(data_dir)
    train_tfms, eval_tfms = build_transforms(image_size=image_size, inception=inception)
    train_dir = _resolve_split_dir(data_dir, "Train", "train", "Training", "training")
    test_dir = _resolve_split_dir(data_dir, "Test", "test", "Testing", "testing")

    if train_dir is not None and test_dir is not None:
        full_train = datasets.ImageFolder(root=str(train_dir))
        full_train = _limit_classes(full_train, max_classes)
        full_train = _limit_samples_per_class(full_train, max_samples_per_class)
        class_names = full_train.classes if hasattr(full_train, "classes") else []

        total_train = len(full_train)
        train_size, val_size = _train_val_lengths(total_train, val_split)
        generator = torch.Generator().manual_seed(seed)
        train_split_subset, val_split_subset = random_split(full_train, [train_size, val_size], generator=generator)

        if isinstance(full_train, Subset):
            base_dataset = full_train.dataset
        else:
            base_dataset = full_train

        train_indices = [full_train.indices[i] for i in train_split_subset.indices] if isinstance(full_train, Subset) else train_split_subset.indices
        val_indices = [full_train.indices[i] for i in val_split_subset.indices] if isinstance(full_train, Subset) else val_split_subset.indices

        train_dataset = TransformSubset(base_dataset, train_indices, train_tfms)
        val_dataset = TransformSubset(base_dataset, val_indices, eval_tfms)

        raw_test = datasets.ImageFolder(root=str(test_dir))
        raw_test = _limit_classes(raw_test, max_classes)
        raw_test = _limit_samples_per_class(raw_test, max_samples_per_class)
        if isinstance(raw_test, Subset):
            test_dataset = TransformSubset(raw_test.dataset, raw_test.indices, eval_tfms)
        else:
            test_dataset = datasets.ImageFolder(root=str(test_dir), transform=eval_tfms)
            test_dataset.classes = class_names
    else:
        if round(train_split + val_split + test_split, 5) != 1.0:
            msg = "train_split + val_split + test_split must equal 1.0"
            raise ValueError(msg)

        base_dataset = datasets.ImageFolder(root=str(data_dir))
        base_dataset = _limit_classes(base_dataset, max_classes)
        base_dataset = _limit_samples_per_class(base_dataset, max_samples_per_class)
        class_names = base_dataset.classes if hasattr(base_dataset, "classes") else []

        total_size = len(base_dataset)
        train_size, val_size, test_size = _train_val_test_lengths(total_size, train_split, val_split)

        generator = torch.Generator().manual_seed(seed)
        train_subset, val_subset, test_subset = random_split(
            base_dataset,
            [train_size, val_size, test_size],
            generator=generator,
        )

        if isinstance(base_dataset, Subset):
            base_ref = base_dataset.dataset
            train_indices = [base_dataset.indices[i] for i in train_subset.indices]
            val_indices = [base_dataset.indices[i] for i in val_subset.indices]
            test_indices = [base_dataset.indices[i] for i in test_subset.indices]
        else:
            base_ref = base_dataset
            train_indices = train_subset.indices
            val_indices = val_subset.indices
            test_indices = test_subset.indices

        train_dataset = TransformSubset(base_ref, train_indices, train_tfms)
        val_dataset = TransformSubset(base_ref, val_indices, eval_tfms)
        test_dataset = TransformSubset(base_ref, test_indices, eval_tfms)

    loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    return DatasetBundle(
        train_loader=DataLoader(train_dataset, shuffle=True, **loader_args),
        val_loader=DataLoader(val_dataset, shuffle=False, **loader_args),
        test_loader=DataLoader(test_dataset, shuffle=False, **loader_args),
        class_names=class_names,
    )
