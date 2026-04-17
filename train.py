from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from src.data import build_dataloaders
from src.models import create_model
from src.trainer import train_model
from src.utils import load_config, save_summary, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and compare multiple CNN models.")
    parser.add_argument("--data-dir", required=True, help="Path to the dataset root.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--max-classes", type=int, default=None, help="Limit number of classes for faster experiments.")
    parser.add_argument("--max-samples-per-class", type=int, default=None, help="Limit images per class for faster experiments.")
    parser.add_argument(
        "--mode",
        choices=["scratch", "transfer", "both"],
        default="both",
        help="Training mode for supported backbones.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["seed"])

    results_dir = Path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size or config["batch_size"]
    epochs = args.epochs or config["epochs"]
    max_classes = args.max_classes if args.max_classes is not None else config.get("max_classes")
    max_samples_per_class = (
        args.max_samples_per_class if args.max_samples_per_class is not None else config.get("max_samples_per_class")
    )
    model_names: list[str] = config["models"]

    requested_modes = ["scratch", "transfer"] if args.mode == "both" else [args.mode]
    all_results: list[dict] = []

    for model_name in model_names:
        for mode in requested_modes:
            use_transfer = mode == "transfer"
            spec = create_model(model_name, num_classes=1, transfer_learning=use_transfer)
            if use_transfer and not spec.transfer_enabled:
                continue

            dataloaders = build_dataloaders(
                data_dir=args.data_dir,
                image_size=spec.input_size,
                batch_size=batch_size,
                train_split=config["train_split"],
                val_split=config["val_split"],
                test_split=config["test_split"],
                seed=config["seed"],
                num_workers=config["num_workers"],
                inception=model_name == "inception_v3",
                max_classes=max_classes,
                max_samples_per_class=max_samples_per_class,
            )
            spec = create_model(
                model_name,
                num_classes=len(dataloaders.class_names),
                transfer_learning=use_transfer,
            )

            output_dir = results_dir / model_name / mode
            result = train_model(
                model=spec.model,
                model_name=model_name,
                mode=mode,
                dataloaders={
                    "train": dataloaders.train_loader,
                    "val": dataloaders.val_loader,
                    "test": dataloaders.test_loader,
                },
                class_names=dataloaders.class_names,
                epochs=epochs,
                learning_rate=config["learning_rate"],
                device=device,
                output_dir=output_dir,
            )
            all_results.append(asdict(result))

    save_summary(all_results, results_dir / "summary.csv")


if __name__ == "__main__":
    main()
