from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torch import nn
from torchvision.models.googlenet import GoogLeNetOutputs
from torchvision.models.inception import InceptionOutputs
from tqdm import tqdm


@dataclass
class TrainingResult:
    model_name: str
    mode: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_time_sec: float
    best_val_loss: float
    report_path: str
    confusion_matrix_path: str
    checkpoint_path: str


def _extract_logits(outputs: torch.Tensor | InceptionOutputs | GoogLeNetOutputs) -> torch.Tensor:
    if isinstance(outputs, (InceptionOutputs, GoogLeNetOutputs)):
        return outputs.logits
    return outputs


def _training_loss(
    outputs: torch.Tensor | InceptionOutputs | GoogLeNetOutputs,
    labels: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    if isinstance(outputs, InceptionOutputs):
        aux_loss = criterion(outputs.aux_logits, labels) if outputs.aux_logits is not None else 0.0
        return criterion(outputs.logits, labels) + 0.4 * aux_loss
    if isinstance(outputs, GoogLeNetOutputs):
        aux_loss = 0.0
        if outputs.aux_logits1 is not None:
            aux_loss = aux_loss + criterion(outputs.aux_logits1, labels)
        if outputs.aux_logits2 is not None:
            aux_loss = aux_loss + criterion(outputs.aux_logits2, labels)
        return criterion(outputs.logits, labels) + 0.3 * aux_loss
    return criterion(outputs, labels)


def train_model(
    model: nn.Module,
    model_name: str,
    mode: str,
    dataloaders: dict[str, torch.utils.data.DataLoader],
    class_names: list[str],
    epochs: int,
    learning_rate: float,
    device: torch.device,
    output_dir: str | Path,
) -> TrainingResult:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    best_model_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    start_time = perf_counter()

    model.to(device)
    for _epoch in range(epochs):
        model.train()
        for inputs, labels in tqdm(dataloaders["train"], desc=f"{model_name}-{mode}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = _training_loss(outputs, labels, criterion)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for inputs, labels in dataloaders["val"]:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = _training_loss(outputs, labels, criterion)
                val_losses.append(loss.item())
        epoch_val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_state)
    training_time = perf_counter() - start_time

    y_true: list[int] = []
    y_pred: list[int] = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloaders["test"]:
            inputs = inputs.to(device)
            logits = _extract_logits(model(inputs))
            predictions = torch.argmax(logits, dim=1).cpu().tolist()
            y_pred.extend(predictions)
            y_true.extend(labels.tolist())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True)
    matrix = confusion_matrix(y_true, y_pred).tolist()

    stem = f"{model_name}_{mode}"
    report_path = output_path / f"{stem}_classification_report.json"
    confusion_matrix_path = output_path / f"{stem}_confusion_matrix.json"
    checkpoint_path = output_path / f"{stem}.pt"

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    confusion_matrix_path.write_text(json.dumps({"labels": class_names, "matrix": matrix}, indent=2), encoding="utf-8")
    torch.save(
        {
            "model_name": model_name,
            "mode": mode,
            "class_names": class_names,
            "state_dict": model.state_dict(),
        },
        checkpoint_path,
    )

    result = TrainingResult(
        model_name=model_name,
        mode=mode,
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1_score=float(f1),
        training_time_sec=float(training_time),
        best_val_loss=float(best_val_loss),
        report_path=str(report_path),
        confusion_matrix_path=str(confusion_matrix_path),
        checkpoint_path=str(checkpoint_path),
    )
    _ = asdict(result)
    return result
