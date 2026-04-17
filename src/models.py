from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torchvision import models


class LeNet(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class ZFNetApprox(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((6, 6)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model: nn.Module
    input_size: int
    transfer_enabled: bool


def _freeze_backbone(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def _replace_classifier(model_name: str, model: nn.Module, num_classes: int) -> nn.Module:
    if model_name == "alexnet":
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
    elif model_name == "vgg16":
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
    elif model_name == "resnet50":
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name == "googlenet":
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        if hasattr(model, "aux1") and model.aux1 is not None:
            aux1_features = model.aux1.fc2.in_features
            model.aux1.fc2 = nn.Linear(aux1_features, num_classes)
        if hasattr(model, "aux2") and model.aux2 is not None:
            aux2_features = model.aux2.fc2.in_features
            model.aux2.fc2 = nn.Linear(aux2_features, num_classes)
    elif model_name == "inception_v3":
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        if hasattr(model, "AuxLogits") and model.AuxLogits is not None:
            aux_features = model.AuxLogits.fc.in_features
            model.AuxLogits.fc = nn.Linear(aux_features, num_classes)
    elif model_name == "mobilenet_v3_large":
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    elif model_name == "squeezenet1_1":
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
        model.num_classes = num_classes
    return model


def create_model(name: str, num_classes: int, transfer_learning: bool) -> ModelSpec:
    if name == "lenet":
        return ModelSpec(name=name, model=LeNet(num_classes), input_size=224, transfer_enabled=False)
    if name == "zfnet":
        return ModelSpec(name=name, model=ZFNetApprox(num_classes), input_size=224, transfer_enabled=False)

    constructors = {
        "alexnet": (models.alexnet, models.AlexNet_Weights.DEFAULT),
        "googlenet": (models.googlenet, models.GoogLeNet_Weights.DEFAULT),
        "vgg16": (models.vgg16, models.VGG16_Weights.DEFAULT),
        "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
        "inception_v3": (models.inception_v3, models.Inception_V3_Weights.DEFAULT),
        "mobilenet_v3_large": (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.DEFAULT),
        "squeezenet1_1": (models.squeezenet1_1, models.SqueezeNet1_1_Weights.DEFAULT),
    }

    if name not in constructors:
        msg = f"Unsupported model: {name}"
        raise ValueError(msg)

    constructor, default_weights = constructors[name]
    weights = default_weights if transfer_learning else None
    kwargs = {"weights": weights}
    if name in {"googlenet", "inception_v3"}:
        kwargs["aux_logits"] = True
    model = constructor(**kwargs)
    if transfer_learning:
        _freeze_backbone(model)
    model = _replace_classifier(name, model, num_classes)
    return ModelSpec(
        name=name,
        model=model,
        input_size=299 if name == "inception_v3" else 224,
        transfer_enabled=True,
    )
