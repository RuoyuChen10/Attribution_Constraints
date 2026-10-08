"""Shared Hugging Face ViT-B/16 definition used by all ViT baselines."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import ViTForImageClassification


VIT_MODEL_ID = "google/vit-base-patch16-224"
VIT_IMAGE_MEAN = (0.5, 0.5, 0.5)
VIT_IMAGE_STD = (0.5, 0.5, 0.5)


def build_vit_classifier(num_classes: int, freeze_backbone: bool = False) -> ViTForImageClassification:
    model = ViTForImageClassification.from_pretrained(
        VIT_MODEL_ID,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    if freeze_backbone:
        for parameter in model.vit.parameters():
            parameter.requires_grad = False
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
    return model


def vit_logits(model: nn.Module, pixel_values: torch.Tensor) -> torch.Tensor:
    """Return logits for a bare model or a DistributedDataParallel wrapper."""
    return model(pixel_values=pixel_values).logits


class LogitsOnlyViT(nn.Module):
    """Adapt Hugging Face output objects to black-box explainers expecting tensors."""

    def __init__(self, model: ViTForImageClassification):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixel_values).logits
