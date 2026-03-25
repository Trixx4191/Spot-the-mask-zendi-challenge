"""
models/model.py
───────────────
Model factory using timm.
Supports EfficientNet-B4/B7, ConvNeXt, ViT, and any other timm model.
All models output a single logit (binary classification).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import timm

from src.utils.common import get_logger

logger = get_logger("model")


class MaskClassifier(nn.Module):
    """
    Thin wrapper around a timm backbone.
    Adds a dropout + linear head for binary classification.
    Output: single raw logit (apply sigmoid for probability).
    """

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,          # remove default head
            global_pool="avg",
        )
        in_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1),
        )
        logger.info(f"Built {model_name} | features={in_features} | dropout={dropout}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features).squeeze(1)    # (B,)


def build_model(model_name: str, pretrained: bool = True, dropout: float = 0.4) -> MaskClassifier:
    """Convenience factory used by training and inference scripts."""
    return MaskClassifier(model_name=model_name, pretrained=pretrained, dropout=dropout)
