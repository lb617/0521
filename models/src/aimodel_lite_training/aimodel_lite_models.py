#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
aimodel_lite_models.py

AIMODEL-friendly models for the old three-stage GrimACE pipeline.

1) LiteFrameQualityModel
   Input : [N, 1, 224, 224]
   Output: [N, 1, 1, 1], sigmoid score in 0~1

2) LiteFaceCropperModel
   Input : [N, 3, 256, 256]
   Output: [N, 5, 1, 1], sigmoid:
           [conf, x1, y1, x2, y2], normalized xyxy

3) LiteGrimaceScoreModel
   Input : [N, 3, 224, 224]
   Output: [N, 20, 1, 1], raw logits.
           CPU-side reshape to [N, 5, 4].

Only simple ops are used:
Conv, BatchNorm, ReLU, MaxPool, GlobalAveragePool, Sigmoid.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


ORGAN_COLUMNS = ["eye", "nose", "cheek", "ear", "whisker"]


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LiteBackbone(nn.Module):
    def __init__(
        self,
        in_ch: int,
        channels: Sequence[int],
        blocks_per_stage: int = 2,
    ):
        super().__init__()
        layers = []
        prev = int(in_ch)

        for stage_idx, ch in enumerate(channels):
            ch = int(ch)
            for _ in range(int(blocks_per_stage)):
                layers.append(ConvBNReLU(prev, ch, kernel_size=3, stride=1))
                prev = ch
            if stage_idx != len(channels) - 1:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.features = nn.Sequential(*layers)
        self.out_channels = int(channels[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class LiteFrameQualityModel(nn.Module):
    def __init__(
        self,
        in_ch: int = 1,
        channels: Sequence[int] = (16, 32, 64, 96),
        blocks_per_stage: int = 2,
    ):
        super().__init__()
        self.backbone = LiteBackbone(in_ch, channels, blocks_per_stage)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Conv2d(self.backbone.out_channels, 1, kernel_size=1, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.pool(x)
        x = self.head(x)
        return self.act(x)


class LiteFaceCropperModel(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        channels: Sequence[int] = (16, 32, 64, 96),
        blocks_per_stage: int = 2,
    ):
        super().__init__()
        self.backbone = LiteBackbone(in_ch, channels, blocks_per_stage)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Conv2d(self.backbone.out_channels, 5, kernel_size=1, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.pool(x)
        x = self.head(x)
        return self.act(x)


class LiteGrimaceScoreModel(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        channels: Sequence[int] = (24, 48, 96, 128),
        blocks_per_stage: int = 2,
        num_organs: int = 5,
        num_classes: int = 4,
    ):
        super().__init__()
        self.num_organs = int(num_organs)
        self.num_classes = int(num_classes)
        self.backbone = LiteBackbone(in_ch, channels, blocks_per_stage)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Conv2d(
            self.backbone.out_channels,
            self.num_organs * self.num_classes,
            kernel_size=1,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.pool(x)
        return self.head(x)


def build_model(model_name: str) -> nn.Module:
    name = str(model_name).lower().strip()
    if name in {"quality", "frame_quality", "fq"}:
        return LiteFrameQualityModel()
    if name in {"cropper", "face_cropper", "bbox", "fc"}:
        return LiteFaceCropperModel()
    if name in {"grimace", "grimace_scorer", "gs"}:
        return LiteGrimaceScoreModel()
    raise ValueError(f"Unknown model_name={model_name!r}. Choose quality, cropper, or grimace.")


def input_shape_for(model_name: str) -> tuple[int, int, int, int]:
    name = str(model_name).lower().strip()
    if name in {"quality", "frame_quality", "fq"}:
        return (1, 1, 224, 224)
    if name in {"cropper", "face_cropper", "bbox", "fc"}:
        return (1, 3, 256, 256)
    if name in {"grimace", "grimace_scorer", "gs"}:
        return (1, 3, 224, 224)
    raise ValueError(f"Unknown model_name={model_name!r}.")
