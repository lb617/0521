#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
aimodel_lite_models.py

Three AIMODEL-friendly PyTorch models for the old three-stage GrimACE pipeline:

1) LiteFrameQualityModel
   Input : [N, 1, 224, 224]
   Output: [N, 1, 1, 1], sigmoid score in 0~1

2) LiteFaceCropperModel
   Input : [N, 3, 256, 256]
   Output: [N, 5, 1, 1], sigmoid values:
           [conf, x1, y1, x2, y2], normalized to 0~1

3) LiteGrimaceScoreModel
   Input : [N, 3, 224, 224]
   Output: [N, 20, 1, 1], raw logits:
           reshape to [N, 5, 4] on CPU side.
           Organ order: eye, nose, cheek, ear, whisker.
           Class order: 0, 1, 2, 3.

Design principle:
- Use only simple deployment-friendly ops:
  Conv, BatchNorm, ReLU, MaxPool, GlobalAveragePool, Sigmoid.
- Do NOT put crop, softmax, argmax, sorting, thresholding, or averaging inside ONNX.
  Keep those in the board-side pipeline code.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn


ORGAN_COLUMNS = ["eye", "nose", "cheek", "ear", "whisker"]


class ConvBNReLU(nn.Module):
    """Deployment-friendly Conv -> BatchNorm -> ReLU block."""

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
    """
    Small CNN backbone built only from ConvBNReLU and MaxPool.
    AdaptiveAvgPool2d((1, 1)) is exported as GlobalAveragePool.
    """

    def __init__(
        self,
        in_ch: int,
        channels: Sequence[int] = (16, 32, 64, 96),
        blocks_per_stage: int = 2,
    ):
        super().__init__()
        layers = []
        prev = int(in_ch)

        for stage_idx, ch in enumerate(channels):
            ch = int(ch)
            for block_idx in range(int(blocks_per_stage)):
                layers.append(ConvBNReLU(prev, ch, kernel_size=3, stride=1))
                prev = ch

            # Keep spatial downsampling explicit and simple.
            if stage_idx != len(channels) - 1:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.features = nn.Sequential(*layers)
        self.out_channels = int(channels[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class LiteFrameQualityModel(nn.Module):
    """
    Replacement for the old FrameQualityModel.

    Input:
        [N, 1, 224, 224]
    Output:
        [N, 1, 1, 1], sigmoid normalized quality score in 0~1.

    Board-side postprocess:
        quality_score_0_to_6 = output[0] * 6.0
        quality_pass = quality_score_0_to_6 >= threshold
    """

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
        x = self.act(x)
        return x


class LiteFaceCropperModel(nn.Module):
    """
    Replacement for YOLO face cropper when AIMODEL conversion is the priority.

    Input:
        [N, 3, 256, 256]
    Output:
        [N, 5, 1, 1], all sigmoid values in 0~1:
        channel 0: confidence
        channel 1: x1 normalized
        channel 2: y1 normalized
        channel 3: x2 normalized
        channel 4: y2 normalized

    Board-side postprocess should:
    - squeeze to [5]
    - check conf threshold
    - sort x1/x2 and y1/y2 if needed
    - convert normalized xyxy to pixel xyxy
    - square bbox, pad, crop
    """

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
        x = self.act(x)
        return x


class LiteGrimaceScoreModel(nn.Module):
    """
    Replacement for the old MobileNetV3 GrimaceScoreModel.

    Input:
        [N, 3, 224, 224]
    Output:
        [N, 20, 1, 1], raw logits.

    Board-side postprocess:
        logits = output.reshape(5, 4)
        pred = argmax(logits, axis=1)

    Organ order:
        eye, nose, cheek, ear, whisker.
    """

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
        x = self.head(x)
        return x


def build_model(model_name: str) -> nn.Module:
    """Factory used by export_aimodel_onnx.py."""
    name = str(model_name).lower().strip()

    if name in {"quality", "frame_quality", "fq"}:
        return LiteFrameQualityModel()

    if name in {"cropper", "face_cropper", "bbox", "fc"}:
        return LiteFaceCropperModel()

    if name in {"grimace", "grimace_scorer", "gs"}:
        return LiteGrimaceScoreModel()

    raise ValueError(f"Unknown model_name={model_name!r}. Choose quality, cropper, or grimace.")


def input_shape_for(model_name: str) -> tuple[int, int, int, int]:
    """Static NCHW input shape. Keep N=1 for the official hardware pipeline."""
    name = str(model_name).lower().strip()

    if name in {"quality", "frame_quality", "fq"}:
        return (1, 1, 224, 224)

    if name in {"cropper", "face_cropper", "bbox", "fc"}:
        return (1, 3, 256, 256)

    if name in {"grimace", "grimace_scorer", "gs"}:
        return (1, 3, 224, 224)

    raise ValueError(f"Unknown model_name={model_name!r}.")


def decode_cropper_output(output: np.ndarray, confidence_threshold: float = 0.5) -> np.ndarray | None:
    """
    CPU-side helper. Not part of ONNX.

    Args:
        output: model output with shape [1, 5, 1, 1], [5, 1, 1], or [5].
    Returns:
        None if confidence is below threshold; otherwise:
        np.array([conf, x1, y1, x2, y2], dtype=float32), normalized xyxy.
    """
    arr = np.asarray(output, dtype=np.float32).reshape(-1)
    if arr.size != 5:
        raise ValueError(f"Cropper output should contain 5 values, got shape={np.asarray(output).shape}")

    conf, x1, y1, x2, y2 = arr.tolist()
    if conf < float(confidence_threshold):
        return None

    xa, xb = sorted([float(x1), float(x2)])
    ya, yb = sorted([float(y1), float(y2)])

    bbox = np.array(
        [
            float(np.clip(conf, 0.0, 1.0)),
            float(np.clip(xa, 0.0, 1.0)),
            float(np.clip(ya, 0.0, 1.0)),
            float(np.clip(xb, 0.0, 1.0)),
            float(np.clip(yb, 0.0, 1.0)),
        ],
        dtype=np.float32,
    )
    return bbox


def decode_grimace_logits(output: np.ndarray) -> np.ndarray:
    """
    CPU-side helper. Not part of ONNX.

    Args:
        output: model output [1, 20, 1, 1], [20, 1, 1], or [20].
    Returns:
        np.ndarray shape [5], one class id per organ.
    """
    logits = np.asarray(output, dtype=np.float32).reshape(5, 4)
    return np.argmax(logits, axis=1).astype(np.int64)


def average_ignore3(scores: Iterable[int]) -> float:
    """CPU-side helper matching the old pipeline's convention that class 3 is ignored."""
    vals = [float(v) for v in scores if int(v) != 3]
    return float(np.mean(vals)) if vals else float("nan")
