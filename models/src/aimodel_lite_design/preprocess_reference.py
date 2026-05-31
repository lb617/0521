#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
preprocess_reference.py

Reference CPU-side preprocessing and postprocessing for the three AIMODEL-friendly models.

Important:
- These functions output already-normalized float32 NCHW arrays.
- If you use these functions before feeding the board SDK, set converter/config mean=0 and std=1.
- Do not normalize twice.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


ORGAN_COLUMNS = ["eye", "nose", "cheek", "ear", "whisker"]


def ensure_bgr(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def crop_image_to_square(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    if h == w:
        return image
    d = abs(h - w)
    left = d // 2
    right = d - left
    if h > w:
        return image[left : h - right, :]
    return image[:, left : w - right]


def average_channels(image: np.ndarray) -> np.ndarray:
    return np.mean(image, axis=2, keepdims=True)


def to_nchw_batch(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        x = x[:, :, None]
    x = x.transpose(2, 0, 1)
    return x[None, ...].astype(np.float32)


def preprocess_quality(frame: np.ndarray) -> np.ndarray:
    """
    Match old FrameQualityScorer behavior:
    square crop -> resize 224 -> average channels -> /255 -> (x - 0.6901) / 0.1750

    Returns:
        [1, 1, 224, 224] float32.
    """
    frame = crop_image_to_square(frame)
    frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)

    if frame.ndim == 2:
        frame = frame[:, :, None]
    elif frame.shape[2] > 1:
        frame = average_channels(frame)

    frame = frame.astype(np.float32) / 255.0
    frame = (frame - 0.6901) / 0.1750
    return to_nchw_batch(frame)


def preprocess_cropper(frame: np.ndarray) -> np.ndarray:
    """
    New cropper input.

    Recommended normalization:
    - BGR or RGB consistency must match training.
    - Here we keep OpenCV BGR and scale to 0~1 only.
    - If you train with mean/std, put the same values here.

    Returns:
        [1, 3, 256, 256] float32.
    """
    frame = ensure_bgr(frame)
    frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LINEAR)
    frame = frame.astype(np.float32) / 255.0
    return to_nchw_batch(frame)


def preprocess_grimace(face_crop: np.ndarray) -> np.ndarray:
    """
    Match old GrimaceScorer behavior:
    resize 224 -> force 3 channels -> /255 -> (x - 0.5175) / 0.14

    Returns:
        [1, 3, 224, 224] float32.
    """
    frame = ensure_bgr(face_crop)
    frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)

    if frame.ndim == 2:
        frame = np.stack([frame] * 3, axis=2)
    elif frame.shape[2] == 1:
        frame = np.repeat(frame, 3, axis=2)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    frame = frame.astype(np.float32) / 255.0
    frame = (frame - 0.5175) / 0.14
    return to_nchw_batch(frame)


def square_bbox_xyxy(box: np.ndarray) -> np.ndarray:
    """
    box: [x1, y1, x2, y2] in pixel coordinates.
    Return square xyxy bbox around the same center.
    """
    x1, y1, x2, y2 = [float(v) for v in box]
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    side = max(w, h)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    return np.array([cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2], dtype=np.float32)


def pad_bbox_xyxy(box: np.ndarray, padding: float) -> np.ndarray:
    """
    padding: fraction of bbox side, same spirit as old FrameCropper padding.
    """
    x1, y1, x2, y2 = [float(v) for v in box]
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    px = w * float(padding)
    py = h * float(padding)
    return np.array([x1 - px, y1 - py, x2 + px, y2 + py], dtype=np.float32)


def crop_by_normalized_bbox(
    frame: np.ndarray,
    bbox_norm: np.ndarray,
    padding: float = 0.0,
) -> np.ndarray:
    """
    Args:
        frame: original image.
        bbox_norm: [conf, x1, y1, x2, y2], normalized xyxy.
    Returns:
        cropped image.
    """
    frame = ensure_bgr(frame)
    h, w = frame.shape[:2]

    _, x1, y1, x2, y2 = [float(v) for v in bbox_norm]
    box = np.array([x1 * w, y1 * h, x2 * w, y2 * h], dtype=np.float32)
    box = square_bbox_xyxy(box)
    box = pad_bbox_xyxy(box, padding)

    x1, y1, x2, y2 = box
    x1 = int(max(0, math.floor(x1)))
    y1 = int(max(0, math.floor(y1)))
    x2 = int(min(w, math.ceil(x2)))
    y2 = int(min(h, math.ceil(y2)))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid crop bbox after clipping: {(x1, y1, x2, y2)}")

    return frame[y1:y2, x1:x2].copy()


def postprocess_quality(output: np.ndarray, threshold_0_to_6: float) -> tuple[float, bool]:
    q_norm = float(np.asarray(output).reshape(-1)[0])
    q_score = q_norm * 6.0
    return q_score, bool(q_score >= float(threshold_0_to_6))


def postprocess_cropper(output: np.ndarray, confidence_threshold: float = 0.5) -> Optional[np.ndarray]:
    arr = np.asarray(output, dtype=np.float32).reshape(-1)
    if arr.size != 5:
        raise ValueError(f"Cropper output must have 5 values, got {arr.size}")

    conf, x1, y1, x2, y2 = arr.tolist()
    if conf < float(confidence_threshold):
        return None

    xa, xb = sorted([x1, x2])
    ya, yb = sorted([y1, y2])
    return np.array(
        [
            np.clip(conf, 0.0, 1.0),
            np.clip(xa, 0.0, 1.0),
            np.clip(ya, 0.0, 1.0),
            np.clip(xb, 0.0, 1.0),
            np.clip(yb, 0.0, 1.0),
        ],
        dtype=np.float32,
    )


def postprocess_grimace(output: np.ndarray) -> tuple[np.ndarray, float]:
    logits = np.asarray(output, dtype=np.float32).reshape(5, 4)
    scores = np.argmax(logits, axis=1).astype(np.int64)
    vals = [float(v) for v in scores if int(v) != 3]
    avg = float(np.mean(vals)) if vals else float("nan")
    return scores, avg
