#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
video_pipeline_scores_with_frame_score.py

读取视频，按频率抽帧，跑三阶段 GrimACE lite pipeline，并输出 CSV：

    frame_index,timestamp_sec,frame_score,eye,nose,cheek,ear,whisker

关键行为：
1. frame_score = LiteFrameQualityModel 输出的 0~1 分数 * 6
2. 不用 frame_score 做筛选
3. 不用 cropper conf 做筛选
4. 每个成功读取并抽到的帧都会尽量输出一行
5. cropper bbox 无效或太小时，不跳过该帧，而是 fallback 到整帧送入 grimace scorer

注意：
- cropper 预处理保持和训练脚本一致：OpenCV BGR，resize 256，/255，CHW，不做 BGR->RGB。
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch


DEFAULT_ORGAN_COLUMNS = ["eye", "nose", "cheek", "ear", "whisker"]


def load_module_from_path(py_path: Path):
    spec = importlib.util.spec_from_file_location("user_model_def", str(py_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load Python file: {py_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_model_from_module(module, model_name: str) -> torch.nn.Module:
    if hasattr(module, "build_model"):
        return module.build_model(model_name)

    name_to_class = {
        "quality": "LiteFrameQualityModel",
        "cropper": "LiteFaceCropperModel",
        "grimace": "LiteGrimaceScoreModel",
    }
    class_name = name_to_class[model_name]
    if not hasattr(module, class_name):
        raise RuntimeError(f"Model definition must provide build_model() or {class_name}")
    return getattr(module, class_name)()


def load_state_dict_flexible(model: torch.nn.Module, ckpt_path: Path, device: torch.device):
    ckpt = torch.load(str(ckpt_path), map_location=device)

    if isinstance(ckpt, dict):
        for key in ["model_state_dict", "state_dict", "model"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break

    model.load_state_dict(ckpt, strict=True)


def ensure_bgr(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    if frame.ndim == 3 and frame.shape[2] == 3:
        return frame
    raise RuntimeError(f"Unsupported frame shape: {frame.shape}")


def preprocess_quality(frame_bgr: np.ndarray) -> torch.Tensor:
    gray = cv2.cvtColor(ensure_bgr(frame_bgr), cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (224, 224), interpolation=cv2.INTER_LINEAR)
    gray = gray.astype(np.float32) / 255.0
    return torch.from_numpy(gray).unsqueeze(0).unsqueeze(0)


def preprocess_cropper(frame_bgr: np.ndarray) -> torch.Tensor:
    # 和 cropper 训练脚本一致：OpenCV BGR，不转 RGB
    x = cv2.resize(ensure_bgr(frame_bgr), (256, 256), interpolation=cv2.INTER_LINEAR)
    x = x.astype(np.float32) / 255.0
    x = x.transpose(2, 0, 1)
    return torch.from_numpy(x).unsqueeze(0)


def preprocess_grimace(crop_bgr: np.ndarray, grimace_color: str) -> torch.Tensor:
    x = cv2.resize(ensure_bgr(crop_bgr), (224, 224), interpolation=cv2.INTER_LINEAR)
    if grimace_color == "rgb":
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    elif grimace_color != "bgr":
        raise ValueError("--grimace-color must be bgr or rgb")
    x = x.astype(np.float32) / 255.0
    x = x.transpose(2, 0, 1)
    return torch.from_numpy(x).unsqueeze(0)


def sanitize_xyxy(box4: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box4.astype(np.float32).tolist()
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    return np.array([
        np.clip(x1, 0.0, 1.0),
        np.clip(y1, 0.0, 1.0),
        np.clip(x2, 0.0, 1.0),
        np.clip(y2, 0.0, 1.0),
    ], dtype=np.float32)


def expand_box_xyxy(box4: np.ndarray, margin: float) -> np.ndarray:
    x1, y1, x2, y2 = box4.astype(np.float32).tolist()
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return sanitize_xyxy(np.array([
        x1 - w * margin,
        y1 - h * margin,
        x2 + w * margin,
        y2 + h * margin,
    ], dtype=np.float32))


def normalized_xyxy_to_pixels(box4: np.ndarray, width: int, height: int) -> Tuple[int, int, int, int]:
    x1 = int(round(float(box4[0]) * (width - 1)))
    y1 = int(round(float(box4[1]) * (height - 1)))
    x2 = int(round(float(box4[2]) * (width - 1)))
    y2 = int(round(float(box4[3]) * (height - 1)))
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))
    return x1, y1, x2, y2


def crop_or_full_frame(frame_bgr: np.ndarray, box_px: Tuple[int, int, int, int], min_crop_size: int) -> np.ndarray:
    # 不筛选帧；预测框太小/无效时用整帧代替 crop
    x1, y1, x2, y2 = box_px
    if x2 - x1 + 1 < min_crop_size or y2 - y1 + 1 < min_crop_size:
        return frame_bgr
    crop = frame_bgr[y1:y2 + 1, x1:x2 + 1]
    return frame_bgr if crop.size == 0 else crop


def compute_organ_scores(logits_5x4: torch.Tensor, score_mode: str) -> List[float]:
    probs = torch.softmax(logits_5x4, dim=-1)
    if score_mode == "class":
        return probs.argmax(dim=-1).detach().cpu().numpy().astype(float).tolist()
    if score_mode == "expected":
        values = torch.arange(4, device=probs.device, dtype=probs.dtype)
        return (probs * values).sum(dim=-1).detach().cpu().numpy().astype(float).tolist()
    raise ValueError("--score-mode must be class or expected")


def compute_frame_step(video_fps: float, frame_step: int, sample_fps: float) -> int:
    if frame_step > 0:
        return max(1, frame_step)
    if sample_fps <= 0:
        raise ValueError("Need either --frame-step > 0 or --sample-fps > 0")
    if video_fps <= 1e-6:
        raise RuntimeError("Cannot read valid video FPS. Please use --frame-step instead.")
    return max(1, int(round(video_fps / sample_fps)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-def", required=True, help="Path to aimodel_lite_models.py")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--quality-ckpt", required=True, help="Frame quality checkpoint")
    parser.add_argument("--cropper-ckpt", required=True, help="Cropper checkpoint")
    parser.add_argument("--grimace-ckpt", required=True, help="Grimace scorer checkpoint")
    parser.add_argument("--output", required=True, help="Output CSV path")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--frame-step", type=int, default=0, help="Process one frame every N frames")
    group.add_argument("--sample-fps", type=float, default=0.0, help="Approximate sampled FPS")

    parser.add_argument("--score-mode", choices=["class", "expected"], default="class")
    parser.add_argument("--frame-score-scale", type=float, default=6.0)
    parser.add_argument("--crop-margin", type=float, default=0.10)
    parser.add_argument("--min-crop-size", type=int, default=8)
    parser.add_argument("--grimace-color", choices=["bgr", "rgb"], default="bgr")
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    model_module = load_module_from_path(Path(args.model_def).resolve())
    organ_columns = list(getattr(model_module, "ORGAN_COLUMNS", DEFAULT_ORGAN_COLUMNS))

    quality_model = build_model_from_module(model_module, "quality").to(device).eval()
    cropper_model = build_model_from_module(model_module, "cropper").to(device).eval()
    grimace_model = build_model_from_module(model_module, "grimace").to(device).eval()

    load_state_dict_flexible(quality_model, Path(args.quality_ckpt).resolve(), device)
    load_state_dict_flexible(cropper_model, Path(args.cropper_ckpt).resolve(), device)
    load_state_dict_flexible(grimace_model, Path(args.grimace_ckpt).resolve(), device)

    cap = cv2.VideoCapture(str(Path(args.video).resolve()))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    step = compute_frame_step(video_fps, args.frame_step, args.sample_fps)

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[Info] device={device}")
    print(f"[Info] video_fps={video_fps:.6f}, step={step}")
    print("[Info] no filtering: quality score and cropper conf will not drop frames")
    print(f"[Info] output={output_path}")

    fieldnames = ["frame_index", "timestamp_sec", "frame_score", *organ_columns]

    processed = 0
    written = 0

    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        with torch.inference_mode():
            frame_index = -1
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame_index += 1
                if frame_index % step != 0:
                    continue

                processed += 1
                frame_bgr = ensure_bgr(frame)
                h, w = frame_bgr.shape[:2]

                if video_fps > 1e-6:
                    timestamp_sec = frame_index / video_fps
                else:
                    timestamp_sec = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0

                # 1. frame quality score: 0~1 -> *6, but no filtering.
                qx = preprocess_quality(frame_bgr).to(device)
                quality_score = float(quality_model(qx).flatten()[0].detach().cpu())
                frame_score = quality_score * float(args.frame_score_scale)

                # 2. cropper: no conf-threshold filtering.
                cx = preprocess_cropper(frame_bgr).to(device)
                cropper_pred = cropper_model(cx).flatten(1)[0].detach().cpu().numpy().astype(np.float32)

                box_norm = sanitize_xyxy(cropper_pred[1:5])
                if args.crop_margin > 0:
                    box_norm = expand_box_xyxy(box_norm, args.crop_margin)

                box_px = normalized_xyxy_to_pixels(box_norm, width=w, height=h)
                crop_bgr = crop_or_full_frame(frame_bgr, box_px, args.min_crop_size)

                # 3. grimace scorer.
                gx = preprocess_grimace(crop_bgr, args.grimace_color).to(device)
                logits = grimace_model(gx).reshape(1, 5, 4)[0]
                organ_scores = compute_organ_scores(logits, args.score_mode)

                row: Dict[str, object] = {
                    "frame_index": frame_index,
                    "timestamp_sec": f"{timestamp_sec:.6f}",
                    "frame_score": f"{frame_score:.6f}",
                }

                for organ, score in zip(organ_columns, organ_scores):
                    row[organ] = int(score) if args.score_mode == "class" else f"{score:.6f}"

                writer.writerow(row)
                written += 1

                print(
                    f"[OK] frame={frame_index} "
                    f"time={timestamp_sec:.3f}s "
                    f"frame_score={frame_score:.4f} "
                    f"scores={organ_scores}"
                )

    cap.release()
    print(f"[Done] processed_sampled_frames={processed}, written_rows={written}")


if __name__ == "__main__":
    main()
