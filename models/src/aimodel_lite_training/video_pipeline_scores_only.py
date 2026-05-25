#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
video_pipeline_scores_only.py

Read a video, sample frames at a fixed interval or target FPS, run the three-stage
GrimACE-lite pipeline, and output only:

    frame_index, timestamp_sec, eye, nose, cheek, ear, whisker

By default the five organ scores are argmax class scores 0/1/2/3.
Use --score-mode expected to output soft expected scores in [0, 3].

Pipeline:
    frame -> LiteFrameQualityModel -> LiteFaceCropperModel -> LiteGrimaceScoreModel
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import cv2
import numpy as np
import torch
from torch import nn


DEFAULT_ORGANS = ["eye", "nose", "cheek", "ear", "whisker"]
CLASS_VALUES = torch.tensor([0.0, 1.0, 2.0, 3.0])


def import_model_file(model_def_path: str):
    """Import aimodel_lite_models.py from any path, including filenames with parentheses."""
    path = Path(model_def_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"model definition file not found: {path}")

    spec = importlib.util.spec_from_file_location("aimodel_lite_models_runtime", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import model definition from: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def torch_load(path: str, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def find_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    """Accept common checkpoint formats and return a state_dict."""
    if isinstance(ckpt, nn.Module):
        return ckpt.state_dict()

    if not isinstance(ckpt, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")

    for key in ("state_dict", "model_state_dict", "model", "net", "network"):
        value = ckpt.get(key)
        if isinstance(value, nn.Module):
            return value.state_dict()
        if isinstance(value, dict) and value and all(torch.is_tensor(v) for v in value.values()):
            return value

    if ckpt and all(torch.is_tensor(v) for v in ckpt.values()):
        return ckpt

    raise KeyError(
        "Could not find a state_dict in checkpoint. Expected raw state_dict or one of: "
        "state_dict, model_state_dict, model, net, network."
    )


def clean_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove common wrapper prefixes such as module., model., net., network."""
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "model.", "net.", "network."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    return cleaned


def build_and_load_model(
    module: Any,
    model_name: str,
    ckpt_path: str,
    device: torch.device,
    strict: bool = True,
) -> nn.Module:
    if not hasattr(module, "build_model"):
        raise AttributeError("model definition file must define build_model(model_name)")

    model = module.build_model(model_name)
    ckpt = torch_load(ckpt_path, device)
    state_dict = clean_state_dict_keys(find_state_dict(ckpt))
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)

    if not strict:
        if missing:
            print(f"[WARN] {model_name}: missing keys: {missing}")
        if unexpected:
            print(f"[WARN] {model_name}: unexpected keys: {unexpected}")

    model.to(device)
    model.eval()
    return model


def bgr_to_tensor(
    image_bgr: np.ndarray,
    size_hw: Tuple[int, int],
    rgb: bool = True,
    gray: bool = False,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Convert OpenCV BGR uint8 image to torch float tensor [1, C, H, W] in 0..1.

    If training used mean/std normalization, add exactly the same normalization here.
    """
    out_h, out_w = size_hw

    if gray:
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA)
        image = image.astype(np.float32) / 255.0
        tensor = torch.from_numpy(image)[None, None, :, :]
    else:
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) if rgb else image_bgr
        image = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA)
        image = image.astype(np.float32) / 255.0
        tensor = torch.from_numpy(image).permute(2, 0, 1)[None, :, :, :]

    if device is not None:
        tensor = tensor.to(device, non_blocking=True)
    return tensor


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def decode_bbox_xyxy(
    cropper_output: torch.Tensor,
    frame_w: int,
    frame_h: int,
    margin: float = 0.0,
) -> Tuple[float, Tuple[int, int, int, int]]:
    """
    Cropper output shape: [1, 5, 1, 1], sigmoid values [conf, x1, y1, x2, y2].
    Return confidence and clipped pixel bbox xyxy.
    """
    values = cropper_output.detach().float().cpu().view(-1).tolist()
    if len(values) != 5:
        raise ValueError(f"cropper output should contain 5 values, got {len(values)}")

    conf, x1, y1, x2, y2 = values
    x1, y1, x2, y2 = map(clamp01, (x1, y1, x2, y2))

    # Direct xyxy sigmoid outputs do not guarantee x1<x2/y1<y2, so fix ordering here.
    left, right = sorted([x1, x2])
    top, bottom = sorted([y1, y2])

    if margin > 0:
        bw = right - left
        bh = bottom - top
        left -= bw * margin
        right += bw * margin
        top -= bh * margin
        bottom += bh * margin
        left, top, right, bottom = map(clamp01, (left, top, right, bottom))

    px1 = int(round(left * (frame_w - 1)))
    py1 = int(round(top * (frame_h - 1)))
    px2 = int(round(right * (frame_w - 1)))
    py2 = int(round(bottom * (frame_h - 1)))

    px1 = max(0, min(frame_w - 1, px1))
    py1 = max(0, min(frame_h - 1, py1))
    px2 = max(0, min(frame_w - 1, px2))
    py2 = max(0, min(frame_h - 1, py2))

    return float(conf), (px1, py1, px2, py2)


def decode_five_scores(
    logits: torch.Tensor,
    organ_names: Iterable[str],
    score_mode: str,
) -> Dict[str, float | int]:
    """
    Grimace output shape: [1, 20, 1, 1]. Reshape to [5, 4].

    score_mode='class'    -> five integer argmax class scores, each in {0,1,2,3}
    score_mode='expected' -> five soft expected scores, each in [0,3]
    """
    organ_names = list(organ_names)
    num_organs = len(organ_names)

    logits = logits.detach().float().cpu().view(num_organs, 4)
    probs = torch.softmax(logits, dim=-1)

    if score_mode == "class":
        pred_class = probs.argmax(dim=-1)
        return {name: int(pred_class[i].item()) for i, name in enumerate(organ_names)}

    if score_mode == "expected":
        expected = (probs * CLASS_VALUES.view(1, 4)).sum(dim=-1)
        return {name: round(float(expected[i].item()), 6) for i, name in enumerate(organ_names)}

    raise ValueError(f"unknown score_mode={score_mode!r}")


def make_sampling_step(video_fps: float, frame_step: Optional[int], sample_fps: Optional[float]) -> int:
    if sample_fps is not None and sample_fps > 0:
        if video_fps and not math.isnan(video_fps) and video_fps > 0:
            return max(1, int(round(video_fps / sample_fps)))
        print("[WARN] video FPS is unavailable; falling back to --frame-step or 1")
        return max(1, int(frame_step or 1))

    return max(1, int(frame_step or 1))


def infer_video(args: argparse.Namespace) -> None:
    model_module = import_model_file(args.model_def)
    organ_names = list(getattr(model_module, "ORGAN_COLUMNS", DEFAULT_ORGANS))
    if len(organ_names) != 5:
        raise ValueError(f"expected exactly 5 organ names, got {len(organ_names)}: {organ_names}")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    strict = not args.non_strict_load

    quality_model = build_and_load_model(model_module, "quality", args.quality_ckpt, device, strict=strict)
    cropper_model = build_and_load_model(model_module, "cropper", args.cropper_ckpt, device, strict=strict)
    grimace_model = build_and_load_model(model_module, "grimace", args.grimace_ckpt, device, strict=strict)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {args.video}")

    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = make_sampling_step(video_fps, args.frame_step, args.sample_fps)

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_frames_dir = Path(args.save_frames_dir).expanduser().resolve() if args.save_frames_dir else None
    if save_frames_dir:
        save_frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] device={device}")
    print(f"[INFO] video_fps={video_fps:.3f}, total_frames={total_frames}, sample_step={step}")
    print(f"[INFO] score_mode={args.score_mode}")
    print(f"[INFO] writing scores to: {output_path}")
    if save_frames_dir:
        print(f"[INFO] saving scored frames to: {save_frames_dir}")

    sampled = 0
    scored = 0
    skipped_low_quality = 0
    skipped_low_crop_conf = 0
    skipped_too_small_crop = 0

    with output_path.open("w", encoding="utf-8", newline="") as f, torch.inference_mode():
        writer = csv.DictWriter(f, fieldnames=["frame_index", "timestamp_sec", *organ_names])
        writer.writeheader()

        frame_index = -1
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_index += 1

            if frame_index % step != 0:
                continue

            sampled += 1
            frame_h, frame_w = frame_bgr.shape[:2]
            timestamp_sec = frame_index / video_fps if video_fps > 0 else float(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)

            # Stage 1: frame quality. Low-quality frames are skipped and not written,
            # because they do not have valid five-organ scores.
            quality_input = bgr_to_tensor(frame_bgr, (224, 224), gray=True, device=device)
            quality_score = float(quality_model(quality_input).detach().cpu().view(-1)[0].item())
            if quality_score < args.quality_threshold:
                skipped_low_quality += 1
                continue

            # Stage 2: face / mouse-face cropper.
            cropper_input = bgr_to_tensor(frame_bgr, (256, 256), rgb=True, device=device)
            cropper_output = cropper_model(cropper_input)
            crop_conf, bbox = decode_bbox_xyxy(
                cropper_output,
                frame_w=frame_w,
                frame_h=frame_h,
                margin=args.crop_margin,
            )
            if crop_conf < args.crop_conf_threshold:
                skipped_low_crop_conf += 1
                continue

            x1, y1, x2, y2 = bbox
            box_w = x2 - x1 + 1
            box_h = y2 - y1 + 1
            if box_w < args.min_crop_size or box_h < args.min_crop_size:
                skipped_too_small_crop += 1
                continue

            crop_bgr = frame_bgr[y1:y2 + 1, x1:x2 + 1]

            # Stage 3: five-organ grimace scores.
            grimace_input = bgr_to_tensor(crop_bgr, (224, 224), rgb=True, device=device)
            logits = grimace_model(grimace_input)
            scores = decode_five_scores(logits, organ_names, args.score_mode)

            row = {
                "frame_index": frame_index,
                "timestamp_sec": round(float(timestamp_sec), 6),
                **scores,
            }
            writer.writerow(row)
            scored += 1

            if save_frames_dir:
                cv2.imwrite(str(save_frames_dir / f"frame_{frame_index:08d}.jpg"), frame_bgr)

    cap.release()
    print(
        "[INFO] done. "
        f"sampled={sampled}, scored={scored}, "
        f"skipped_low_quality={skipped_low_quality}, "
        f"skipped_low_crop_conf={skipped_low_crop_conf}, "
        f"skipped_too_small_crop={skipped_too_small_crop}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample video frames, run LiteFrameQualityModel -> LiteFaceCropperModel "
            "-> LiteGrimaceScoreModel, and output only frame_index + five organ scores."
        )
    )

    parser.add_argument("--model-def", required=True, help="Path to aimodel_lite_models.py.")
    parser.add_argument("--video", required=True, help="Input video path.")

    parser.add_argument("--quality-ckpt", required=True, help="Checkpoint for LiteFrameQualityModel.")
    parser.add_argument("--cropper-ckpt", required=True, help="Checkpoint for LiteFaceCropperModel.")
    parser.add_argument("--grimace-ckpt", required=True, help="Checkpoint for LiteGrimaceScoreModel.")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--frame-step", type=int, default=30, help="Process every N frames. Default: 30.")
    group.add_argument("--sample-fps", type=float, default=None, help="Process approximately this many frames per second.")

    parser.add_argument("--score-mode", choices=["class", "expected"], default="class",
                        help="class: argmax 0/1/2/3 per organ; expected: soft expected score in [0,3]. Default: class.")
    parser.add_argument("--quality-threshold", type=float, default=0.5, help="Skip frames below this quality score.")
    parser.add_argument("--crop-conf-threshold", type=float, default=0.5, help="Skip frames below this cropper confidence.")
    parser.add_argument("--crop-margin", type=float, default=0.10, help="Relative margin around predicted bbox before scoring.")
    parser.add_argument("--min-crop-size", type=int, default=8, help="Skip crop boxes smaller than this many pixels in width/height.")

    parser.add_argument("--output", default="grimace_frame_scores.csv", help="Output CSV file.")
    parser.add_argument("--save-frames-dir", default=None,
                        help="Optional directory to save original frames that produced five scores.")

    parser.add_argument("--device", default=None, help="cpu, cuda, cuda:0, etc. Default: auto.")
    parser.add_argument("--non-strict-load", action="store_true", help="Load checkpoints with strict=False.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    infer_video(args)


if __name__ == "__main__":
    main()
