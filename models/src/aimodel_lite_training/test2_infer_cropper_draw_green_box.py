#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
infer_cropper_draw_green_box.py

Standalone inference/debug script for LiteFaceCropperModel.

It reads one image or a folder/glob of images, runs the cropper model, and draws
one green bounding box on each image.

Important:
- This script matches the training preprocess in train_face_cropper_lite.py:
  cv2.imread BGR -> resize to 256x256 -> float32 / 255 -> CHW.
- Cropper output is interpreted as: [conf, x1, y1, x2, y2], normalized xyxy.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np
import torch


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def import_module_from_path(module_path: str):
    path = Path(module_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Model definition file not found: {path}")
    spec = importlib.util.spec_from_file_location("aimodel_lite_models_dynamic", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import module from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_state_dict_flexible(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> None:
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        for key in ["model_state_dict", "state_dict", "model"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    model.load_state_dict(ckpt, strict=True)


def build_cropper(model_def: str, ckpt_path: str, device: torch.device) -> torch.nn.Module:
    module = import_module_from_path(model_def)

    if hasattr(module, "build_model"):
        model = module.build_model("cropper")
    elif hasattr(module, "LiteFaceCropperModel"):
        model = module.LiteFaceCropperModel()
    else:
        raise AttributeError(
            "Model definition must provide build_model('cropper') or LiteFaceCropperModel."
        )

    load_state_dict_flexible(model, ckpt_path, device)
    model.to(device)
    model.eval()
    return model


def preprocess_like_training(frame_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    """Match FaceCropperLiteDataset.preprocess from train_face_cropper_lite.py."""
    if frame_bgr.ndim == 2:
        frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)
    elif frame_bgr.ndim == 3 and frame_bgr.shape[2] == 4:
        frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_BGRA2BGR)

    resized = cv2.resize(frame_bgr, (256, 256), interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.float32) / 255.0
    x = x.transpose(2, 0, 1)  # HWC BGR -> CHW BGR
    x = torch.from_numpy(x).unsqueeze(0).to(device=device, dtype=torch.float32)
    return x


def sanitize_normalized_xyxy(raw_box: np.ndarray) -> Tuple[float, float, float, float]:
    """Clamp to [0,1] and guarantee x1 <= x2, y1 <= y2."""
    x1, y1, x2, y2 = [float(v) for v in raw_box]
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))
    return x1, y1, x2, y2


def normalized_to_pixel_xyxy(
    box: Tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    margin: float = 0.0,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box

    if margin > 0:
        bw = x2 - x1
        bh = y2 - y1
        x1 -= bw * margin
        x2 += bw * margin
        y1 -= bh * margin
        y2 += bh * margin
        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))
        x2 = max(0.0, min(1.0, x2))
        y2 = max(0.0, min(1.0, y2))

    px1 = int(round(x1 * (image_width - 1)))
    py1 = int(round(y1 * (image_height - 1)))
    px2 = int(round(x2 * (image_width - 1)))
    py2 = int(round(y2 * (image_height - 1)))

    px1 = max(0, min(image_width - 1, px1))
    px2 = max(0, min(image_width - 1, px2))
    py1 = max(0, min(image_height - 1, py1))
    py2 = max(0, min(image_height - 1, py2))
    return px1, py1, px2, py2


@torch.inference_mode()
def infer_one(
    model: torch.nn.Module,
    image_path: Path,
    output_path: Path,
    device: torch.device,
    conf_threshold: float,
    margin: float,
    thickness: int,
    draw_text: bool,
) -> dict:
    frame = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if frame is None:
        raise RuntimeError(f"cv2.imread failed: {image_path}")

    if frame.ndim == 2:
        draw_img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        draw_img = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    else:
        draw_img = frame.copy()

    h, w = draw_img.shape[:2]
    x = preprocess_like_training(frame, device)
    pred = model(x).flatten(1)[0].detach().cpu().numpy().astype(float)

    conf = float(pred[0])
    norm_box = sanitize_normalized_xyxy(pred[1:5])
    px1, py1, px2, py2 = normalized_to_pixel_xyxy(norm_box, w, h, margin=margin)

    status = "drawn" if conf >= conf_threshold else "low_conf"
    if conf >= conf_threshold:
        cv2.rectangle(draw_img, (px1, py1), (px2, py2), (0, 255, 0), thickness)
        if draw_text:
            label = f"conf={conf:.3f}"
            text_org = (px1, max(0, py1 - 8))
            cv2.putText(
                draw_img,
                label,
                text_org,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                max(1, thickness),
                cv2.LINE_AA,
            )
    elif draw_text:
        label = f"low conf={conf:.3f}"
        cv2.putText(
            draw_img,
            label,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            max(1, thickness),
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), draw_img)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed: {output_path}")

    return {
        "image": str(image_path),
        "output": str(output_path),
        "conf": conf,
        "norm_xyxy": norm_box,
        "pixel_xyxy": [px1, py1, px2, py2],
        "status": status,
    }


def collect_images(input_path: str, recursive: bool) -> List[Path]:
    p = Path(input_path)

    # Existing single image file.
    if p.exists() and p.is_file():
        if p.suffix.lower() not in IMG_EXTS:
            raise ValueError(f"Input file is not a supported image: {p}")
        return [p]

    # Existing directory.
    if p.exists() and p.is_dir():
        paths: List[Path] = []
        iterator: Iterable[Path] = p.rglob("*") if recursive else p.glob("*")
        for item in iterator:
            if item.is_file() and item.suffix.lower() in IMG_EXTS:
                paths.append(item)
        return sorted(paths)

    # Glob pattern.
    paths = [Path(x) for x in sorted(map(str, Path().glob(input_path)))]
    paths = [x for x in paths if x.is_file() and x.suffix.lower() in IMG_EXTS]
    if not paths:
        raise FileNotFoundError(f"No images found from input: {input_path}")
    return paths


def output_path_for(image_path: Path, input_root: Path | None, output_dir: Path) -> Path:
    if input_root is not None and input_root.exists() and input_root.is_dir():
        try:
            rel = image_path.relative_to(input_root)
        except ValueError:
            rel = Path(image_path.name)
        return output_dir / rel.with_name(rel.stem + "_cropper_box" + rel.suffix)
    return output_dir / f"{image_path.stem}_cropper_box{image_path.suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LiteFaceCropperModel on image(s) and draw green bbox."
    )
    parser.add_argument("--model-def", required=True, help="Path to aimodel_lite_models.py")
    parser.add_argument("--ckpt", required=True, help="Path to cropper checkpoint .pt")
    parser.add_argument("--input", required=True, help="Image file, directory, or glob pattern")
    parser.add_argument("--output-dir", default="cropper_vis", help="Directory for output images")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan input directory")
    parser.add_argument("--conf-threshold", type=float, default=0.5, help="Draw box only if conf >= threshold")
    parser.add_argument("--margin", type=float, default=0.0, help="Optional bbox expansion ratio, e.g. 0.1")
    parser.add_argument("--thickness", type=int, default=2, help="Green box thickness")
    parser.add_argument("--no-text", action="store_true", help="Do not draw confidence text")
    parser.add_argument("--device", default="", help="cuda, cpu, cuda:0, etc. Default: auto")
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_cropper(args.model_def, args.ckpt, device)

    images = collect_images(args.input, recursive=args.recursive)
    if not images:
        raise RuntimeError(f"No supported images found: {args.input}")

    input_root = Path(args.input) if Path(args.input).exists() and Path(args.input).is_dir() else None
    output_dir = Path(args.output_dir)

    print(f"[Info] device: {device}")
    print(f"[Info] images: {len(images)}")
    print(f"[Info] output_dir: {output_dir.resolve()}")

    for image_path in images:
        out_path = output_path_for(image_path, input_root, output_dir)
        result = infer_one(
            model=model,
            image_path=image_path,
            output_path=out_path,
            device=device,
            conf_threshold=args.conf_threshold,
            margin=args.margin,
            thickness=args.thickness,
            draw_text=(not args.no_text),
        )
        print(
            f"{Path(result['image']).name} | "
            f"conf={result['conf']:.4f} | "
            f"xyxy_norm={tuple(round(v, 4) for v in result['norm_xyxy'])} | "
            f"xyxy_px={result['pixel_xyxy']} | "
            f"{result['status']} -> {result['output']}"
        )


if __name__ == "__main__":
    main()
