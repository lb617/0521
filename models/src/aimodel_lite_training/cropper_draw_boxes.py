#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cropper_draw_boxes.py

Run LiteFaceCropperModel on one image or a folder of images and save images
with predicted green bounding boxes.

Important:
- Preprocessing follows the training script of the cropper: OpenCV BGR input,
  resize to 256x256, divide by 255, CHW. No BGR->RGB conversion.
- The cropper model output is interpreted as:
    [conf, x1, y1, x2, y2]
  where xyxy are normalized to [0, 1].
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np
import torch

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def load_module_from_path(py_path: Path):
    spec = importlib.util.spec_from_file_location("user_model_def", str(py_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load Python file: {py_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_cropper(model_def_path: Path, device: torch.device):
    module = load_module_from_path(model_def_path)

    if hasattr(module, "build_model"):
        model = module.build_model("cropper")
    elif hasattr(module, "LiteFaceCropperModel"):
        model = module.LiteFaceCropperModel()
    else:
        raise RuntimeError(
            f"{model_def_path} must provide build_model('cropper') or LiteFaceCropperModel"
        )

    model = model.to(device)
    model.eval()
    return model


def load_state_dict_flexible(model: torch.nn.Module, ckpt_path: Path, device: torch.device):
    ckpt = torch.load(str(ckpt_path), map_location=device)
    if isinstance(ckpt, dict):
        for key in ["model_state_dict", "state_dict", "model"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    model.load_state_dict(ckpt, strict=True)



def find_images(input_path: Path, recursive: bool) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMG_EXTS:
            raise RuntimeError(f"Unsupported image file: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise RuntimeError(f"Input path not found: {input_path}")

    paths: List[Path] = []
    for ext in IMG_EXTS:
        if recursive:
            paths.extend(input_path.rglob(f"*{ext}"))
            paths.extend(input_path.rglob(f"*{ext.upper()}"))
        else:
            paths.extend(input_path.glob(f"*{ext}"))
            paths.extend(input_path.glob(f"*{ext.upper()}"))
    return sorted(set(p for p in paths if p.is_file()))



def preprocess_bgr(frame_bgr: np.ndarray) -> torch.Tensor:
    if frame_bgr.ndim == 2:
        frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)
    elif frame_bgr.ndim == 3 and frame_bgr.shape[2] == 4:
        frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_BGRA2BGR)

    frame = cv2.resize(frame_bgr, (256, 256), interpolation=cv2.INTER_LINEAR)
    frame = frame.astype(np.float32) / 255.0
    frame = frame.transpose(2, 0, 1)  # HWC -> CHW
    return torch.from_numpy(frame).unsqueeze(0)



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
    x1 -= w * margin
    y1 -= h * margin
    x2 += w * margin
    y2 += h * margin
    return sanitize_xyxy(np.array([x1, y1, x2, y2], dtype=np.float32))



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



def draw_box(frame_bgr: np.ndarray, box_xyxy: Tuple[int, int, int, int], conf: float, line_width: int, show_conf: bool) -> np.ndarray:
    out = frame_bgr.copy()
    x1, y1, x2, y2 = box_xyxy
    cv2.rectangle(out, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=line_width)

    if show_conf:
        text = f"{conf:.3f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = max(1, line_width - 1)
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        text_y1 = max(0, y1 - th - baseline - 6)
        text_y2 = text_y1 + th + baseline + 6
        text_x2 = min(out.shape[1] - 1, x1 + tw + 8)
        cv2.rectangle(out, (x1, text_y1), (text_x2, text_y2), color=(0, 255, 0), thickness=-1)
        cv2.putText(out, text, (x1 + 4, text_y2 - baseline - 3), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return out



def relative_output_path(src: Path, input_root: Path, output_dir: Path) -> Path:
    if input_root.is_file():
        return output_dir / src.name
    return output_dir / src.relative_to(input_root)



def main():
    parser = argparse.ArgumentParser(description="Run cropper inference and draw green boxes on images.")
    parser.add_argument("--model-def", required=True, help="Path to aimodel_lite_models.py")
    parser.add_argument("--ckpt", required=True, help="Path to cropper checkpoint")
    parser.add_argument("--input", required=True, help="Single image path or folder of images")
    parser.add_argument("--output-dir", required=True, help="Where to save images with drawn boxes")
    parser.add_argument("--recursive", action="store_true", help="Recursively search input folder for images")
    parser.add_argument("--conf-threshold", type=float, default=0.5, help="Only draw box when conf >= threshold")
    parser.add_argument("--margin", type=float, default=0.0, help="Optional bbox expansion ratio before drawing")
    parser.add_argument("--line-width", type=int, default=2, help="Rectangle line width")
    parser.add_argument("--show-conf", action="store_true", help="Draw confidence text above the green box")
    parser.add_argument("--save-no-box", action="store_true", help="Save original image even if confidence is below threshold")
    parser.add_argument("--device", default="", help="cuda / cpu; default auto")
    args = parser.parse_args()

    model_def_path = Path(args.model_def).resolve()
    ckpt_path = Path(args.ckpt).resolve()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_cropper(model_def_path, device)
    load_state_dict_flexible(model, ckpt_path, device)

    image_paths = find_images(input_path, args.recursive)
    if not image_paths:
        raise RuntimeError("No images found.")

    print(f"[Info] device={device}")
    print(f"[Info] images={len(image_paths)}")
    print(f"[Info] output_dir={output_dir}")

    num_saved = 0
    num_drawn = 0

    with torch.inference_mode():
        for img_path in image_paths:
            frame = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if frame is None:
                print(f"[Warn] skip unreadable image: {img_path}")
                continue

            original_h, original_w = frame.shape[:2]
            x = preprocess_bgr(frame).to(device)
            pred = model(x).flatten(1)[0].detach().cpu().numpy().astype(np.float32)

            conf = float(pred[0])
            box = sanitize_xyxy(pred[1:5])
            if args.margin > 0:
                box = expand_box_xyxy(box, args.margin)

            out_path = relative_output_path(img_path, input_path, output_dir)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if conf >= args.conf_threshold:
                pix_box = normalized_xyxy_to_pixels(box, original_w, original_h)
                vis = draw_box(frame, pix_box, conf, line_width=max(1, args.line_width), show_conf=args.show_conf)
                cv2.imwrite(str(out_path), vis)
                num_saved += 1
                num_drawn += 1
                print(
                    f"[OK] {img_path.name} | conf={conf:.4f} | "
                    f"box_norm=({box[0]:.4f}, {box[1]:.4f}, {box[2]:.4f}, {box[3]:.4f}) | "
                    f"box_pix={pix_box}"
                )
            else:
                if args.save_no_box:
                    if frame.ndim == 2:
                        save_img = frame
                    else:
                        save_img = frame.copy()
                    cv2.imwrite(str(out_path), save_img)
                    num_saved += 1
                print(f"[Skip] {img_path.name} | conf={conf:.4f} < threshold={args.conf_threshold:.4f}")

    print(f"[Done] saved={num_saved}, drawn={num_drawn}, total={len(image_paths)}")


if __name__ == "__main__":
    main()
