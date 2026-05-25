#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GrimACE Lite Web Demo

左侧播放视频，右侧实时显示：
- quality / frame_score
- eye, nose, cheek, ear, whisker 五个部位评分

架构：
- 前端播放视频，并定时把当前视频帧抓成 JPEG 发给 /infer_frame
- 后端加载三个 PyTorch 模型，只对收到的当前帧推理
- 如果后端正在处理上一帧，/infer_frame 直接返回 {"busy": true}
  前端会跳过这一帧，不排队，避免延迟越积越多

运行示例：
python app.py \
  --model-def ./aimodel_lite_models.py \
  --quality-ckpt runs/frame_quality_lite/frame_quality_lite_best.pt \
  --cropper-ckpt runs/face_cropper_lite/face_cropper_lite_best.pt \
  --grimace-ckpt runs/grimace_scorer_lite/grimace_scorer_lite_best.pt \
  --video data/v1.mp4 \
  --host 0.0.0.0 \
  --port 7860
"""

from __future__ import annotations

import argparse
import importlib.util
import threading
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


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
    # 和你的 cropper 训练脚本保持一致：OpenCV BGR，不转 RGB
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

    return np.array(
        [
            np.clip(x1, 0.0, 1.0),
            np.clip(y1, 0.0, 1.0),
            np.clip(x2, 0.0, 1.0),
            np.clip(y2, 0.0, 1.0),
        ],
        dtype=np.float32,
    )


def expand_box_xyxy(box4: np.ndarray, margin: float) -> np.ndarray:
    x1, y1, x2, y2 = box4.astype(np.float32).tolist()
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return sanitize_xyxy(
        np.array(
            [
                x1 - w * margin,
                y1 - h * margin,
                x2 + w * margin,
                y2 + h * margin,
            ],
            dtype=np.float32,
        )
    )


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
    # 不筛选帧：预测框太小或无效时用整帧代替 crop
    x1, y1, x2, y2 = box_px
    if x2 - x1 + 1 < min_crop_size or y2 - y1 + 1 < min_crop_size:
        return frame_bgr
    crop = frame_bgr[y1 : y2 + 1, x1 : x2 + 1]
    return frame_bgr if crop.size == 0 else crop


def compute_organ_scores(logits_5x4: torch.Tensor, score_mode: str) -> List[float]:
    probs = torch.softmax(logits_5x4, dim=-1)

    if score_mode == "class":
        return probs.argmax(dim=-1).detach().cpu().numpy().astype(float).tolist()

    if score_mode == "expected":
        values = torch.arange(4, device=probs.device, dtype=probs.dtype)
        return (probs * values).sum(dim=-1).detach().cpu().numpy().astype(float).tolist()

    raise ValueError("--score-mode must be class or expected")


class GrimaceRunner:
    def __init__(
        self,
        model_def: Path,
        quality_ckpt: Path,
        cropper_ckpt: Path,
        grimace_ckpt: Path,
        device: str = "",
        score_mode: str = "class",
        frame_score_scale: float = 6.0,
        crop_margin: float = 0.10,
        min_crop_size: int = 8,
        grimace_color: str = "bgr",
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.score_mode = score_mode
        self.frame_score_scale = float(frame_score_scale)
        self.crop_margin = float(crop_margin)
        self.min_crop_size = int(min_crop_size)
        self.grimace_color = grimace_color

        module = load_module_from_path(model_def)
        self.organ_columns = list(getattr(module, "ORGAN_COLUMNS", DEFAULT_ORGAN_COLUMNS))

        self.quality_model = build_model_from_module(module, "quality").to(self.device).eval()
        self.cropper_model = build_model_from_module(module, "cropper").to(self.device).eval()
        self.grimace_model = build_model_from_module(module, "grimace").to(self.device).eval()

        load_state_dict_flexible(self.quality_model, quality_ckpt, self.device)
        load_state_dict_flexible(self.cropper_model, cropper_ckpt, self.device)
        load_state_dict_flexible(self.grimace_model, grimace_ckpt, self.device)

    def infer_bgr(self, frame_bgr: np.ndarray) -> Dict:
        t0 = time.perf_counter()

        frame_bgr = ensure_bgr(frame_bgr)
        h, w = frame_bgr.shape[:2]

        with torch.inference_mode():
            # 1. quality
            qx = preprocess_quality(frame_bgr).to(self.device)
            quality_raw = float(self.quality_model(qx).flatten()[0].detach().cpu())
            frame_score = quality_raw * self.frame_score_scale

            # 2. cropper
            cx = preprocess_cropper(frame_bgr).to(self.device)
            cropper_pred = self.cropper_model(cx).flatten(1)[0].detach().cpu().numpy().astype(np.float32)
            cropper_conf = float(cropper_pred[0])

            box_norm = sanitize_xyxy(cropper_pred[1:5])
            if self.crop_margin > 0:
                box_norm = expand_box_xyxy(box_norm, self.crop_margin)

            box_px = normalized_xyxy_to_pixels(box_norm, width=w, height=h)
            crop_bgr = crop_or_full_frame(frame_bgr, box_px, self.min_crop_size)

            # 3. grimace
            gx = preprocess_grimace(crop_bgr, self.grimace_color).to(self.device)
            logits = self.grimace_model(gx).reshape(1, 5, 4)[0]
            organ_scores = compute_organ_scores(logits, self.score_mode)

        scores = {}
        for organ, score in zip(self.organ_columns, organ_scores):
            scores[organ] = int(score) if self.score_mode == "class" else float(score)

        return {
            "busy": False,
            "quality_raw": quality_raw,          # 0~1
            "quality_score": frame_score,        # 0~6
            "frame_score": frame_score,          # alias
            "scores": scores,
            "score_mode": self.score_mode,
            "cropper_conf": cropper_conf,
            "bbox_xyxy": [int(v) for v in box_px],
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
        }


def decode_upload_to_bgr(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("cv2.imdecode failed. The uploaded image may be invalid.")
    return frame


def create_app(args) -> FastAPI:
    app = FastAPI(title="GrimACE Lite Web Demo")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.runner = GrimaceRunner(
        model_def=Path(args.model_def).resolve(),
        quality_ckpt=Path(args.quality_ckpt).resolve(),
        cropper_ckpt=Path(args.cropper_ckpt).resolve(),
        grimace_ckpt=Path(args.grimace_ckpt).resolve(),
        device=args.device,
        score_mode=args.score_mode,
        frame_score_scale=args.frame_score_scale,
        crop_margin=args.crop_margin,
        min_crop_size=args.min_crop_size,
        grimace_color=args.grimace_color,
    )
    app.state.infer_lock = threading.Lock()
    app.state.video_path = Path(args.video).resolve() if args.video else None

    static_dir = Path(__file__).resolve().parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    def index():
        return FileResponse(static_dir / "index.html")

    @app.get("/config")
    def config():
        video_path = app.state.video_path
        has_video = bool(video_path and video_path.exists())
        return {
            "has_video": has_video,
            "video_url": "/video" if has_video else "",
            "score_mode": app.state.runner.score_mode,
            "organ_columns": app.state.runner.organ_columns,
            "device": str(app.state.runner.device),
        }

    @app.get("/video")
    def video():
        video_path = app.state.video_path
        if not video_path or not video_path.exists():
            return JSONResponse({"error": "No --video provided or file does not exist."}, status_code=404)
        return FileResponse(str(video_path))

    @app.post("/infer_frame")
    async def infer_frame(
        image: UploadFile = File(...),
        video_time: float = Form(0.0),
    ):
        acquired = app.state.infer_lock.acquire(blocking=False)
        if not acquired:
            return {
                "busy": True,
                "video_time": float(video_time),
                "message": "backend is still processing the previous frame; this frame was skipped",
            }

        try:
            data = await image.read()
            frame_bgr = decode_upload_to_bgr(data)
            result = app.state.runner.infer_bgr(frame_bgr)
            result["video_time"] = float(video_time)
            return result
        except Exception as e:
            return JSONResponse({"busy": False, "error": str(e)}, status_code=500)
        finally:
            app.state.infer_lock.release()

    return app


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-def", required=True, help="Path to aimodel_lite_models.py")
    parser.add_argument("--quality-ckpt", required=True, help="Frame quality checkpoint")
    parser.add_argument("--cropper-ckpt", required=True, help="Cropper checkpoint")
    parser.add_argument("--grimace-ckpt", required=True, help="Grimace scorer checkpoint")

    parser.add_argument("--video", default="", help="Optional video path to serve on the left side")
    parser.add_argument("--score-mode", choices=["class", "expected"], default="class")
    parser.add_argument("--frame-score-scale", type=float, default=6.0)
    parser.add_argument("--crop-margin", type=float, default=0.10)
    parser.add_argument("--min-crop-size", type=int, default=8)
    parser.add_argument("--grimace-color", choices=["bgr", "rgb"], default="bgr")
    parser.add_argument("--device", default="")

    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port)
