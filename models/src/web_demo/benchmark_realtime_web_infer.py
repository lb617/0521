#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
benchmark_realtime_web_infer.py

Benchmark the real-time web inference backend for MGS-PainLens Lite.

It sends video frames to the running FastAPI backend endpoint /infer_frame
at a fixed frame rate, without waiting for previous requests to finish.

Metrics:
1. Real-time ratio:
       R = FPS_app / FPS_sensor
   where FPS_app = scored frames per second, and FPS_sensor = original video FPS.

2. End-to-end latency:
       latency_ms
   client-side wall time from sending a frame to receiving backend response.

3. Skip / busy rate:
       busy_count / send_count
   ratio of requests for which backend returns {"busy": true}.

4. Parameter count:
   optional; if --model-def is provided, the script counts parameters for
   quality, cropper, and grimace models.

Usage example:

python benchmark_realtime_web_infer.py \
  --server-url http://127.0.0.1:6006 \
  --video data/v1.mp4 \
  --duration 60 \
  --model-def ./aimodel_lite_models.py \
  --output-dir runs/realtime_benchmark

If you want to stress test at original video FPS, keep --send-fps 0.
If you only want to test 5 frames per second:

python benchmark_realtime_web_infer.py \
  --server-url http://127.0.0.1:6006 \
  --video data/v1.mp4 \
  --send-fps 5 \
  --duration 60 \
  --model-def ./aimodel_lite_models.py
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin

import cv2
import numpy as np


def require_requests():
    try:
        import requests
        return requests
    except Exception as e:
        raise RuntimeError(
            "This script needs the 'requests' package. Install it with: pip install requests"
        ) from e


def load_module_from_path(py_path: Path):
    spec = importlib.util.spec_from_file_location("user_model_def", str(py_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load Python file: {py_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_model_from_module(module, model_name: str):
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


def count_model_params(model) -> Dict[str, int]:
    total = sum(int(p.numel()) for p in model.parameters())
    trainable = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": trainable}


def count_all_params(model_def: Optional[str]) -> Dict:
    if not model_def:
        return {"enabled": False}

    module = load_module_from_path(Path(model_def).resolve())
    result = {"enabled": True, "models": {}, "total_params": 0, "trainable_params": 0}

    for name in ["quality", "cropper", "grimace"]:
        model = build_model_from_module(module, name)
        c = count_model_params(model)
        result["models"][name] = c
        result["total_params"] += c["total_params"]
        result["trainable_params"] += c["trainable_params"]

    return result


def percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    arr = sorted(values)
    if len(arr) == 1:
        return float(arr[0])
    k = (len(arr) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(arr[int(k)])
    return float(arr[f] * (c - k) + arr[c] * (k - f))


def summarize_values(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }

    return {
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def resize_for_upload(frame_bgr: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return frame_bgr

    h, w = frame_bgr.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return frame_bgr

    scale = max_side / float(side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def encode_jpeg(frame_bgr: np.ndarray, jpeg_quality: int) -> bytes:
    jpeg_quality = int(np.clip(jpeg_quality, 1, 100))
    ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not ok:
        raise RuntimeError("cv2.imencode('.jpg') failed")
    return encoded.tobytes()


def post_frame(
    request_id: int,
    frame_index: int,
    video_time: float,
    image_bytes: bytes,
    infer_url: str,
    timeout: float,
):
    requests = require_requests()

    t0 = time.perf_counter()
    row = {
        "request_id": int(request_id),
        "frame_index": int(frame_index),
        "video_time": float(video_time),
        "send_t_perf": float(t0),
        "status_code": None,
        "busy": None,
        "ok": False,
        "client_latency_ms": None,
        "backend_latency_ms": None,
        "quality_score": None,
        "error": "",
    }

    try:
        files = {"image": ("frame.jpg", image_bytes, "image/jpeg")}
        data = {"video_time": str(video_time)}
        resp = requests.post(infer_url, files=files, data=data, timeout=timeout)
        t1 = time.perf_counter()

        row["status_code"] = int(resp.status_code)
        row["client_latency_ms"] = float((t1 - t0) * 1000.0)

        try:
            payload = resp.json()
        except Exception:
            payload = {"error": resp.text[:300]}

        busy = bool(payload.get("busy", False))
        row["busy"] = int(busy)
        row["backend_latency_ms"] = payload.get("latency_ms", None)
        row["quality_score"] = payload.get("quality_score", payload.get("frame_score", None))

        if resp.status_code == 200 and not busy and "error" not in payload:
            row["ok"] = True
        else:
            row["ok"] = False
            if "error" in payload:
                row["error"] = str(payload["error"])
            elif busy:
                row["error"] = "busy"
            else:
                row["error"] = f"http_status_{resp.status_code}"

    except Exception as e:
        t1 = time.perf_counter()
        row["client_latency_ms"] = float((t1 - t0) * 1000.0)
        row["busy"] = 0
        row["ok"] = False
        row["error"] = repr(e)

    return row


def compute_step_and_send_fps(video_fps: float, requested_send_fps: float) -> tuple[int, float]:
    if video_fps <= 1e-6:
        raise RuntimeError("Cannot read valid video FPS from input video.")

    if requested_send_fps <= 0:
        return 1, float(video_fps)

    requested_send_fps = min(float(requested_send_fps), float(video_fps))
    step = max(1, int(round(video_fps / requested_send_fps)))
    actual_send_fps = float(video_fps) / float(step)
    return step, actual_send_fps


def run_benchmark(args) -> tuple[Dict, List[Dict]]:
    cap = cv2.VideoCapture(str(Path(args.video).resolve()))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_video = total_frames / video_fps if video_fps > 1e-6 and total_frames > 0 else float("nan")

    step, actual_send_fps = compute_step_and_send_fps(video_fps, args.send_fps)
    frame_interval_sec = step / video_fps

    base_url = args.server_url.rstrip("/") + "/"
    infer_url = urljoin(base_url, "infer_frame")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[Info] benchmark target")
    print(f"  server_url       : {args.server_url}")
    print(f"  infer_url        : {infer_url}")
    print(f"  video            : {Path(args.video).resolve()}")
    print(f"  FPS_sensor       : {video_fps:.6f}")
    print(f"  total_frames     : {total_frames}")
    print(f"  video_duration   : {duration_video:.3f}s")
    print(f"  frame_step       : {step}")
    print(f"  send_fps_actual  : {actual_send_fps:.6f}")
    print(f"  test_duration    : {args.duration if args.duration > 0 else 'full video'}s")
    print(f"  max_side         : {args.max_side}")
    print(f"  jpeg_quality     : {args.jpeg_quality}")
    print()

    futures = []
    rows: List[Dict] = []

    request_id = 0
    first_send_perf = None
    last_send_perf = None
    send_loop_start = time.perf_counter()

    max_test_video_time = float(args.duration) if args.duration > 0 else float("inf")

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        frame_index = -1

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_index += 1
            if frame_index % step != 0:
                continue

            video_time = frame_index / video_fps
            if video_time > max_test_video_time:
                break

            # Realtime scheduling: send this frame at its video timestamp.
            target_send_perf = send_loop_start + video_time
            now = time.perf_counter()
            sleep_sec = target_send_perf - now
            if sleep_sec > 0:
                time.sleep(sleep_sec)

            upload_frame = resize_for_upload(frame, args.max_side)
            image_bytes = encode_jpeg(upload_frame, args.jpeg_quality)

            send_perf = time.perf_counter()
            if first_send_perf is None:
                first_send_perf = send_perf
            last_send_perf = send_perf

            futures.append(
                executor.submit(
                    post_frame,
                    request_id,
                    frame_index,
                    video_time,
                    image_bytes,
                    infer_url,
                    args.timeout,
                )
            )
            request_id += 1

        cap.release()

        print(f"[Info] all requests submitted: {request_id}")
        for future in as_completed(futures):
            rows.append(future.result())

    end_perf = time.perf_counter()

    rows.sort(key=lambda r: r["request_id"])

    if first_send_perf is None:
        raise RuntimeError("No frames were sent. Check video and duration settings.")

    send_count = len(rows)
    busy_count = sum(1 for r in rows if int(r.get("busy") or 0) == 1)
    ok_count = sum(1 for r in rows if bool(r.get("ok")))
    error_count = send_count - busy_count - ok_count

    total_wall_time_sec = max(1e-9, end_perf - first_send_perf)
    send_window_sec = max(1e-9, (last_send_perf or first_send_perf) - first_send_perf)
    fps_app = ok_count / total_wall_time_sec
    fps_app_send_window = ok_count / send_window_sec if send_window_sec > 1e-9 else float("nan")

    client_lat_all = [
        float(r["client_latency_ms"])
        for r in rows
        if r.get("client_latency_ms") is not None and math.isfinite(float(r["client_latency_ms"]))
    ]
    client_lat_ok = [
        float(r["client_latency_ms"])
        for r in rows
        if r.get("ok") and r.get("client_latency_ms") is not None and math.isfinite(float(r["client_latency_ms"]))
    ]
    backend_lat_ok = [
        float(r["backend_latency_ms"])
        for r in rows
        if r.get("ok") and r.get("backend_latency_ms") is not None and math.isfinite(float(r["backend_latency_ms"]))
    ]

    params = count_all_params(args.model_def)

    metrics = {
        "video": str(Path(args.video).resolve()),
        "server_url": args.server_url,
        "infer_url": infer_url,
        "FPS_sensor": float(video_fps),
        "send_fps_requested": float(args.send_fps),
        "send_fps_actual": float(actual_send_fps),
        "frame_step": int(step),
        "duration_setting_sec": float(args.duration),
        "send_count": int(send_count),
        "ok_count": int(ok_count),
        "busy_count": int(busy_count),
        "error_count": int(error_count),
        "busy_rate": float(busy_count / send_count) if send_count else float("nan"),
        "score_success_rate": float(ok_count / send_count) if send_count else float("nan"),
        "total_wall_time_sec": float(total_wall_time_sec),
        "send_window_sec": float(send_window_sec),
        "FPS_app": float(fps_app),
        "FPS_app_send_window": float(fps_app_send_window),
        "R_FPS_app_over_FPS_sensor": float(fps_app / video_fps) if video_fps > 1e-6 else float("nan"),
        "R_FPS_app_over_send_fps": float(fps_app / actual_send_fps) if actual_send_fps > 1e-6 else float("nan"),
        "client_latency_ms_all": summarize_values(client_lat_all),
        "client_latency_ms_ok": summarize_values(client_lat_ok),
        "backend_latency_ms_ok": summarize_values(backend_lat_ok),
        "params": params,
        "settings": {
            "max_side": int(args.max_side),
            "jpeg_quality": int(args.jpeg_quality),
            "timeout": float(args.timeout),
            "max_workers": int(args.max_workers),
        },
    }

    return metrics, rows


def save_outputs(metrics: Dict, rows: List[Dict], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "realtime_benchmark_metrics.json"
    rows_path = output_dir / "realtime_benchmark_requests.csv"

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    if rows:
        fieldnames = [
            "request_id",
            "frame_index",
            "video_time",
            "send_t_perf",
            "status_code",
            "busy",
            "ok",
            "client_latency_ms",
            "backend_latency_ms",
            "quality_score",
            "error",
        ]
        with rows_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"[Saved] metrics : {metrics_path}")
    print(f"[Saved] requests: {rows_path}")


def print_summary(metrics: Dict):
    print()
    print("========== Realtime Inference Benchmark ==========")
    print(f"FPS_sensor              : {metrics['FPS_sensor']:.4f}")
    print(f"send_fps_actual          : {metrics['send_fps_actual']:.4f}")
    print(f"send_count               : {metrics['send_count']}")
    print(f"ok_count/scored_count    : {metrics['ok_count']}")
    print(f"busy_count               : {metrics['busy_count']}")
    print(f"error_count              : {metrics['error_count']}")
    print(f"FPS_app                  : {metrics['FPS_app']:.4f}")
    print(f"R = FPS_app/FPS_sensor   : {metrics['R_FPS_app_over_FPS_sensor']:.4f}")
    print(f"R_vs_send_fps            : {metrics['R_FPS_app_over_send_fps']:.4f}")
    print(f"busy_rate                : {metrics['busy_rate']:.4f}")
    print(f"score_success_rate       : {metrics['score_success_rate']:.4f}")

    lat = metrics["client_latency_ms_ok"]
    print()
    print("Client end-to-end latency, OK responses only:")
    print(f"  mean   : {lat['mean']:.3f} ms")
    print(f"  median : {lat['median']:.3f} ms")
    print(f"  p90    : {lat['p90']:.3f} ms")
    print(f"  p95    : {lat['p95']:.3f} ms")
    print(f"  max    : {lat['max']:.3f} ms")

    blat = metrics["backend_latency_ms_ok"]
    print()
    print("Backend model latency from server response, OK responses only:")
    print(f"  mean   : {blat['mean']:.3f} ms")
    print(f"  median : {blat['median']:.3f} ms")
    print(f"  p90    : {blat['p90']:.3f} ms")
    print(f"  p95    : {blat['p95']:.3f} ms")
    print(f"  max    : {blat['max']:.3f} ms")

    params = metrics.get("params", {})
    if params.get("enabled"):
        print()
        print("Parameter count:")
        for name, item in params["models"].items():
            print(f"  {name:8s}: total={item['total_params']:,}, trainable={item['trainable_params']:,}")
        print(f"  {'total':8s}: total={params['total_params']:,}, trainable={params['trainable_params']:,}")
    else:
        print()
        print("Parameter count: skipped. Provide --model-def to enable.")

    print("==================================================")
    print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark real-time video inference through /infer_frame.")
    parser.add_argument("--server-url", default="http://127.0.0.1:6006", help="Base URL of running web demo")
    parser.add_argument("--video", required=True, help="Video file used for sending frames")
    parser.add_argument("--duration", type=float, default=30.0, help="Benchmark duration in video seconds; <=0 means full video")
    parser.add_argument("--send-fps", type=float, default=0.0, help="Frame sending FPS; 0 means original video FPS")
    parser.add_argument("--max-side", type=int, default=960, help="Resize uploaded frame so max side <= this; <=0 disables")
    parser.add_argument("--jpeg-quality", type=int, default=82, help="JPEG quality for upload")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP request timeout in seconds")
    parser.add_argument("--max-workers", type=int, default=64, help="Max concurrent HTTP requests")
    parser.add_argument("--model-def", default="", help="Optional aimodel_lite_models.py path for parameter count")
    parser.add_argument("--output-dir", default="runs/realtime_benchmark", help="Directory to save metrics and per-request log")
    args = parser.parse_args()

    metrics, rows = run_benchmark(args)
    output_dir = Path(args.output_dir).resolve()
    save_outputs(metrics, rows, output_dir)
    print_summary(metrics)


if __name__ == "__main__":
    main()
