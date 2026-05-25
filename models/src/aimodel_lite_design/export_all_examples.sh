#!/usr/bin/env bash
set -euo pipefail

mkdir -p onnx

# Smoke-test exports without trained weights.
# Replace --weights paths after training the new lightweight models.

python export_aimodel_onnx.py --model quality --out onnx/frame_quality_lite.onnx --device cpu --opset 11
python export_aimodel_onnx.py --model cropper --out onnx/face_cropper_lite.onnx --device cpu --opset 11
python export_aimodel_onnx.py --model grimace --out onnx/grimace_scorer_lite.onnx --device cpu --opset 11

python check_onnx_ops.py onnx/frame_quality_lite.onnx
python check_onnx_ops.py onnx/face_cropper_lite.onnx
python check_onnx_ops.py onnx/grimace_scorer_lite.onnx
