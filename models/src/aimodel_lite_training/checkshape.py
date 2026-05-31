#!/usr/bin/env python
# -*- coding: utf-8 -*-

import onnx
from onnx import shape_inference
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ONNX_FILES = [
    PROJECT_ROOT / "artifacts/onnx/frame_quality_lite.onnx",
    PROJECT_ROOT / "artifacts/onnx/face_cropper_lite.onnx",
    PROJECT_ROOT / "artifacts/onnx/grimace_scorer_lite.onnx",
]


def dim_to_str(d):
    if d.dim_value:
        return str(d.dim_value)
    if d.dim_param:
        return d.dim_param
    return "?"


def print_value_info(prefix, value_info):
    tensor_type = value_info.type.tensor_type
    shape = [dim_to_str(d) for d in tensor_type.shape.dim]
    elem_type = tensor_type.elem_type
    print(f"{prefix} name={value_info.name}")
    print(f"{prefix} shape={shape}")
    print(f"{prefix} elem_type={elem_type}")


for path in ONNX_FILES:
    path = Path(path)
    print("=" * 90)
    print(path)

    if not path.exists():
        print("ERROR: file not found")
        continue

    model = onnx.load(str(path))
    model = shape_inference.infer_shapes(model)

    print("\nInputs:")
    for inp in model.graph.input:
        print_value_info("  input ", inp)

    print("\nOutputs:")
    for out in model.graph.output:
        print_value_info("  output", out)

    print("\nOps:")
    ops = sorted({node.op_type for node in model.graph.node})
    print(" ", ops)