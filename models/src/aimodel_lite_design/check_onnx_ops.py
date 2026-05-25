#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
check_onnx_ops.py

Small ONNX op checker for the AIMODEL-friendly model exports.

Usage:
python check_onnx_ops.py onnx/frame_quality.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_ALLOWED = {
    # Core ops expected from our models
    "Conv",
    "BatchNormalization",
    "Relu",
    "MaxPool",
    "GlobalAveragePool",
    "Sigmoid",

    # Sometimes produced by exporters / optimizers
    "Identity",
    "Constant",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx_path")
    parser.add_argument(
        "--allowed",
        nargs="*",
        default=sorted(DEFAULT_ALLOWED),
        help="Allowed op types. Defaults to the conservative list used by this project.",
    )
    args = parser.parse_args()

    import onnx

    path = Path(args.onnx_path)
    model = onnx.load(str(path))
    onnx.checker.check_model(model)

    ops = sorted({node.op_type for node in model.graph.node})
    allowed = set(args.allowed)
    bad = [op for op in ops if op not in allowed]

    print(f"[Info] {path}")
    print("[Info] Ops:")
    for op in ops:
        print("  -", op)

    if bad:
        print("\n[Fail] Unsupported / unexpected ops:")
        for op in bad:
            print("  -", op)
        raise SystemExit(1)

    print("\n[OK] All ops are in the allowed list.")


if __name__ == "__main__":
    main()
