#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_ALLOWED = {
    "Conv",
    "BatchNormalization",
    "Relu",
    "MaxPool",
    "GlobalAveragePool",
    "Sigmoid",
    "Identity",
    "Constant",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx_path")
    parser.add_argument("--allowed", nargs="*", default=sorted(DEFAULT_ALLOWED))
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
        print("\n[Fail] Unexpected ops:")
        for op in bad:
            print("  -", op)
        raise SystemExit(1)
    print("\n[OK] All ops are in the allowed list.")


if __name__ == "__main__":
    main()
