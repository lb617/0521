#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
export_aimodel_onnx.py

Export one of the three AIMODEL-friendly PyTorch models to ONNX.

Examples:

# Export randomly initialized quality model for conversion smoke test
python export_aimodel_onnx.py --model quality --out onnx/frame_quality.onnx

# Export trained weights
python export_aimodel_onnx.py \
  --model grimace \
  --weights runs/grimace_lite/best.pt \
  --out onnx/grimace_scorer.onnx

Recommended:
- Keep batch size static: N=1.
- Do not pass dynamic_axes.
- Use opset 11 or 12 unless the official converter requires another version.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch

from aimodel_lite_models import build_model, input_shape_for


def extract_state_dict(checkpoint) -> Dict[str, torch.Tensor]:
    """Handle common checkpoint formats."""
    if isinstance(checkpoint, dict):
        for key in ["state_dict", "model_state_dict", "model"]:
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break

    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint is not a state_dict or a dict containing a state_dict.")

    clean = {}
    for k, v in checkpoint.items():
        if not torch.is_tensor(v):
            continue
        if k.startswith("module."):
            k = k[len("module.") :]
        clean[k] = v
    return clean


def load_weights_if_needed(model: torch.nn.Module, weights: str | None, device: torch.device, strict: bool):
    if not weights:
        print("[Warn] No --weights provided. Exporting randomly initialized model.")
        return

    path = Path(weights)
    if not path.exists():
        raise FileNotFoundError(f"Weights not found: {path}")

    ckpt = torch.load(path, map_location=device)
    state = extract_state_dict(ckpt)
    missing, unexpected = model.load_state_dict(state, strict=strict)

    if missing:
        print("[Warn] Missing keys:")
        for k in missing:
            print("  -", k)
    if unexpected:
        print("[Warn] Unexpected keys:")
        for k in unexpected:
            print("  -", k)

    print(f"[Info] Loaded weights: {path}")


def export_onnx(args):
    device = torch.device(args.device)
    model = build_model(args.model).to(device)
    model.eval()

    load_weights_if_needed(model, args.weights, device=device, strict=args.strict)

    shape = input_shape_for(args.model)
    dummy = torch.randn(*shape, dtype=torch.float32, device=device)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        y = model(dummy)
    print("=" * 80)
    print(f"[Info] model       : {args.model}")
    print(f"[Info] input shape : {tuple(dummy.shape)}")
    print(f"[Info] output shape: {tuple(y.shape)}")
    print(f"[Info] opset       : {args.opset}")
    print(f"[Info] out         : {out_path}")
    print("=" * 80)

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        export_params=True,
        opset_version=int(args.opset),
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,
    )

    print(f"[Done] Exported ONNX: {out_path}")

    try:
        import onnx

        onnx_model = onnx.load(str(out_path))
        onnx.checker.check_model(onnx_model)
        ops = sorted({node.op_type for node in onnx_model.graph.node})
        print("[Info] ONNX checker passed.")
        print("[Info] ONNX ops:", ", ".join(ops))
    except Exception as e:
        print("[Warn] ONNX checker skipped or failed:", repr(e))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        required=True,
        choices=["quality", "cropper", "grimace"],
        help="Which model to export.",
    )
    p.add_argument("--weights", default="", help="Optional trained .pt/.pth checkpoint.")
    p.add_argument("--out", required=True, help="Output ONNX path.")
    p.add_argument("--device", default="cpu", help="cpu or cuda.")
    p.add_argument("--opset", type=int, default=11, help="ONNX opset. Try 11 first for conservative converters.")
    p.add_argument(
        "--strict",
        action="store_true",
        help="Use strict=True when loading weights. Default is non-strict for convenience.",
    )
    return p.parse_args()


if __name__ == "__main__":
    export_onnx(parse_args())
