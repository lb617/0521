#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_frame_quality_pearson.py

Evaluate trained LiteFrameQualityModel on the frame-quality dataset.

Main metric:
    Pearson correlation between ground-truth score 0~6 and prediction 0~6.

Important:
    The preprocessing is copied from train_frame_quality_lite.py:
    crop square -> resize 224 -> average channels -> /255 -> normalize
    with mean=0.6901 and std=0.1750.

Examples:

1) Evaluate the whole original dataset:
python eval_frame_quality_pearson.py \
  --model-def ./aimodel_lite_models.py \
  --ckpt runs/frame_quality_lite/frame_quality_lite_best.pt \
  --data-root . \
  --csv-glob "*_out.csv" \
  --out-dir runs/frame_quality_eval

2) Evaluate saved validation split:
python eval_frame_quality_pearson.py \
  --model-def ./aimodel_lite_models.py \
  --ckpt runs/frame_quality_lite/frame_quality_lite_best.pt \
  --split-csv runs/frame_quality_lite/val_split.csv \
  --out-dir runs/frame_quality_eval
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df


def add_suffix_candidates(image_name: str) -> List[str]:
    image_name = str(image_name).strip()
    p = Path(image_name)
    if p.suffix:
        return [image_name]
    return [image_name, image_name + ".png", image_name + ".jpg", image_name + ".jpeg"]


def load_module_from_path(py_path: Path):
    spec = importlib.util.spec_from_file_location("user_model_def", str(py_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load Python file: {py_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_quality_model(model_def_path: Path, device: torch.device) -> torch.nn.Module:
    module = load_module_from_path(model_def_path)
    if hasattr(module, "build_model"):
        model = module.build_model("quality")
    elif hasattr(module, "LiteFrameQualityModel"):
        model = module.LiteFrameQualityModel()
    else:
        raise RuntimeError("Model file must provide build_model('quality') or LiteFrameQualityModel.")
    return model.to(device).eval()


def load_state_dict_flexible(model: torch.nn.Module, ckpt_path: Path, device: torch.device):
    ckpt = torch.load(str(ckpt_path), map_location=device)
    if isinstance(ckpt, dict):
        for key in ["model_state_dict", "state_dict", "model"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    model.load_state_dict(ckpt, strict=True)


def crop_image_to_square(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    if h == w:
        return image
    d = abs(h - w)
    dl = d // 2
    dr = d - dl
    if h > w:
        return image[dl : h - dr, :]
    return image[:, dl : w - dr]


def average_channels(image: np.ndarray) -> np.ndarray:
    return np.mean(image, axis=2, keepdims=True)


def preprocess_frame_quality(frame: np.ndarray) -> torch.Tensor:
    frame = crop_image_to_square(frame)
    frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)

    if frame.ndim == 2:
        frame = frame[:, :, None]
    elif frame.shape[2] > 1:
        frame = average_channels(frame)

    frame = frame.astype(np.float32) / 255.0
    frame = (frame - 0.6901) / 0.1750
    frame = frame.transpose(2, 0, 1)
    return torch.tensor(frame, dtype=torch.float32)


def load_label_csvs(data_root: Path, csv_paths: Optional[List[str]], csv_glob: str) -> pd.DataFrame:
    if csv_paths:
        paths = []
        for item in csv_paths:
            p = Path(item)
            if not p.is_absolute():
                p = data_root / p
            paths.append(p)
    else:
        paths = sorted(data_root.glob(csv_glob))

    if not paths:
        raise FileNotFoundError(f"No CSV found. Looked for: {data_root / csv_glob}")

    dfs = []
    print("[Info] CSV files:")
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"CSV not found: {p}")
        df = pd.read_csv(p, encoding="utf-8-sig")
        df = clean_columns(df)
        required = ["video_name", "image_name", "score"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{p} missing columns: {missing}; got {list(df.columns)}")
        df = df[required].copy()
        df["source_csv"] = p.name
        df["csv_parent"] = str(p.parent)
        dfs.append(df)
        print(f"  - {p} | rows={len(df)}")

    return pd.concat(dfs, ignore_index=True)


def resolve_image_path(row: pd.Series, data_root: Path) -> Optional[Path]:
    video_name = str(row["video_name"]).strip()
    image_name = str(row["image_name"]).strip()
    csv_parent = Path(str(row.get("csv_parent", data_root)))

    candidates = []
    for nm in add_suffix_candidates(image_name):
        candidates.append(data_root / video_name / nm)
        candidates.append(csv_parent / video_name / nm)
        candidates.append(data_root / nm)
        candidates.append(csv_parent / nm)

    seen = set()
    for p in candidates:
        p = p.resolve()
        if p in seen:
            continue
        seen.add(p)
        if p.exists():
            return p
    return None


def prepare_dataframe_from_original_csvs(data_root: Path, csv_paths: Optional[List[str]], csv_glob: str) -> pd.DataFrame:
    df = load_label_csvs(data_root, csv_paths, csv_glob)
    df = df.dropna(subset=["video_name", "image_name", "score"]).copy()
    df["video_name"] = df["video_name"].astype(str).str.strip()
    df["image_name"] = df["image_name"].astype(str).str.strip()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"]).copy()
    df["score"] = df["score"].astype(float)

    before = len(df)
    df = df[(df["score"] >= 0) & (df["score"] <= 6)].copy()
    print(f"[Info] score range filter: {len(df)} / {before}")

    image_paths = []
    missing_examples = []
    for _, row in df.iterrows():
        p = resolve_image_path(row, data_root)
        if p is None:
            image_paths.append(None)
            if len(missing_examples) < 10:
                missing_examples.append(f"{row['video_name']}/{row['image_name']}")
        else:
            image_paths.append(str(p))

    df["image_path"] = image_paths
    before = len(df)
    df = df.dropna(subset=["image_path"]).copy()
    print(f"[Info] image path matched: {len(df)} / {before}")

    if missing_examples:
        print("[Warn] Missing image examples:")
        for x in missing_examples:
            print("  -", x)

    if len(df) == 0:
        raise RuntimeError("No valid quality samples found.")

    return df.reset_index(drop=True)


def prepare_dataframe_from_split_csv(split_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(split_csv, encoding="utf-8-sig")
    df = clean_columns(df)
    if "image_path" not in df.columns or "score" not in df.columns:
        raise ValueError(f"{split_csv} must contain image_path and score columns.")

    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["image_path", "score"]).copy()
    df["image_path"] = df["image_path"].astype(str)
    df["score"] = df["score"].astype(float)
    df = df[(df["score"] >= 0) & (df["score"] <= 6)].copy()

    exists = df["image_path"].map(lambda p: Path(str(p)).exists())
    if not exists.all():
        print("[Warn] Missing image examples from split CSV:")
        for p in df.loc[~exists, "image_path"].head(10).tolist():
            print("  -", p)
        df = df.loc[exists].copy()

    if len(df) == 0:
        raise RuntimeError("No valid samples found in split CSV.")

    return df.reset_index(drop=True)


class FrameQualityEvalDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.items = []
        for i, row in df.iterrows():
            self.items.append({
                "index": int(i),
                "image_path": str(row["image_path"]),
                "score": float(row["score"]),
                "video_name": str(row["video_name"]) if "video_name" in df.columns else "",
                "image_name": str(row["image_name"]) if "image_name" in df.columns else Path(str(row["image_path"])).name,
                "source_csv": str(row["source_csv"]) if "source_csv" in df.columns else "",
            })
        print(f"[Dataset] eval samples={len(self.items)}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        frame = cv2.imread(item["image_path"], cv2.IMREAD_UNCHANGED)
        if frame is None:
            raise RuntimeError(f"cv2.imread failed: {item['image_path']}")
        x = preprocess_frame_quality(frame)
        y = torch.tensor(item["score"] / 6.0, dtype=torch.float32)
        return x, y, item


def collate_fn(batch):
    xs, ys, metas = zip(*batch)
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0), list(metas)


def pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) < 2:
        return float("nan")
    if np.std(y_true) <= 1e-12 or np.std(y_pred) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot <= 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, pass_threshold: float):
    model.eval()
    rows = []
    y_true_all = []
    y_pred_all = []

    for x, y, metas in loader:
        x = x.to(device)
        y = y.to(device)

        pred01 = model(x).view(-1).clamp(0, 1)
        true6 = (y * 6.0).detach().cpu().numpy().astype(np.float64)
        pred6 = (pred01 * 6.0).detach().cpu().numpy().astype(np.float64)
        pred01_np = pred01.detach().cpu().numpy().astype(np.float64)

        for meta, yt, yp, yp01 in zip(metas, true6, pred6, pred01_np):
            err = float(yp - yt)
            rows.append({
                "index": meta["index"],
                "video_name": meta["video_name"],
                "image_name": meta["image_name"],
                "image_path": meta["image_path"],
                "source_csv": meta["source_csv"],
                "true_score_0_6": float(yt),
                "pred_score_0_6": float(yp),
                "pred_raw_0_1": float(yp01),
                "error": err,
                "abs_error": abs(err),
                "true_pass": int(yt >= pass_threshold),
                "pred_pass": int(yp >= pass_threshold),
            })

        y_true_all.extend(true6.tolist())
        y_pred_all.extend(pred6.tolist())

    y_true_np = np.asarray(y_true_all, dtype=np.float64)
    y_pred_np = np.asarray(y_pred_all, dtype=np.float64)
    err = y_pred_np - y_true_np

    metrics = {
        "n": int(len(y_true_np)),
        "pearson_r": pearson_corr(y_true_np, y_pred_np),
        "mae_0_6": float(np.mean(np.abs(err))) if len(err) else float("nan"),
        "rmse_0_6": float(np.sqrt(np.mean(err ** 2))) if len(err) else float("nan"),
        "bias_mean_pred_minus_true": float(np.mean(err)) if len(err) else float("nan"),
        "r2": r2_score(y_true_np, y_pred_np),
        "true_mean_0_6": float(np.mean(y_true_np)) if len(y_true_np) else float("nan"),
        "pred_mean_0_6": float(np.mean(y_pred_np)) if len(y_pred_np) else float("nan"),
        "true_std_0_6": float(np.std(y_true_np)) if len(y_true_np) else float("nan"),
        "pred_std_0_6": float(np.std(y_pred_np)) if len(y_pred_np) else float("nan"),
        "pass_threshold": float(pass_threshold),
        "pass_acc": float(np.mean((y_true_np >= pass_threshold) == (y_pred_np >= pass_threshold))) if len(y_true_np) else float("nan"),
    }

    return metrics, rows


def save_scatter_plot(rows: List[dict], out_path: Path):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[Warn] matplotlib not available, skip scatter plot: {e}")
        return

    y_true = np.asarray([r["true_score_0_6"] for r in rows], dtype=np.float64)
    y_pred = np.asarray([r["pred_score_0_6"] for r in rows], dtype=np.float64)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=14, alpha=0.55)
    plt.plot([0, 6], [0, 6], linestyle="--", linewidth=1)
    plt.xlim(-0.2, 6.2)
    plt.ylim(-0.2, 6.2)
    plt.xlabel("Ground Truth Score (0~6)")
    plt.ylabel("Predicted Score (0~6)")
    plt.title("Frame Quality: Prediction vs Ground Truth")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"[Saved] scatter plot: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate LiteFrameQualityModel with Pearson correlation.")
    parser.add_argument("--model-def", required=True, help="Path to aimodel_lite_models.py")
    parser.add_argument("--ckpt", required=True, help="Path to frame_quality_lite checkpoint")

    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--data-root", default=None, help="Root containing *_out.csv and image folders")
    data_group.add_argument("--split-csv", default=None, help="Saved train_split.csv or val_split.csv with image_path and score columns")

    parser.add_argument("--csv", nargs="*", default=None, help="Optional explicit CSV path(s), used with --data-root")
    parser.add_argument("--csv-glob", default="*_out.csv", help="CSV glob under --data-root")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pass-threshold", type=float, default=4.0)
    parser.add_argument("--device", default="", help="Default: cuda if available else cpu")
    parser.add_argument("--out-dir", default="runs/frame_quality_eval")
    parser.add_argument("--no-plot", action="store_true", help="Do not save scatter plot")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.split_csv:
        df = prepare_dataframe_from_split_csv(Path(args.split_csv).resolve())
        eval_name = Path(args.split_csv).stem
    else:
        data_root = Path(args.data_root).resolve()
        df = prepare_dataframe_from_original_csvs(data_root, args.csv, args.csv_glob)
        eval_name = "all_data"

    print("\n[Info] score distribution:")
    print(df["score"].round().astype(int).value_counts().sort_index())

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"\n[Info] device={device}")

    model = build_quality_model(Path(args.model_def).resolve(), device)
    load_state_dict_flexible(model, Path(args.ckpt).resolve(), device)

    loader = DataLoader(
        FrameQualityEvalDataset(df),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    metrics, rows = evaluate(model, loader, device, args.pass_threshold)

    print("\n========== Frame Quality Evaluation ==========")
    print(f"Samples       : {metrics['n']}")
    print(f"Pearson r     : {metrics['pearson_r']:.6f}")
    print(f"MAE  (0~6)    : {metrics['mae_0_6']:.6f}")
    print(f"RMSE (0~6)    : {metrics['rmse_0_6']:.6f}")
    print(f"Bias pred-true: {metrics['bias_mean_pred_minus_true']:.6f}")
    print(f"R^2           : {metrics['r2']:.6f}")
    print(f"Pass acc      : {metrics['pass_acc']:.6f}  threshold={metrics['pass_threshold']}")
    print("=============================================\n")

    pred_csv = out_dir / f"{eval_name}_predictions.csv"
    metrics_json = out_dir / f"{eval_name}_metrics.json"
    scatter_png = out_dir / f"{eval_name}_scatter.png"

    with pred_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with metrics_json.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[Saved] predictions: {pred_csv}")
    print(f"[Saved] metrics    : {metrics_json}")

    if not args.no_plot:
        save_scatter_plot(rows, scatter_png)


if __name__ == "__main__":
    main()
