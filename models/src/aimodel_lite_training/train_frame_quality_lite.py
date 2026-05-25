#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_frame_quality_lite.py

Train LiteFrameQualityModel on your original quality-label data.

Expected data layout:

data_root/
│  v0_out.csv
│  v1_out.csv
│  v3_out.csv
│
├─v0_out/
│      1.png
│      2.png
│      ...
├─v1_out/
│      1.png
│      2.png
│      ...
└─v3_out/
       1.png
       2.png
       ...

CSV columns:
    video_name,image_name,score

score is 0~6. The model outputs 0~1, so target = score / 6.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from aimodel_lite_models import LiteFrameQualityModel


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def prepare_dataframe(data_root: Path, csv_paths: Optional[List[str]], csv_glob: str) -> pd.DataFrame:
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

    print("\n[Info] valid samples by source_csv:")
    print(df.groupby("source_csv").size())
    print("\n[Info] score distribution:")
    print(df["score"].round().astype(int).value_counts().sort_index())

    return df.reset_index(drop=True)


class FrameQualityLiteDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.items = [(Path(r["image_path"]), float(r["score"])) for _, r in df.iterrows()]
        print(f"[Dataset] quality samples={len(self.items)}")

    @staticmethod
    def preprocess(frame: np.ndarray) -> torch.Tensor:
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

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, score = self.items[idx]
        frame = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if frame is None:
            raise RuntimeError(f"cv2.imread failed: {img_path}")
        x = self.preprocess(frame)
        y = torch.tensor(score / 6.0, dtype=torch.float32)
        score_class = torch.tensor(int(round(score)), dtype=torch.long)
        return x, y, score_class, str(img_path)


def stratified_split(df: pd.DataFrame, val_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["score_class"] = df["score"].round().astype(int)

    train_parts, val_parts = [], []
    for score_class, g in df.groupby("score_class"):
        idx = np.array(g.index)
        rng.shuffle(idx)
        if len(idx) <= 1:
            n_val = 0
        else:
            n_val = max(1, int(round(len(idx) * val_ratio)))
            n_val = min(n_val, len(idx) - 1)
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
        train_parts.append(df.loc[train_idx])
        if len(val_idx) > 0:
            val_parts.append(df.loc[val_idx])
        print(f"[Split] score={score_class}: total={len(idx)}, train={len(train_idx)}, val={len(val_idx)}")

    train_df = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if not val_parts:
        raise RuntimeError("Validation set is empty.")
    val_df = pd.concat(val_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return train_df.drop(columns=["score_class"]), val_df.drop(columns=["score_class"])


def make_class_weights(train_df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    counts = train_df["score"].round().astype(int).value_counts().to_dict()
    weights = []
    for c in range(7):
        cnt = counts.get(c, 0)
        weights.append(0.0 if cnt <= 0 else 1.0 / np.sqrt(cnt))
    weights = np.array(weights, dtype=np.float32)
    nz = weights[weights > 0]
    if len(nz) > 0:
        weights = weights / nz.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


@torch.no_grad()
def evaluate(model, loader, device, pass_threshold: float):
    model.eval()
    losses, abs_errors, sq_errors = [], [], []
    pass_correct, total = 0, 0
    y_true_all, y_pred_all = [], []

    for x, y, score_class, paths in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x).view(-1).clamp(0, 1)
        loss = F.smooth_l1_loss(pred, y, reduction="mean")
        losses.append(loss.item())

        pred6 = pred * 6.0
        true6 = y * 6.0
        err = pred6 - true6
        abs_errors.extend(torch.abs(err).cpu().numpy().tolist())
        sq_errors.extend((err ** 2).cpu().numpy().tolist())

        pass_correct += ((pred6 >= pass_threshold) == (true6 >= pass_threshold)).sum().item()
        total += y.numel()
        y_true_all.extend(true6.cpu().numpy().tolist())
        y_pred_all.extend(pred6.cpu().numpy().tolist())

    mae = float(np.mean(abs_errors)) if abs_errors else 999.0
    rmse = float(np.sqrt(np.mean(sq_errors))) if sq_errors else 999.0
    corr = 0.0
    if len(y_true_all) >= 2 and np.std(y_true_all) > 1e-8 and np.std(y_pred_all) > 1e-8:
        corr = float(np.corrcoef(y_true_all, y_pred_all)[0, 1])

    return {
        "loss": float(np.mean(losses)) if losses else 999.0,
        "mae_0_6": mae,
        "rmse_0_6": rmse,
        "pass_acc": pass_correct / max(1, total),
        "corr": corr,
    }


def train_one_epoch(model, loader, optimizer, device, class_weights: Optional[torch.Tensor]):
    model.train()
    losses = []
    for x, y, score_class, paths in loader:
        x = x.to(device)
        y = y.to(device)
        score_class = score_class.to(device)

        pred = model(x).view(-1).clamp(0, 1)
        loss_vec = F.smooth_l1_loss(pred, y, reduction="none")

        if class_weights is not None:
            w = class_weights[score_class]
            loss = (loss_vec * w).mean()
        else:
            loss = loss_vec.mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return float(np.mean(losses))


def load_state_dict_flexible(model, path: Path, device):
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict):
        for key in ["model_state_dict", "state_dict", "model"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    model.load_state_dict(ckpt, strict=True)


def save_checkpoint(path: Path, model, optimizer, epoch: int, args, metrics: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": "quality",
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "metrics": metrics,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=".", help="Root containing *_out.csv and image folders.")
    parser.add_argument("--csv", nargs="*", default=None, help="Optional explicit CSV path(s).")
    parser.add_argument("--csv_glob", default="*_out.csv")
    parser.add_argument("--out_dir", default="runs/frame_quality_lite")
    parser.add_argument("--resume", default="")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weighted_loss", action="store_true")
    parser.add_argument("--pass_threshold", type=float, default=4.0)
    parser.add_argument("--device", default="", help="Default: cuda if available else cpu.")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = prepare_dataframe(data_root, args.csv, args.csv_glob)
    if args.dry_run:
        print("[Dry Run] Data parsing OK. Not training.")
        return

    train_df, val_df = stratified_split(df, args.val_ratio, args.seed)
    train_df.to_csv(out_dir / "train_split.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(out_dir / "val_split.csv", index=False, encoding="utf-8-sig")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = LiteFrameQualityModel().to(device)

    if args.resume:
        load_state_dict_flexible(model, Path(args.resume), device)
        print(f"[Info] Resumed model weights: {args.resume}")

    train_loader = DataLoader(
        FrameQualityLiteDataset(train_df),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        FrameQualityLiteDataset(val_df),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    class_weights = make_class_weights(train_df, device) if args.weighted_loss else None

    (out_dir / "model_config.json").write_text(
        json.dumps({"model": "LiteFrameQualityModel", "input_shape": [1, 1, 224, 224], "args": vars(args)}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    best_mae = 999.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, class_weights)
        metrics = evaluate(model, val_loader, device, args.pass_threshold)
        scheduler.step()

        print(
            f"Epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.5f} | "
            f"val_loss={metrics['loss']:.5f} | mae={metrics['mae_0_6']:.4f} | "
            f"rmse={metrics['rmse_0_6']:.4f} | pass_acc={metrics['pass_acc']:.4f} | corr={metrics['corr']:.4f}"
        )

        save_checkpoint(out_dir / "frame_quality_lite_last.pt", model, optimizer, epoch, args, metrics)
        if metrics["mae_0_6"] < best_mae:
            best_mae = metrics["mae_0_6"]
            save_checkpoint(out_dir / "frame_quality_lite_best.pt", model, optimizer, epoch, args, metrics)
            print(f"  -> saved best: {out_dir / 'frame_quality_lite_best.pt'} | mae={best_mae:.4f}")

    print(f"[Done] best weight: {out_dir / 'frame_quality_lite_best.pt'}")


if __name__ == "__main__":
    main()
