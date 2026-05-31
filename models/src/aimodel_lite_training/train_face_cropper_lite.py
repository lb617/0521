#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_face_cropper_lite.py

Train LiteFaceCropperModel on your original YOLO face labels.

Expected data layout:

data_root/
    output.txt
    v0_000320.png
    v0_000320.txt
    v0_000325.png
    v0_000325.txt
    ...

YOLO label format:
    class_id x_center y_center width height

The new cropper target is:
    conf, x1, y1, x2, y2
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from aimodel_lite_models import LiteFaceCropperModel


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_group_key(path: Path) -> str:
    stem = path.stem
    if "_" in stem:
        return stem.split("_")[0]
    return "default"


def read_text_safely(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return path.read_text(encoding="gbk")


def find_images(data_root: Path, image_glob: str = "", recursive: bool = False) -> List[Path]:
    if image_glob:
        paths = list(data_root.rglob(image_glob)) if recursive else list(data_root.glob(image_glob))
        return sorted(set([p for p in paths if p.is_file() and p.suffix.lower() in IMG_EXTS]))

    paths = []
    for ext in IMG_EXTS:
        if recursive:
            paths.extend(data_root.rglob(f"*{ext}"))
            paths.extend(data_root.rglob(f"*{ext.upper()}"))
        else:
            paths.extend(data_root.glob(f"*{ext}"))
            paths.extend(data_root.glob(f"*{ext.upper()}"))

    return sorted(set([p for p in paths if p.is_file() and p.suffix.lower() in IMG_EXTS]))


def parse_yolo_label_to_xyxy(label_path: Path) -> Optional[np.ndarray]:
    """Return [x1, y1, x2, y2] normalized. If multiple boxes exist, choose largest area."""
    text = read_text_safely(label_path).strip()
    if text == "":
        return None

    boxes = []
    for line_idx, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{label_path.name} line {line_idx}: need 5 columns, got {len(parts)}")
        try:
            cls = float(parts[0])
            xc, yc, bw, bh = map(float, parts[1:])
        except ValueError as e:
            raise ValueError(f"{label_path.name} line {line_idx}: non-numeric value") from e
        if not all(0.0 <= v <= 1.0 for v in [xc, yc, bw, bh]):
            raise ValueError(f"{label_path.name} line {line_idx}: xywh not in [0,1]")
        if bw <= 0 or bh <= 0:
            raise ValueError(f"{label_path.name} line {line_idx}: width/height <= 0")

        x1 = max(0.0, xc - bw / 2.0)
        y1 = max(0.0, yc - bh / 2.0)
        x2 = min(1.0, xc + bw / 2.0)
        y2 = min(1.0, yc + bh / 2.0)
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        boxes.append((area, np.array([x1, y1, x2, y2], dtype=np.float32)))

    if not boxes:
        return None
    boxes.sort(key=lambda x: x[0], reverse=True)
    return boxes[0][1]


def collect_samples(
    data_root: Path,
    image_glob: str,
    recursive: bool,
    include_unlabeled_as_negative: bool,
    include_empty_label_as_negative: bool,
    negative_root: str,
) -> pd.DataFrame:
    rows = []
    invalid_examples = []
    image_paths = find_images(data_root, image_glob, recursive)
    print(f"[Scan] images under data_root: {len(image_paths)}")

    for img_path in image_paths:
        label_path = img_path.with_suffix(".txt")
        if not label_path.exists():
            if include_unlabeled_as_negative:
                rows.append({"image_path": str(img_path.resolve()), "label_path": "", "conf": 0.0, "x1": 0.0, "y1": 0.0, "x2": 0.0, "y2": 0.0, "source": "unlabeled_negative", "group": get_group_key(img_path)})
            continue

        try:
            xyxy = parse_yolo_label_to_xyxy(label_path)
            if xyxy is None:
                if include_empty_label_as_negative:
                    rows.append({"image_path": str(img_path.resolve()), "label_path": str(label_path.resolve()), "conf": 0.0, "x1": 0.0, "y1": 0.0, "x2": 0.0, "y2": 0.0, "source": "empty_label_negative", "group": get_group_key(img_path)})
                else:
                    if len(invalid_examples) < 20:
                        invalid_examples.append((label_path.name, "empty label"))
                continue

            rows.append({"image_path": str(img_path.resolve()), "label_path": str(label_path.resolve()), "conf": 1.0, "x1": float(xyxy[0]), "y1": float(xyxy[1]), "x2": float(xyxy[2]), "y2": float(xyxy[3]), "source": "yolo_positive", "group": get_group_key(img_path)})
        except Exception as e:
            if len(invalid_examples) < 20:
                invalid_examples.append((label_path.name, str(e)))

    if negative_root:
        neg_root = Path(negative_root).resolve()
        neg_imgs = find_images(neg_root, "", True)
        print(f"[Scan] explicit negative images: {len(neg_imgs)} from {neg_root}")
        for img_path in neg_imgs:
            rows.append({"image_path": str(img_path.resolve()), "label_path": "", "conf": 0.0, "x1": 0.0, "y1": 0.0, "x2": 0.0, "y2": 0.0, "source": "explicit_negative", "group": get_group_key(img_path)})

    if invalid_examples:
        print("[Warn] invalid label examples:")
        for name, reason in invalid_examples:
            print(f"  - {name}: {reason}")

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise RuntimeError("No cropper samples found.")

    print("\n[Info] sample counts:")
    print(df["source"].value_counts())
    print("\n[Info] group counts:")
    print(df.groupby(["group", "source"]).size())
    return df.reset_index(drop=True)


def split_by_group(df: pd.DataFrame, val_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    train_parts, val_parts = [], []
    for group, g in df.groupby("group"):
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
        print(f"[Split] {group}: total={len(idx)}, train={len(train_idx)}, val={len(val_idx)}")
    train_df = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if not val_parts:
        raise RuntimeError("Validation set is empty.")
    val_df = pd.concat(val_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train_df, val_df


class FaceCropperLiteDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.items = []
        for _, r in df.iterrows():
            target = np.array([r["conf"], r["x1"], r["y1"], r["x2"], r["y2"]], dtype=np.float32)
            self.items.append((Path(r["image_path"]), target))
        print(f"[Dataset] cropper samples={len(self.items)}")

    @staticmethod
    def preprocess(frame: np.ndarray) -> torch.Tensor:
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LINEAR)
        frame = frame.astype(np.float32) / 255.0
        frame = frame.transpose(2, 0, 1)
        return torch.tensor(frame, dtype=torch.float32)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, target = self.items[idx]
        frame = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if frame is None:
            raise RuntimeError(f"cv2.imread failed: {img_path}")
        x = self.preprocess(frame)
        y = torch.tensor(target, dtype=torch.float32)
        return x, y, str(img_path)


def bbox_iou_xyxy(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    x1 = torch.maximum(a[:, 0], b[:, 0])
    y1 = torch.maximum(a[:, 1], b[:, 1])
    x2 = torch.minimum(a[:, 2], b[:, 2])
    y2 = torch.minimum(a[:, 3], b[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    return inter / (area_a + area_b - inter + eps)


def sanitize_xyxy(box: torch.Tensor) -> torch.Tensor:
    x1 = torch.minimum(box[:, 0], box[:, 2])
    y1 = torch.minimum(box[:, 1], box[:, 3])
    x2 = torch.maximum(box[:, 0], box[:, 2])
    y2 = torch.maximum(box[:, 1], box[:, 3])
    return torch.stack([x1, y1, x2, y2], dim=1).clamp(0, 1)


def compute_loss(pred: torch.Tensor, target: torch.Tensor, bbox_loss_weight: float, conf_loss_weight: float):
    pred_conf = pred[:, 0].clamp(1e-6, 1 - 1e-6)
    pred_box = sanitize_xyxy(pred[:, 1:5])
    target_conf = target[:, 0]
    target_box = target[:, 1:5]
    conf_loss = F.binary_cross_entropy(pred_conf, target_conf, reduction="mean")
    pos_mask = target_conf > 0.5
    if pos_mask.any():
        bbox_loss = F.smooth_l1_loss(pred_box[pos_mask], target_box[pos_mask], reduction="mean")
    else:
        bbox_loss = pred_box.sum() * 0.0
    return conf_loss_weight * conf_loss + bbox_loss_weight * bbox_loss, conf_loss.detach(), bbox_loss.detach()


@torch.no_grad()
def evaluate(model, loader, device, conf_threshold: float, bbox_loss_weight: float, conf_loss_weight: float):
    model.eval()
    losses, conf_losses, bbox_losses, ious = [], [], [], []
    conf_correct, total = 0, 0
    for x, y, paths in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x).flatten(1)
        loss, conf_loss, bbox_loss = compute_loss(pred, y, bbox_loss_weight, conf_loss_weight)
        losses.append(loss.item())
        conf_losses.append(conf_loss.item())
        bbox_losses.append(bbox_loss.item())
        pred_conf_label = pred[:, 0] >= conf_threshold
        true_conf_label = y[:, 0] > 0.5
        conf_correct += (pred_conf_label == true_conf_label).sum().item()
        total += y.shape[0]
        pos = true_conf_label
        if pos.any():
            pred_box = sanitize_xyxy(pred[:, 1:5])
            iou = bbox_iou_xyxy(pred_box[pos], y[:, 1:5][pos])
            ious.extend(iou.cpu().numpy().tolist())
    return {
        "loss": float(np.mean(losses)) if losses else 999.0,
        "conf_loss": float(np.mean(conf_losses)) if conf_losses else 999.0,
        "bbox_loss": float(np.mean(bbox_losses)) if bbox_losses else 999.0,
        "mean_iou_pos": float(np.mean(ious)) if ious else 0.0,
        "median_iou_pos": float(np.median(ious)) if ious else 0.0,
        "conf_acc": conf_correct / max(1, total),
    }


def train_one_epoch(model, loader, optimizer, device, bbox_loss_weight: float, conf_loss_weight: float):
    model.train()
    losses = []
    for x, y, paths in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x).flatten(1)
        loss, _, _ = compute_loss(pred, y, bbox_loss_weight, conf_loss_weight)
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
    torch.save({"model_name": "cropper", "epoch": int(epoch), "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "args": vars(args), "metrics": metrics}, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, help="Folder containing flat images and same-name YOLO txt labels.")
    parser.add_argument("--image_glob", default="", help="Optional image glob, e.g. v0_*.png")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--negative_root", default="", help="Optional folder of no-face negative images.")
    parser.add_argument("--include_unlabeled_as_negative", action="store_true", help="Treat images without txt as negative samples.")
    parser.add_argument("--include_empty_label_as_negative", action="store_true", help="Treat empty txt as negative samples.")
    parser.add_argument("--out_dir", default="runs/face_cropper_lite")
    parser.add_argument("--resume", default="")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bbox_loss_weight", type=float, default=5.0)
    parser.add_argument("--conf_loss_weight", type=float, default=1.0)
    parser.add_argument("--conf_threshold", type=float, default=0.5)
    parser.add_argument("--device", default="")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = collect_samples(data_root, args.image_glob, args.recursive, args.include_unlabeled_as_negative, args.include_empty_label_as_negative, args.negative_root)
    if args.dry_run:
        print("[Dry Run] Data parsing OK. Not training.")
        return

    train_df, val_df = split_by_group(df, args.val_ratio, args.seed)
    train_df.to_csv(out_dir / "train_split.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(out_dir / "val_split.csv", index=False, encoding="utf-8-sig")
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = LiteFaceCropperModel().to(device)
    if args.resume:
        load_state_dict_flexible(model, Path(args.resume), device)
        print(f"[Info] Resumed model weights: {args.resume}")
    train_loader = DataLoader(FaceCropperLiteDataset(train_df), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(FaceCropperLiteDataset(val_df), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    (out_dir / "model_config.json").write_text(json.dumps({"model": "LiteFaceCropperModel", "input_shape": [1, 3, 256, 256], "args": vars(args)}, indent=2, ensure_ascii=False), encoding="utf-8")

    best_iou = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, args.bbox_loss_weight, args.conf_loss_weight)
        metrics = evaluate(model, val_loader, device, args.conf_threshold, args.bbox_loss_weight, args.conf_loss_weight)
        scheduler.step()
        print(f"Epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.5f} | val_loss={metrics['loss']:.5f} | conf_acc={metrics['conf_acc']:.4f} | mean_iou={metrics['mean_iou_pos']:.4f} | median_iou={metrics['median_iou_pos']:.4f}")
        save_checkpoint(out_dir / "face_cropper_lite_last.pt", model, optimizer, epoch, args, metrics)
        if metrics["mean_iou_pos"] > best_iou:
            best_iou = metrics["mean_iou_pos"]
            save_checkpoint(out_dir / "face_cropper_lite_best.pt", model, optimizer, epoch, args, metrics)
            print(f"  -> saved best: {out_dir / 'face_cropper_lite_best.pt'} | mean_iou={best_iou:.4f}")
    print(f"[Done] best weight: {out_dir / 'face_cropper_lite_best.pt'}")


if __name__ == "__main__":
    main()
