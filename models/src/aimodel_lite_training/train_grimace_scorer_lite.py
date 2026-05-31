#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_grimace_scorer_lite.py

Train LiteGrimaceScoreModel on your original Grimace scorer labels.

Expected data layout follows the old training script style:

data_root/
├─pic/
│   v0_000001.png
│   v0_000002.png
│   ...
├─v0/
│   000001.csv
│   000002.csv
│   ...
├─v1/
│   000001.csv
│   ...
└─v3/
    ...

Each label CSV should contain eye/nose/cheek/ear/whisker scores in 0~3.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from aimodel_lite_models import LiteGrimaceScoreModel, ORGAN_COLUMNS


IMG_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_one_label_csv(csv_path: Path) -> Optional[Dict[str, int]]:
    raw = csv_path.read_bytes()
    text = None
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = raw.decode("utf-8", errors="ignore")

    labels = {}
    for line in text.splitlines():
        line = line.strip().replace("，", ",")
        if not line:
            continue
        parts = [x.strip() for x in line.split(",")]
        for i, part in enumerate(parts):
            key = part.lower()
            if key in ORGAN_COLUMNS:
                score_val = None
                for j in range(i + 1, len(parts)):
                    if parts[j] != "":
                        score_val = parts[j]
                        break
                if score_val is None:
                    non_empty = [x for x in parts if x != ""]
                    if len(non_empty) >= 2:
                        score_val = non_empty[-1]
                if score_val is None:
                    continue
                try:
                    score = int(float(score_val))
                except ValueError:
                    continue
                if 0 <= score <= 3:
                    labels[key] = score

    if all(k in labels for k in ORGAN_COLUMNS):
        return labels
    return None


def frame_id_candidates(stem: str) -> List[str]:
    stem = str(stem).strip()
    candidates = [stem]
    if stem.isdigit():
        candidates.append(f"{int(stem):06d}")
    out = []
    for x in candidates:
        if x not in out:
            out.append(x)
    return out


def resolve_image_path(data_root: Path, pic_dir: Path, video_prefix: str, frame_stem: str) -> Optional[Path]:
    candidates = []
    for fid in frame_id_candidates(frame_stem):
        for ext in IMG_EXTS:
            candidates.append(pic_dir / f"{video_prefix}_{fid}{ext}")
            candidates.append(pic_dir / f"{video_prefix}_{fid}{ext.upper()}")
            candidates.append(data_root / "pic" / f"{video_prefix}_{fid}{ext}")
            candidates.append(data_root / "pic" / f"{video_prefix}_{fid}{ext.upper()}")
    for p in candidates:
        if p.exists():
            return p.resolve()
    return None


def discover_label_dirs(data_root: Path, pic_subdir: str, explicit: Optional[List[str]]) -> List[str]:
    if explicit:
        return explicit
    dirs = []
    for p in sorted(data_root.iterdir()):
        if p.is_dir() and p.name != pic_subdir and not p.name.startswith("."):
            dirs.append(p.name)
    return dirs


def collect_samples(data_root: Path, label_dirs: List[str], pic_subdir: str, csv_glob: str) -> pd.DataFrame:
    pic_dir = (data_root / pic_subdir).resolve()
    if not pic_dir.exists():
        raise FileNotFoundError(f"Image folder not found: {pic_dir}")

    rows = []
    total_csv = 0
    bad_label = 0
    missing_image = 0
    bad_examples = []
    missing_examples = []

    print("[Info] label dirs:")
    for label_dir_name in label_dirs:
        label_dir = data_root / label_dir_name
        if not label_dir.exists():
            print(f"  - skip missing: {label_dir}")
            continue
        csv_paths = sorted(label_dir.glob(csv_glob))
        print(f"  - {label_dir}: {len(csv_paths)} CSV")
        for csv_path in csv_paths:
            total_csv += 1
            video_prefix = label_dir.name
            frame_stem = csv_path.stem
            labels = parse_one_label_csv(csv_path)
            if labels is None:
                bad_label += 1
                if len(bad_examples) < 10:
                    bad_examples.append(str(csv_path))
                continue
            img_path = resolve_image_path(data_root, pic_dir, video_prefix, frame_stem)
            if img_path is None:
                missing_image += 1
                if len(missing_examples) < 20:
                    missing_examples.append(f"{csv_path} -> expected {pic_subdir}/{video_prefix}_{frame_stem}.png")
                continue
            row = {"video_name": video_prefix, "frame_id": frame_stem, "csv_path": str(csv_path.resolve()), "image_path": str(img_path)}
            for k in ORGAN_COLUMNS:
                row[k] = int(labels[k])
            rows.append(row)

    print("\n[Info] scan result:")
    print(f"  CSV total     : {total_csv}")
    print(f"  valid samples : {len(rows)}")
    print(f"  bad labels    : {bad_label}")
    print(f"  missing images: {missing_image}")
    if bad_examples:
        print("[Warn] bad label examples:")
        for x in bad_examples:
            print("  -", x)
    if missing_examples:
        print("[Warn] missing image examples:")
        for x in missing_examples:
            print("  -", x)
    if not rows:
        raise RuntimeError("No valid grimace samples found.")

    df = pd.DataFrame(rows)
    print("\n[Info] samples by video_name:")
    print(df.groupby("video_name").size())
    print("\n[Info] label distribution:")
    for col in ORGAN_COLUMNS:
        print(f"  {col}:")
        print(df[col].value_counts().sort_index())
    return df.reset_index(drop=True)


class GrimaceLiteDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.items = []
        for _, row in df.iterrows():
            labels = np.array([int(row[c]) for c in ORGAN_COLUMNS], dtype=np.int64)
            self.items.append((Path(row["image_path"]), labels))
        print(f"[Dataset] grimace samples={len(self.items)}")

    @staticmethod
    def preprocess(frame: np.ndarray) -> torch.Tensor:
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=2)
        elif frame.ndim == 3 and frame.shape[2] == 1:
            frame = np.repeat(frame, 3, axis=2)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
        frame = frame.astype(np.float32) / 255.0
        frame = (frame - 0.5175) / 0.14
        frame = frame.transpose(2, 0, 1)
        return torch.tensor(frame, dtype=torch.float32)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, labels = self.items[idx]
        frame = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if frame is None:
            raise RuntimeError(f"cv2.imread failed: {img_path}")
        x = self.preprocess(frame)
        y = torch.tensor(labels, dtype=torch.long)
        return x, y, str(img_path)


def split_by_video_prefix(df: pd.DataFrame, val_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    train_parts, val_parts = [], []
    for video_name, g in df.groupby("video_name"):
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
        print(f"[Split] {video_name}: total={len(idx)}, train={len(train_idx)}, val={len(val_idx)}")
    train_df = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if not val_parts:
        raise RuntimeError("Validation set is empty.")
    val_df = pd.concat(val_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train_df, val_df


def make_head_class_weights(train_df: pd.DataFrame, device: torch.device):
    weights = []
    for col in ORGAN_COLUMNS:
        counts = train_df[col].astype(int).value_counts().to_dict()
        w = []
        for c in range(4):
            cnt = counts.get(c, 0)
            w.append(0.0 if cnt <= 0 else 1.0 / np.sqrt(cnt))
        w = np.array(w, dtype=np.float32)
        nz = w[w > 0]
        if len(nz) > 0:
            w = w / nz.mean()
        weights.append(torch.tensor(w, dtype=torch.float32, device=device))
    return weights


def compute_loss(logits: torch.Tensor, targets: torch.Tensor, head_weights=None):
    losses = []
    for i in range(5):
        weight = head_weights[i] if head_weights is not None else None
        losses.append(F.cross_entropy(logits[:, i, :], targets[:, i], weight=weight))
    return sum(losses) / len(losses)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    losses = []
    correct_all, total_all = 0, 0
    correct_not3, total_not3 = 0, 0
    organ_correct = np.zeros(5, dtype=np.int64)
    organ_total = np.zeros(5, dtype=np.int64)
    for x, y, paths in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x).view(-1, 5, 4)
        loss = compute_loss(logits, y)
        losses.append(loss.item())
        pred = logits.argmax(dim=2)
        correct = pred == y
        correct_all += correct.sum().item()
        total_all += y.numel()
        mask_not3 = y != 3
        correct_not3 += (correct & mask_not3).sum().item()
        total_not3 += mask_not3.sum().item()
        for i in range(5):
            organ_correct[i] += correct[:, i].sum().item()
            organ_total[i] += y[:, i].numel()
    organ_acc = organ_correct / np.maximum(organ_total, 1)
    return {"loss": float(np.mean(losses)) if losses else 999.0, "acc_all": correct_all / max(1, total_all), "acc_not3": correct_not3 / max(1, total_not3), "organ_acc": organ_acc.tolist()}


def train_one_epoch(model, loader, optimizer, device, head_weights=None):
    model.train()
    losses = []
    for x, y, paths in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x).view(-1, 5, 4)
        loss = compute_loss(logits, y, head_weights=head_weights)
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
    torch.save({"model_name": "grimace", "epoch": int(epoch), "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "args": vars(args), "metrics": metrics, "organ_columns": ORGAN_COLUMNS}, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, help="Root containing pic/ and label dirs.")
    parser.add_argument("--label_dirs", nargs="*", default=None, help="Label dirs such as v0 v1 v3. Default: auto-discover dirs except pic.")
    parser.add_argument("--pic_subdir", default="pic")
    parser.add_argument("--csv_glob", default="*.csv")
    parser.add_argument("--out_dir", default="runs/grimace_scorer_lite")
    parser.add_argument("--resume", default="")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weighted_loss", action="store_true")
    parser.add_argument("--device", default="")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    data_root = Path(args.data_root).resolve()
    label_dirs = discover_label_dirs(data_root, args.pic_subdir, args.label_dirs)
    print(f"[Info] using label_dirs={label_dirs}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = collect_samples(data_root, label_dirs, args.pic_subdir, args.csv_glob)
    if args.dry_run:
        print("[Dry Run] Data parsing OK. Not training.")
        return

    train_df, val_df = split_by_video_prefix(df, args.val_ratio, args.seed)
    train_df.to_csv(out_dir / "train_split.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(out_dir / "val_split.csv", index=False, encoding="utf-8-sig")
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = LiteGrimaceScoreModel().to(device)
    if args.resume:
        load_state_dict_flexible(model, Path(args.resume), device)
        print(f"[Info] Resumed model weights: {args.resume}")
    train_loader = DataLoader(GrimaceLiteDataset(train_df), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(GrimaceLiteDataset(val_df), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    head_weights = make_head_class_weights(train_df, device) if args.weighted_loss else None
    (out_dir / "model_config.json").write_text(json.dumps({"model": "LiteGrimaceScoreModel", "input_shape": [1, 3, 224, 224], "output_shape": [1, 20, 1, 1], "organ_columns": ORGAN_COLUMNS, "args": vars(args)}, indent=2, ensure_ascii=False), encoding="utf-8")

    best_acc_not3 = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, head_weights)
        metrics = evaluate(model, val_loader, device)
        scheduler.step()
        organ_acc_str = ", ".join([f"{name}:{acc:.3f}" for name, acc in zip(ORGAN_COLUMNS, metrics["organ_acc"])])
        print(f"Epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.5f} | val_loss={metrics['loss']:.5f} | acc_all={metrics['acc_all']:.4f} | acc_not3={metrics['acc_not3']:.4f} | {organ_acc_str}")
        save_checkpoint(out_dir / "grimace_scorer_lite_last.pt", model, optimizer, epoch, args, metrics)
        if metrics["acc_not3"] > best_acc_not3:
            best_acc_not3 = metrics["acc_not3"]
            save_checkpoint(out_dir / "grimace_scorer_lite_best.pt", model, optimizer, epoch, args, metrics)
            print(f"  -> saved best: {out_dir / 'grimace_scorer_lite_best.pt'} | acc_not3={best_acc_not3:.4f}")
    print(f"[Done] best weight: {out_dir / 'grimace_scorer_lite_best.pt'}")


if __name__ == "__main__":
    main()
