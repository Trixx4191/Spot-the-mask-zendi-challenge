"""
training/trainer.py
────────────────────
K-fold training with crash recovery:
  - Skips already-completed folds (checks for existing checkpoint)
  - Flushes memory aggressively between folds
  - Saves OOF predictions incrementally (not just at the end)
  - Disk guard before every fold
"""
from __future__ import annotations

import gc
import time
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from src.data.dataset import MaskDataset, get_train_transforms, get_val_transforms
from src.models.model import build_model
from src.utils.common import set_seed, get_logger, get_device, compute_auc
from src.utils.disk import check_disk_space

logger = get_logger("trainer")


# ── Loss ─────────────────────────────────────────────────────────────────────

class LabelSmoothBCE(nn.Module):
    def __init__(self, smoothing: float = 0.05):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(logits, targets)


# ── LR scheduler ─────────────────────────────────────────────────────────────

def get_scheduler(optimizer, warmup_steps: int, total_steps: int):
    import math
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# ── AMP context ───────────────────────────────────────────────────────────────

def _amp_ctx(device: torch.device, enabled: bool):
    if not enabled or device.type == "cpu":
        return nullcontext()
    return torch.amp.autocast(device_type=device.type)


# ── Memory flush ──────────────────────────────────────────────────────────────

def _flush_memory(*objects):
    """Delete objects and aggressively free RAM."""
    for obj in objects:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    torch.cuda.empty_cache()


# ── Train one epoch ───────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, scheduler,
                    device, accum_steps, use_amp) -> dict:
    model.train()
    total_loss = 0.0
    all_targets, all_preds = [], []
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(loader, desc="  train", leave=False)):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        with _amp_ctx(device, use_amp):
            logits = model(images)
            loss   = criterion(logits, labels) / accum_steps

        loss.backward()

        if (step + 1) % accum_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        total_loss += loss.item() * accum_steps
        with torch.no_grad():
            all_preds.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
        all_targets.extend(labels.cpu().numpy().tolist())

        # free batch tensors immediately
        del images, labels, logits, loss

    return {
        "loss": total_loss / len(loader),
        "auc":  compute_auc(np.array(all_targets), np.array(all_preds)),
    }


# ── Validate ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, criterion, device, use_amp) -> dict:
    model.eval()
    total_loss = 0.0
    all_targets, all_preds = [], []

    for batch in tqdm(loader, desc="  val  ", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        with _amp_ctx(device, use_amp):
            logits = model(images)
            loss   = criterion(logits, labels)

        total_loss += loss.item()
        all_preds.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        all_targets.extend(labels.cpu().numpy().tolist())
        del images, labels, logits, loss

    return {
        "loss": total_loss / len(loader),
        "auc":  compute_auc(np.array(all_targets), np.array(all_preds)),
    }


# ── Predict loader ────────────────────────────────────────────────────────────

@torch.no_grad()
def _predict_loader(model, loader, device, use_amp) -> np.ndarray:
    model.eval()
    preds = []
    for batch in loader:
        images = batch["image"].to(device)
        with _amp_ctx(device, use_amp):
            logits = model(images)
        preds.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        del images, logits
    return np.array(preds)


# ── K-fold trainer ────────────────────────────────────────────────────────────

def train_kfold(cfg, model_cfg: dict) -> None:
    set_seed(cfg.project.seed)
    device  = get_device()
    use_amp = cfg.training.mixed_precision and device.type != "cpu"
    logger.info(f"Device: {device} | AMP: {use_amp}")

    model_name = model_cfg["name"]
    image_size = model_cfg.get("image_size", cfg.data.image_size)
    dropout    = model_cfg.get("dropout", 0.3)

    df = pd.read_csv(cfg.paths.train_csv)

    images_dir    = Path(cfg.paths.images_dir)
    processed_dir = Path(cfg.paths.processed_dir) / "images"
    if processed_dir.exists() and len(list(processed_dir.glob("*"))) > 100:
        images_dir = processed_dir
        logger.info(f"Using preprocessed images: {images_dir}")

    model_dir = Path(cfg.paths.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load existing OOF if resuming after crash
    oof_path  = model_dir / f"{model_name}_oof.csv"
    oof_preds = np.zeros(len(df))
    completed_folds = set()

    if oof_path.exists():
        existing = pd.read_csv(oof_path)
        if "oof_pred" in existing.columns:
            oof_preds = existing["oof_pred"].values.copy()
            # A fold is "done" if its checkpoint exists AND its OOF preds are non-zero
            for f_idx in range(cfg.data.n_folds):
                ckpt = model_dir / f"{model_name}_fold{f_idx+1}.pt"
                if ckpt.exists():
                    completed_folds.add(f_idx)
            if completed_folds:
                logger.info(f"Resuming — folds already done: {sorted(f+1 for f in completed_folds)}")

    skf = StratifiedKFold(n_splits=cfg.data.n_folds, shuffle=True,
                          random_state=cfg.project.seed)
    pin = device.type == "cuda"

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["target"])):

        # Skip completed folds
        if fold in completed_folds:
            logger.info(f"Fold {fold+1} already complete — skipping")
            continue

        logger.info(f"\n{'='*52}")
        logger.info(f"  {model_name}  ·  Fold {fold+1}/{cfg.data.n_folds}")
        logger.info(f"{'='*52}")

        free_gb = check_disk_space(".")
        if free_gb < 0.3:
            logger.error("< 300MB free — stopping to protect disk")
            break

        # Flush before building anything new
        _flush_memory()

        train_df = df.iloc[train_idx]
        val_df   = df.iloc[val_idx]

        train_ds = MaskDataset(train_df, images_dir, get_train_transforms(image_size))
        val_ds   = MaskDataset(val_df,   images_dir, get_val_transforms(image_size))

        train_loader = DataLoader(
            train_ds, batch_size=cfg.training.batch_size,
            shuffle=True, num_workers=cfg.data.num_workers,
            pin_memory=pin, persistent_workers=False,
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.training.batch_size * 2,
            shuffle=False, num_workers=cfg.data.num_workers,
            pin_memory=pin, persistent_workers=False,
        )

        model     = build_model(model_name, pretrained=True, dropout=dropout).to(device)
        criterion = LabelSmoothBCE(cfg.training.label_smoothing)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.training.base_lr,
            weight_decay=cfg.training.weight_decay,
        )

        steps_per_epoch = max(1, len(train_loader) // cfg.training.accumulation_steps)
        total_steps     = steps_per_epoch * cfg.training.epochs
        warmup_steps    = steps_per_epoch * cfg.training.warmup_epochs
        scheduler       = get_scheduler(optimizer, warmup_steps, total_steps)

        best_auc   = 0.0
        no_improve = 0
        model_path = model_dir / f"{model_name}_fold{fold+1}.pt"

        for epoch in range(1, cfg.training.epochs + 1):
            t0      = time.time()
            train_m = train_one_epoch(model, train_loader, optimizer, criterion,
                                      scheduler, device, cfg.training.accumulation_steps, use_amp)
            val_m   = validate(model, val_loader, criterion, device, use_amp)
            elapsed = time.time() - t0

            logger.info(
                f"  Ep {epoch:02d}/{cfg.training.epochs} | "
                f"loss={train_m['loss']:.4f} auc={train_m['auc']:.4f} | "
                f"val_loss={val_m['loss']:.4f} val_auc={val_m['auc']:.4f} | "
                f"{elapsed:.0f}s"
            )

            if val_m["auc"] > best_auc:
                best_auc   = val_m["auc"]
                no_improve = 0
                torch.save(model.state_dict(), model_path)
                logger.info(f"  ✓ val_auc={best_auc:.5f} saved ({model_path.stat().st_size/1e6:.1f}MB)")
            else:
                no_improve += 1
                if no_improve >= cfg.training.early_stopping_patience:
                    logger.info(f"  Early stop ep {epoch}")
                    break

        # OOF predictions
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        oof_preds[val_idx] = _predict_loader(model, val_loader, device, use_amp)
        fold_auc = compute_auc(df.iloc[val_idx]["target"].values, oof_preds[val_idx])
        logger.info(f"  Fold {fold+1} OOF AUC: {fold_auc:.5f}")

        # Save OOF incrementally after EVERY fold — crash-safe
        oof_df = pd.DataFrame({
            "image":    df["image"],
            "target":   df["target"],
            "oof_pred": oof_preds,
        })
        oof_df.to_csv(oof_path, index=False)
        logger.info(f"  OOF saved incrementally → {oof_path}")

        # Aggressively free everything before next fold
        _flush_memory(model, optimizer, scheduler, criterion,
                      train_loader, val_loader, train_ds, val_ds)

    # Final summary
    done_mask   = oof_preds != 0
    overall_auc = compute_auc(df["target"].values[done_mask], oof_preds[done_mask]) if done_mask.sum() > 0 else 0
    logger.info(f"\n{'='*52}")
    logger.info(f"  {model_name} — OOF AUC (completed folds): {overall_auc:.5f}")
    logger.info(f"{'='*52}\n")
    check_disk_space(".")
