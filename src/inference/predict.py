"""
inference/predict.py
─────────────────────
Inference pipeline:
  • Loads all trained fold checkpoints per model
  • Applies TTA (8 augmented passes per image)
  • Averages across folds and models (weighted ensemble)
  • Outputs a Zindi-format submission CSV

Usage:
    python -m src.inference.predict --config configs/config.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import MaskDataset, get_tta_transforms, get_val_transforms
from src.models.model import build_model
from src.utils.common import set_seed, get_logger, get_device, load_config

logger = get_logger("predict")


@torch.no_grad()
def predict_with_tta(model, df, images_dir, tta_transforms, batch_size, num_workers, device) -> np.ndarray:
    """Average predictions over all TTA pipelines."""
    model.eval()
    all_preds = []

    for tta_idx, transform in enumerate(tta_transforms):
        ds = MaskDataset(df, images_dir, transform, label_col=None)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
        fold_preds = []
        for batch in tqdm(loader, desc=f"    TTA {tta_idx+1}/{len(tta_transforms)}", leave=False):
            images = batch["image"].to(device)
            with autocast():
                logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            fold_preds.extend(probs.tolist())
        all_preds.append(np.array(fold_preds))

    return np.mean(all_preds, axis=0)


def run_inference(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(cfg.project.seed)
    device = get_device()
    logger.info(f"Running inference on {device}")

    # Load test dataframe from sample submission
    sample_sub = pd.read_csv(cfg.paths.sample_sub)
    test_df = sample_sub[["id"]].rename(columns={"id": "image"})

    # Prefer processed images
    images_dir = Path(cfg.paths.images_dir)
    processed_dir = Path(cfg.paths.processed_dir) / "images"
    if processed_dir.exists() and len(list(processed_dir.glob("*"))) > 100:
        images_dir = processed_dir

    tta_transforms = get_tta_transforms(cfg.data.image_size)
    model_dir = Path(cfg.paths.model_dir)

    ensemble_preds = []    # list of (weight, preds_array)

    for model_cfg in cfg.models:
        model_name = model_cfg["name"]
        image_size = model_cfg.get("image_size", cfg.data.image_size)
        dropout    = model_cfg.get("dropout", 0.4)
        weight     = cfg.inference.ensemble_weights.get(model_name, 0.25)

        # Collect fold checkpoints
        ckpts = sorted(model_dir.glob(f"{model_name}_fold*.pt"))
        if not ckpts:
            logger.warning(f"No checkpoints found for {model_name} — skipping")
            continue

        logger.info(f"\n{model_name} — {len(ckpts)} folds — weight={weight}")
        tta_tfms = get_tta_transforms(image_size)
        fold_preds_list = []

        for ckpt in ckpts:
            logger.info(f"  Loading {ckpt.name}")
            model = build_model(model_name, pretrained=False, dropout=dropout).to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))

            preds = predict_with_tta(
                model, test_df, images_dir, tta_tfms,
                cfg.training.batch_size * 2, cfg.data.num_workers, device
            )
            fold_preds_list.append(preds)

        model_preds = np.mean(fold_preds_list, axis=0)
        ensemble_preds.append((weight, model_preds))
        logger.info(f"  {model_name} pred range: [{model_preds.min():.4f}, {model_preds.max():.4f}]")

    if not ensemble_preds:
        raise RuntimeError("No predictions generated — check that models are trained.")

    # Weighted average
    total_weight = sum(w for w, _ in ensemble_preds)
    final_preds = sum(w * p for w, p in ensemble_preds) / total_weight
    logger.info(f"\nEnsemble pred range: [{final_preds.min():.4f}, {final_preds.max():.4f}]")

    # Build submission
    submission = pd.DataFrame({"id": test_df["image"], "label": final_preds})
    sub_path = Path(cfg.paths.submission_dir) / "submission_ensemble.csv"
    sub_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(sub_path, index=False)
    logger.info(f"Submission saved → {sub_path}")

    # Sanity check
    logger.info(f"Submission shape: {submission.shape}")
    logger.info(submission.describe())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    run_inference(args.config)
