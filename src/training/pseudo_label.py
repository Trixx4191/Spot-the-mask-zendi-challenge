"""
training/pseudo_label.py
─────────────────────────
Pseudo labeling pipeline:
  1. Load current best ensemble predictions on test set
  2. Filter high-confidence predictions (above threshold)
  3. Append to training data
  4. Retrain all models on augmented dataset
  5. Repeat for N rounds

Usage:
    python -m src.training.pseudo_label --config configs/config.yaml --round 1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.common import get_logger, load_config
from src.training.trainer import train_kfold

logger = get_logger("pseudo_label")


def run_pseudo_labeling(config_path: str, round_num: int = 1) -> None:
    cfg = load_config(config_path)

    if not cfg.pseudo_labeling.enabled:
        logger.info("Pseudo labeling disabled in config — exiting")
        return

    threshold = cfg.pseudo_labeling.confidence_threshold
    sub_dir = Path(cfg.paths.submission_dir)

    # Find the latest ensemble submission
    sub_files = sorted(sub_dir.glob("submission_ensemble*.csv"))
    if not sub_files:
        raise FileNotFoundError(
            "No ensemble submission found. Run inference first:\n"
            "  python -m src.inference.predict --config configs/config.yaml"
        )

    sub = pd.read_csv(sub_files[-1])
    logger.info(f"Loaded predictions from {sub_files[-1]}")

    # Filter high-confidence samples
    confident = sub[(sub["label"] >= threshold) | (sub["label"] <= (1 - threshold))].copy()
    confident["target"] = (confident["label"] >= threshold).astype(int)
    confident = confident.rename(columns={"id": "image"})[["image", "target"]]
    logger.info(
        f"High-confidence test samples: {len(confident)} / {len(sub)} "
        f"(threshold={threshold})"
    )
    logger.info(f"  Mask={confident['target'].sum()} | No-mask={(confident['target']==0).sum()}")

    if len(confident) == 0:
        logger.warning("No samples above threshold — skipping pseudo labeling")
        return

    # Merge with original training data
    train_df = pd.read_csv(cfg.paths.train_csv)
    combined = pd.concat([train_df, confident], ignore_index=True)
    logger.info(f"Combined dataset: {len(combined)} samples "
                f"({len(train_df)} original + {len(confident)} pseudo)")

    # Save augmented labels CSV
    pseudo_csv = Path(cfg.paths.processed_dir) / f"train_pseudo_round{round_num}.csv"
    pseudo_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(pseudo_csv, index=False)
    logger.info(f"Pseudo-label CSV saved → {pseudo_csv}")

    # Patch cfg to use the new CSV and retrain all models
    from omegaconf import OmegaConf
    pseudo_cfg = OmegaConf.merge(cfg, OmegaConf.create({
        "paths": {"train_csv": str(pseudo_csv)},
        "paths_tag": f"pseudo_round{round_num}"
    }))

    for model_cfg in pseudo_cfg.models:
        logger.info(f"\nRetraining {model_cfg['name']} with pseudo labels (round {round_num})")
        train_kfold(pseudo_cfg, model_cfg)

    logger.info(f"\nPseudo labeling round {round_num} complete.")
    logger.info("Run inference again to generate updated submission.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--round",  type=int, default=1)
    args = parser.parse_args()
    run_pseudo_labeling(args.config, args.round)
