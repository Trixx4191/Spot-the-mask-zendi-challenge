"""
scripts/optimize_weights.py
────────────────────────────
Find optimal ensemble weights by minimising OOF AUC loss with Nelder-Mead.
Run after all models are trained. Updates config with best weights.

Usage:
    python scripts/optimize_weights.py --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from src.utils.common import load_config, get_logger, compute_auc

logger = get_logger("weight_optimizer", "outputs/logs")


def neg_auc(weights: np.ndarray, preds_list: list, targets: np.ndarray) -> float:
    weights = np.abs(weights)
    weights /= weights.sum()
    blend = sum(w * p for w, p in zip(weights, preds_list))
    return -compute_auc(targets, blend)


def optimize_weights(config_path: str) -> None:
    cfg = load_config(config_path)
    model_dir = Path(cfg.paths.model_dir)

    oof_files = sorted(model_dir.glob("*_oof.csv"))
    if not oof_files:
        logger.error("No OOF files found. Train models first.")
        return

    logger.info(f"Found {len(oof_files)} OOF files:")
    model_names = []
    preds_list = []
    targets = None

    for f in oof_files:
        df = pd.read_csv(f)
        model_name = f.stem.replace("_oof", "")
        auc = compute_auc(df["target"].values, df["oof_pred"].values)
        logger.info(f"  {model_name:<35}  OOF AUC = {auc:.5f}")
        model_names.append(model_name)
        preds_list.append(df["oof_pred"].values)
        if targets is None:
            targets = df["target"].values

    # Equal weights baseline
    n = len(preds_list)
    equal_blend = np.mean(preds_list, axis=0)
    equal_auc = compute_auc(targets, equal_blend)
    logger.info(f"\nEqual weight baseline AUC: {equal_auc:.5f}")

    # Nelder-Mead optimisation
    x0 = np.ones(n) / n
    result = minimize(
        neg_auc,
        x0,
        args=(preds_list, targets),
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6},
    )
    best_weights = np.abs(result.x)
    best_weights /= best_weights.sum()
    best_auc = -result.fun

    logger.info(f"\nOptimised ensemble AUC: {best_auc:.5f}")
    logger.info("Best weights:")
    weight_dict = {}
    for name, w in zip(model_names, best_weights):
        logger.info(f"  {name:<35}  {w:.4f}")
        weight_dict[name] = round(float(w), 4)

    # Print ready-to-paste config snippet
    logger.info("\nPaste into configs/config.yaml → inference.ensemble_weights:")
    logger.info("  ensemble_weights:")
    for name, w in weight_dict.items():
        logger.info(f"    {name}: {w}")

    return weight_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    optimize_weights(args.config)
