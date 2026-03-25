"""
scripts/validate_submission.py
───────────────────────────────
Validate the submission CSV before uploading to Zindi.
Checks:
  • All test IDs present
  • No missing values
  • Labels are raw probabilities in [0, 1]
  • No rounded values (0 or 1 exactly)
  • Correct column names

Usage:
    python scripts/validate_submission.py --config configs/config.yaml
    python scripts/validate_submission.py --sub outputs/submissions/submission_ensemble.csv \
                                          --sample data/raw/sample_submission.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.utils.common import get_logger

logger = get_logger("validator")

PASS = "✓"
FAIL = "✗"


def validate(sub_path: str, sample_path: str) -> bool:
    sub    = pd.read_csv(sub_path)
    sample = pd.read_csv(sample_path)

    ok = True

    # 1. Column names
    expected_cols = set(sample.columns)
    actual_cols   = set(sub.columns)
    if expected_cols == actual_cols:
        logger.info(f"{PASS}  Columns correct: {list(actual_cols)}")
    else:
        logger.error(f"{FAIL}  Column mismatch. Expected {expected_cols}, got {actual_cols}")
        ok = False

    # Rename to standard names for remaining checks
    id_col    = "id" if "id" in sub.columns else sub.columns[0]
    label_col = "label" if "label" in sub.columns else sub.columns[1]

    # 2. Row count
    if len(sub) == len(sample):
        logger.info(f"{PASS}  Row count correct: {len(sub)}")
    else:
        logger.error(f"{FAIL}  Row count mismatch. Expected {len(sample)}, got {len(sub)}")
        ok = False

    # 3. All IDs present
    expected_ids = set(sample[sample.columns[0]])
    actual_ids   = set(sub[id_col])
    missing = expected_ids - actual_ids
    extra   = actual_ids - expected_ids
    if not missing and not extra:
        logger.info(f"{PASS}  All {len(expected_ids)} IDs present")
    else:
        if missing:
            logger.error(f"{FAIL}  Missing {len(missing)} IDs: {list(missing)[:5]}")
            ok = False
        if extra:
            logger.error(f"{FAIL}  {len(extra)} unexpected IDs: {list(extra)[:5]}")
            ok = False

    # 4. No NaN labels
    n_nan = sub[label_col].isna().sum()
    if n_nan == 0:
        logger.info(f"{PASS}  No NaN values")
    else:
        logger.error(f"{FAIL}  {n_nan} NaN values in label column")
        ok = False

    # 5. Probabilities in [0, 1]
    out_of_range = ((sub[label_col] < 0) | (sub[label_col] > 1)).sum()
    if out_of_range == 0:
        logger.info(f"{PASS}  All values in [0, 1]")
    else:
        logger.error(f"{FAIL}  {out_of_range} values outside [0, 1]")
        ok = False

    # 6. No hard 0 or 1 (should be raw probabilities)
    hard_labels = ((sub[label_col] == 0) | (sub[label_col] == 1)).sum()
    if hard_labels == 0:
        logger.info(f"{PASS}  No hard 0/1 labels — raw probabilities confirmed")
    else:
        logger.warning(f"⚠   {hard_labels} exact 0 or 1 values — ensure these are not rounded predictions")

    # 7. Distribution summary
    labels = sub[label_col].values
    logger.info(f"\n  Prediction distribution:")
    logger.info(f"    min={labels.min():.4f}  max={labels.max():.4f}")
    logger.info(f"    mean={labels.mean():.4f}  std={labels.std():.4f}")
    logger.info(f"    >0.5: {(labels > 0.5).sum()} ({(labels>0.5).mean()*100:.1f}%)")
    logger.info(f"    ≤0.5: {(labels <= 0.5).sum()} ({(labels<=0.5).mean()*100:.1f}%)")

    if ok:
        logger.info(f"\n{PASS}  Submission looks good — safe to upload!")
    else:
        logger.error(f"\n{FAIL}  Submission has errors — fix before uploading.")

    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub",    default="outputs/submissions/submission_ensemble.csv")
    parser.add_argument("--sample", default="data/raw/sample_submission.csv")
    args = parser.parse_args()
    validate(args.sub, args.sample)
