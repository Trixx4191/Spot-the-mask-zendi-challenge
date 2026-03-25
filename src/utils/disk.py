"""
utils/disk.py
─────────────
Disk space guard utilities.
Warn before training if free space is low.
Prune old fold checkpoints to stay within budget.
"""
from __future__ import annotations
import shutil
from pathlib import Path
from src.utils.common import get_logger

logger = get_logger("disk")

MIN_FREE_GB = 0.5   # warn if less than 500MB free


def check_disk_space(path: str | Path = ".") -> float:
    """Return free GB at path. Log warning if below MIN_FREE_GB."""
    usage = shutil.disk_usage(str(path))
    free_gb = usage.free / (1024 ** 3)
    total_gb = usage.total / (1024 ** 3)
    logger.info(f"Disk: {free_gb:.1f} GB free / {total_gb:.1f} GB total")
    if free_gb < MIN_FREE_GB:
        logger.warning(f"⚠ Only {free_gb:.2f} GB free — risk of out-of-disk crash!")
    return free_gb


def prune_fold_checkpoints(model_dir: str | Path, model_name: str,
                           keep_best_only: bool = False) -> None:
    """
    After all folds are done, optionally remove per-fold .pt files
    to reclaim disk space. OOF CSV is kept.

    keep_best_only=False  → keep all fold checkpoints (needed for ensemble)
    keep_best_only=True   → delete all fold checkpoints (saves ~550MB)
                            only safe AFTER you've already run predict.py
    """
    model_dir = Path(model_dir)
    ckpts = sorted(model_dir.glob(f"{model_name}_fold*.pt"))
    if not ckpts:
        return
    if keep_best_only:
        for ckpt in ckpts:
            size_mb = ckpt.stat().st_size / (1024 ** 2)
            ckpt.unlink()
            logger.info(f"  Deleted {ckpt.name} ({size_mb:.0f} MB freed)")
    else:
        total_mb = sum(c.stat().st_size for c in ckpts) / (1024 ** 2)
        logger.info(f"  {model_name}: {len(ckpts)} checkpoints using {total_mb:.0f} MB")
