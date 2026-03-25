"""
utils/common.py
───────────────
Shared utilities: seed fixing, config loading, logging setup.
"""
import os
import random
import logging
import yaml
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


# ── Reproducibility ────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Pin all random sources so every run is reproducible."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Config ──────────────────────────────────────────────────────────────────

def load_config(path: str | Path) -> OmegaConf:
    """Load YAML config into an OmegaConf DictConfig."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return OmegaConf.create(raw)


# ── Logging ─────────────────────────────────────────────────────────────────

def get_logger(name: str, log_dir: str | Path | None = None) -> logging.Logger:
    """
    Return a logger that writes to console and optionally to a file.
    Call once per module at the top level.
    """
    logger = logging.getLogger(name)
    if logger.handlers:          # already configured
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(log_dir) / f"{name}.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ── Device ───────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Return the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── AUC helper ───────────────────────────────────────────────────────────────

from sklearn.metrics import roc_auc_score


def compute_auc(targets: np.ndarray, preds: np.ndarray) -> float:
    """Safe AUC computation — returns 0.5 if only one class present."""
    try:
        return float(roc_auc_score(targets, preds))
    except ValueError:
        return 0.5
