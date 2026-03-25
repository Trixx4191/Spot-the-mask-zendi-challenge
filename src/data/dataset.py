"""
data/dataset.py
───────────────
PyTorch Dataset + Albumentations augmentation pipelines.
Handles albumentations API changes across versions by inspecting
each transform's actual __init__ signature at runtime.
"""
from __future__ import annotations

import inspect
import cv2
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ── Version-safe helpers (inspect actual signature to decide) ────────────────

def _random_resized_crop(image_size: int, scale=(0.7, 1.0), ratio=(0.75, 1.33)):
    sig = inspect.signature(A.RandomResizedCrop.__init__).parameters
    if "size" in sig:
        return A.RandomResizedCrop(
            size=(image_size, image_size), scale=scale, ratio=ratio, p=1.0,
        )
    return A.RandomResizedCrop(
        height=image_size, width=image_size, scale=scale, ratio=ratio, p=1.0,
    )


def _resize(image_size: int) -> A.Resize:
    sig = inspect.signature(A.Resize.__init__).parameters
    if "size" in sig:
        return A.Resize(size=(image_size, image_size))
    return A.Resize(height=image_size, width=image_size)


def _coarse_dropout(image_size: int) -> A.CoarseDropout:
    hole = image_size // 8
    sig = inspect.signature(A.CoarseDropout.__init__).parameters
    if "num_holes_range" in sig:
        # albumentations >= 1.4
        return A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(hole // 2, hole),
            hole_width_range=(hole // 2, hole),
            p=0.4,
        )
    # albumentations < 1.4
    return A.CoarseDropout(
        max_holes=8, max_height=hole, max_width=hole,
        fill_value=0, p=0.4,
    )


# ── Augmentation pipelines ───────────────────────────────────────────────────

def get_train_transforms(image_size: int = 384) -> A.Compose:
    return A.Compose([
        _random_resized_crop(image_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.15,
            rotate_limit=15, border_mode=cv2.BORDER_REFLECT, p=0.6,
        ),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20,
                                 val_shift_limit=10, p=1.0),
        ], p=0.6),
        A.OneOf([
            A.CLAHE(clip_limit=4.0, p=1.0),
            A.Sharpen(p=1.0),
            A.Blur(blur_limit=3, p=1.0),
        ], p=0.4),
        _coarse_dropout(image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 384) -> A.Compose:
    return A.Compose([
        _resize(image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_tta_transforms(image_size: int = 384) -> list[A.Compose]:
    """Return 8 TTA pipelines — predictions are averaged across all."""
    base = [
        _resize(image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
    augments = [
        [],                                                        # 1. original
        [A.HorizontalFlip(p=1.0)],                                 # 2. H-flip
        [A.VerticalFlip(p=1.0)],                                   # 3. V-flip
        [A.RandomBrightnessContrast(brightness_limit=0.1,
                                    contrast_limit=0.1, p=1.0)],   # 4. brighter
        [A.HorizontalFlip(p=1.0),
         A.RandomBrightnessContrast(brightness_limit=0.1,
                                    contrast_limit=0.1, p=1.0)],   # 5. flip+bright
        [A.CLAHE(clip_limit=2.0, p=1.0)],                         # 6. CLAHE
        [A.Sharpen(p=1.0)],                                        # 7. sharpen
        [A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05,
                            rotate_limit=10, p=1.0)],              # 8. slight rotate
    ]
    return [A.Compose(aug + base) for aug in augments]


# ── Dataset class ─────────────────────────────────────────────────────────────

class MaskDataset(Dataset):
    """
    Works for train (with labels) and test (label_col=None) modes.

    Args:
        df          : DataFrame with column 'image' and optionally 'target'.
        images_dir  : Directory containing the raw images.
        transform   : Albumentations Compose pipeline.
        label_col   : Column name for binary labels. Pass None for test set.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: str | Path,
        transform: A.Compose,
        label_col: str | None = "target",
    ):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.label_col = label_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        img_path = self.images_dir / row["image"]

        image = cv2.imread(str(img_path))
        if image is None:
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        augmented = self.transform(image=image)
        tensor = augmented["image"]   # CHW float32

        sample: dict = {"image": tensor, "image_id": row["image"]}

        if self.label_col is not None:
            sample["label"] = torch.tensor(float(row[self.label_col]),
                                           dtype=torch.float32)
        return sample
