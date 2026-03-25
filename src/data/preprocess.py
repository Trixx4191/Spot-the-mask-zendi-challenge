"""
data/preprocess.py
──────────────────
Optional face-crop preprocessing using MTCNN.
Running this script saves cropped face images into data/processed/images/
which can replace the raw images for training.

Usage:
    python -m src.data.preprocess --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.utils.common import get_logger, load_config

logger = get_logger("preprocess")

MARGIN = 0.30       # extra margin around detected face (fraction of box side)
MIN_FACE_SIZE = 40  # ignore tiny detections


def crop_face(image: np.ndarray, box: list, margin: float) -> np.ndarray:
    """Crop a single face from image with margin, clamped to image bounds."""
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = image.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    mx = int(bw * margin)
    my = int(bh * margin)
    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(w, x2 + mx)
    y2 = min(h, y2 + my)
    return image[y1:y2, x1:x2]


def preprocess_images(config_path: str) -> None:
    cfg = load_config(config_path)
    src_dir = Path(cfg.paths.images_dir)
    dst_dir = Path(cfg.paths.processed_dir) / "images"
    dst_dir.mkdir(parents=True, exist_ok=True)

    try:
        from facenet_pytorch import MTCNN
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mtcnn = MTCNN(keep_all=False, device=device, min_face_size=MIN_FACE_SIZE)
        use_mtcnn = True
        logger.info("MTCNN loaded — will crop faces")
    except ImportError:
        use_mtcnn = False
        logger.warning("facenet-pytorch not installed — copying images as-is")

    images = sorted(src_dir.glob("*.jpg")) + sorted(src_dir.glob("*.png"))
    logger.info(f"Processing {len(images)} images → {dst_dir}")

    copied = cropped = failed = 0

    for img_path in tqdm(images, desc="Preprocessing"):
        dst_path = dst_dir / img_path.name
        if dst_path.exists():
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            shutil.copy(img_path, dst_path)
            failed += 1
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if use_mtcnn:
            try:
                boxes, _ = mtcnn.detect(image_rgb)
                if boxes is not None and len(boxes) > 0:
                    # Use the largest (most confident) face
                    areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
                    best_box = boxes[int(np.argmax(areas))]
                    cropped_img = crop_face(image_rgb, best_box, MARGIN)
                    if cropped_img.size > 0:
                        cv2.imwrite(str(dst_path),
                                    cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))
                        cropped += 1
                        continue
            except Exception:
                pass

        # Fallback: copy original
        shutil.copy(img_path, dst_path)
        copied += 1

    logger.info(f"Done. Cropped={cropped} Copied={copied} Failed={failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    preprocess_images(args.config)
