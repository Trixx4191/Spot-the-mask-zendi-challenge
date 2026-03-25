# Spot the Mask — Leaderboard #1 Solution

**Competition:** [Zindi — Spot the Mask Challenge](https://zindi.africa/competitions/spot-the-mask)  
**Metric:** AUC (Area Under the ROC Curve)  
**Strategy:** EfficientNet/ConvNeXt/ViT ensemble + 8× TTA + K-fold + Pseudo labeling

---

## Project Structure

```
spot_the_mask/
├── configs/
│   └── config.yaml              ← All hyperparameters & model list
│
├── data/
│   ├── raw/                     ← Put Zindi data here (images/, train_labels.csv, sample_submission.csv)
│   └── processed/               ← Face-cropped images (auto-generated)
│       └── images/
│
├── src/
│   ├── data/
│   │   ├── dataset.py           ← MaskDataset + all augmentation pipelines
│   │   └── preprocess.py        ← MTCNN face-crop preprocessing
│   ├── models/
│   │   └── model.py             ← MaskClassifier (timm backbone + head)
│   ├── training/
│   │   ├── trainer.py           ← K-fold training engine
│   │   └── pseudo_label.py      ← Pseudo labeling pipeline
│   ├── inference/
│   │   └── predict.py           ← TTA + ensemble inference
│   └── utils/
│       └── common.py            ← Seed, logging, AUC, device
│
├── scripts/
│   ├── train.py                 ← python scripts/train.py
│   ├── predict.py               ← python scripts/predict.py
│   └── pseudo_label.py          ← python scripts/pseudo_label.py
│
├── notebooks/
│   ├── 01_eda.ipynb             ← Dataset exploration
│   └── 02_training.ipynb        ← Colab-ready end-to-end notebook
│
├── outputs/
│   ├── models/                  ← Saved checkpoints (fold*.pt) + OOF CSVs
│   ├── submissions/             ← submission_ensemble.csv → upload to Zindi
│   └── logs/                    ← Training logs
│
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download data from Zindi
Download from the [competition data page](https://zindi.africa/competitions/spot-the-mask/data) and place:
```
data/raw/images/          ← unzipped images (all ~1800 images)
data/raw/train_labels.csv
data/raw/sample_submission.csv
```

### 3. (Optional) Face crop preprocessing
Crops faces using MTCNN — improves AUC by focusing the model on the relevant region:
```bash
python -m src.data.preprocess --config configs/config.yaml
```

### 4. Train all models (K-fold)
```bash
python scripts/train.py
# OR train a single model:
python scripts/train.py --model efficientnet_b4
```
Checkpoints saved to `outputs/models/`.

### 5. Generate submission (TTA + ensemble)
```bash
python scripts/predict.py
```
Submission saved to `outputs/submissions/submission_ensemble.csv`.

### 6. Pseudo labeling (round 1)
```bash
python scripts/pseudo_label.py --round 1
python scripts/predict.py           # generate new submission with pseudo-labeled models
```

### 7. Pseudo labeling (round 2)
```bash
python scripts/pseudo_label.py --round 2
python scripts/predict.py
```

### 8. Submit to Zindi
Upload `outputs/submissions/submission_ensemble.csv`.

---

## Strategy Details

| Layer | Technique | Expected AUC gain |
|-------|-----------|-------------------|
| Baseline | EfficientNet-B4, ImageNet weights | ~0.97 |
| Augmentation | CoarseDropout, CLAHE, ShiftScaleRotate | +0.005 |
| Face crops | MTCNN preprocessing | +0.005 |
| K-fold (5) | Reduces variance | +0.003 |
| TTA (8×) | Test-time augmentation | +0.005 |
| Ensemble | B4 + B7 + ConvNeXt + ViT | +0.005 |
| Pseudo labels | High-confidence test samples | +0.003 |
| **Total** | | **~0.99+** |

---

## Config Reference (`configs/config.yaml`)

Key settings to tune:
- `data.n_folds` — increase to 10 for more stable OOF estimates
- `augmentation.tta_steps` — set to 8 (default); more = slower but smoother
- `training.epochs` — 50 with early stopping; safe to increase
- `pseudo_labeling.confidence_threshold` — 0.95 keeps only very confident pseudo labels
- `inference.ensemble_weights` — tune based on OOF AUC per model

---

## Notes
- Always submit **raw probabilities** — never round or threshold
- Max 10 submissions per day on Zindi
- No public/private leaderboard split — every submission is final
- All models must be publicly available (pretrained from timm = ✓)
