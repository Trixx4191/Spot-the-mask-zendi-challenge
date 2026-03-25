# Spot the Mask — Full Competition Workflow

## Day-by-day game plan to reach #1

---

## STEP 0 — Environment setup

```bash
# Clone / navigate to project
cd spot_the_mask

# Install deps
pip install -r requirements.txt

# Verify GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## STEP 1 — Data setup

Download from [Zindi competition page](https://zindi.africa/competitions/spot-the-mask/data):

```
data/raw/images.zip
data/raw/train_labels.csv
data/raw/sample_submission.csv
```

Then unzip images:
```bash
unzip -q data/raw/images.zip -d data/raw/images/
```

Verify:
```bash
python -c "
import pandas as pd
from pathlib import Path
df = pd.read_csv('data/raw/train_labels.csv')
imgs = list(Path('data/raw/images').glob('*'))
print(f'Labels: {len(df)}  |  Images found: {len(imgs)}')
print(df['target'].value_counts())
"
```

---

## STEP 2 — EDA (optional but recommended)

```bash
jupyter notebook notebooks/01_eda.ipynb
```

Key things to look for:
- Class balance (mask vs no-mask ratio)
- Image size variation → confirm 384px resize is reasonable
- Any corrupted images

---

## STEP 3 — Face crop preprocessing (recommended)

Runs MTCNN to detect and crop the face region. Removes irrelevant background.
Adds ~0.005 AUC on average.

```bash
python -m src.data.preprocess --config configs/config.yaml
```

Output saved to `data/processed/images/`. Training auto-picks this up.

---

## STEP 4 — Train models

### Start with EfficientNet-B4 (fastest, ~30 min on T4 GPU)
```bash
python scripts/train.py --model efficientnet_b4
```

Check OOF AUC in terminal. Target: > 0.97.

### Train EfficientNet-B7 (~90 min)
```bash
python scripts/train.py --model efficientnet_b7
```

### Train ConvNeXt-Base (~60 min)
```bash
python scripts/train.py --model convnext_base
```

### Train ViT (~90 min)
```bash
python scripts/train.py --model vit_base_patch16_384
```

### Or train all at once (sequential)
```bash
python scripts/train.py
```

All checkpoints and OOF CSVs saved to `outputs/models/`.

---

## STEP 5 — Optimize ensemble weights

After training, find the best per-model weights using OOF data:

```bash
python scripts/optimize_weights.py --config configs/config.yaml
```

Copy the printed weights into `configs/config.yaml` under `inference.ensemble_weights`.

---

## STEP 6 — Generate submission (TTA + ensemble)

```bash
python scripts/predict.py
```

This runs 8-pass TTA per model, averages across all folds, then applies weighted ensemble.
Output: `outputs/submissions/submission_ensemble.csv`

---

## STEP 7 — Validate submission before upload

```bash
python scripts/validate_submission.py
```

Should print all green checkmarks. Never upload without validating.

---

## STEP 8 — Upload to Zindi

Upload `outputs/submissions/submission_ensemble.csv` to the competition page.
Note your score and position.

---

## STEP 9 — Pseudo labeling (round 1)

After getting your first score, use confident test predictions as extra training data:

```bash
python scripts/pseudo_label.py --round 1
python scripts/predict.py
python scripts/validate_submission.py
# upload new submission
```

---

## STEP 10 — Pseudo labeling (round 2)

```bash
python scripts/pseudo_label.py --round 2
python scripts/predict.py
python scripts/validate_submission.py
# upload — this should be your best submission
```

---

## Tuning checklist for maximum AUC

| Lever | What to change | Where |
|-------|---------------|-------|
| Bigger model | `efficientnet_b7` → try `tf_efficientnetv2_xl` | `configs/config.yaml` → `models` |
| More folds | `n_folds: 5` → `n_folds: 10` | `configs/config.yaml` → `data` |
| Larger image | `image_size: 384` → `512` | `configs/config.yaml` → `data` |
| More TTA | `tta_steps: 8` → `16` | `configs/config.yaml` → `augmentation` |
| Lower confidence threshold | `0.95` → `0.90` | `configs/config.yaml` → `pseudo_labeling` |
| Label smoothing | `0.05` → `0.0` (try both) | `configs/config.yaml` → `training` |

---

## Submission rules reminder

- **Max 10 submissions/day**
- **Submit raw probabilities** — values like `0.9923`, not `1`
- No public/private split — every submission hits the real leaderboard
- Pretrained models allowed (timm ImageNet weights = ✓)
- No metadata allowed (image size, aspect ratio, etc.)
