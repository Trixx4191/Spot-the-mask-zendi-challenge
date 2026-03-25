"""
scripts/predict.py
───────────────────
Generate the final submission CSV.
Runs TTA + multi-model weighted ensemble.

Usage:
    python scripts/predict.py
    python scripts/predict.py --config configs/config.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.predict import run_inference
from src.utils.common import get_logger

logger = get_logger("predict_script", "outputs/logs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    logger.info("Starting ensemble inference + TTA...")
    run_inference(args.config)
    logger.info("Done — check outputs/submissions/submission_ensemble.csv")


if __name__ == "__main__":
    main()
