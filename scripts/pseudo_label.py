"""
scripts/pseudo_label.py
────────────────────────
Run pseudo labeling after initial ensemble predictions.

Usage:
    python scripts/pseudo_label.py --round 1
    python scripts/pseudo_label.py --round 2   # after re-running predict.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.pseudo_label import run_pseudo_labeling
from src.utils.common import get_logger

logger = get_logger("pseudo_label_script", "outputs/logs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--round", type=int, default=1,
                        help="Pseudo labeling round (1 or 2)")
    args = parser.parse_args()

    logger.info(f"Starting pseudo labeling round {args.round}")
    run_pseudo_labeling(args.config, args.round)


if __name__ == "__main__":
    main()
