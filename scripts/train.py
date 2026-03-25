"""
scripts/train.py
─────────────────
Main training entrypoint.
Trains all models defined in configs/config.yaml sequentially.

Usage:
    python scripts/train.py                          # train all models
    python scripts/train.py --model efficientnet_b4  # train one model only
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.common import set_seed, load_config, get_logger
from src.training.trainer import train_kfold

logger = get_logger("train", "outputs/logs")


def main():
    parser = argparse.ArgumentParser(description="Train Spot the Mask models")
    parser.add_argument("--config", default="configs/config.yaml",
                        help="Path to config YAML")
    parser.add_argument("--model", default=None,
                        help="Train only this model name (optional)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.project.seed)

    models_to_train = cfg.models
    if args.model:
        models_to_train = [m for m in cfg.models if m["name"] == args.model]
        if not models_to_train:
            raise ValueError(f"Model '{args.model}' not found in config.")

    logger.info(f"Training {len(models_to_train)} model(s)")
    for i, model_cfg in enumerate(models_to_train, 1):
        logger.info(f"\n[{i}/{len(models_to_train)}] Starting {model_cfg['name']}")
        train_kfold(cfg, model_cfg)

    logger.info("\nAll training complete. Next step:")
    logger.info("  python scripts/predict.py --config configs/config.yaml")


if __name__ == "__main__":
    main()
