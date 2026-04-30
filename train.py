#!/usr/bin/env python3
"""
Training entry point.
"""

import argparse
import logging

from dotenv import load_dotenv
load_dotenv()

from src.utils.config import get_config
from src.data.tokenizer import SVGTokenizer
from src.training.trainer import train
from src.training.config import TrainConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Train a transformer on SVG data.")
    p.add_argument(
        "--model",
        choices=["tiny", "small", "medium", "large", "xl"],
        default="tiny",
        help="Model size to train.",
    )
    p.add_argument("--run_name", required=True, help="Unique name for this run.")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device (cpu | cuda)")
    p.add_argument("--learning_rate", type=float, required=True)
    return p.parse_args()


def build_config(args) -> TrainConfig:
    shared   = get_config("shared/core")
    data_cfg = get_config("data/data")
    model    = get_config(f"model/{args.model}")
    training = get_config("training/base")

    tokenizer_dir = data_cfg["data_paths"]["tokenizer_dir"]
    vocab_size = SVGTokenizer.from_pretrained(tokenizer_dir)._tokenizer.get_vocab_size()

    config = TrainConfig(
        **shared,
        **{k: v for k, v in training.items() if v is not None and k != "learning_rate"},    # exclude learning_rate if it's set via CLI
        **model,
        vocab_size=vocab_size,
        learning_rate=args.learning_rate,
        device=args.device,
        run_name=args.run_name,
    )

    logger.info(f"Run: {config.run_name} | model: {args.model} | device: {config.device}")
    logger.info(
        f"Architecture: n_layer={config.n_layer}  n_head={config.n_head}  "
        f"d_model={config.d_model}  block_size={config.block_size}  vocab_size={config.vocab_size}"
    )
    logger.info(
        f"Training: lr={config.learning_rate:.2e}  warmup={config.warmup_steps}  "
        f"batch_tokens={config.batch_size_tokens}  grad_clip={config.grad_clip:.1f}"
    )
    return config


def main() -> None:
    args   = parse_args()
    config = build_config(args)
    train(config)


if __name__ == "__main__":
    main()
