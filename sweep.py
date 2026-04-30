#!/usr/bin/env python3
"""
Learning rate sweep.
"""

import argparse
import logging

from src.utils.config import get_config
from src.data.tokenizer import SVGTokenizer
from src.training.config import TrainConfig
from src.training.trainer import train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_LRS = [1e-5, 1e-4, 3e-4, 1e-3, 1e-2]


def parse_args():
    p = argparse.ArgumentParser(description="LR sweep.")
    p.add_argument(
        "--model",
        choices=["tiny", "small", "medium", "large", "xl"],
        default="tiny",
        help="Model size to train.",
    )
    p.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"], help="Device (cpu | cuda)"
    )
    p.add_argument(
        "--lrs", nargs="+", type=float, default=DEFAULT_LRS, help="Learning rates to try."
    )
    return p.parse_args()


def build_config(args, lr: float, idx: int) -> TrainConfig:
    shared   = get_config("shared/core")
    data_cfg = get_config("data/data")
    model    = get_config(f"model/{args.model}")
    training = get_config("training/base")

    tokenizer_dir = data_cfg["data_paths"]["tokenizer_dir"]
    vocab_size = SVGTokenizer.from_pretrained(tokenizer_dir)._tokenizer.get_vocab_size()

    return TrainConfig(
        **shared,
        **{k: v for k, v in training.items() if v is not None and k != "learning_rate"},
        **model,
        vocab_size=vocab_size,
        learning_rate=lr,
        device=args.device,
        run_name=f"sweep_{idx:02d}_{lr:.0e}",
    )


def main() -> None:
    args = parse_args()

    logger.info(f"LR sweep — model: {args.model}  |  lrs: {args.lrs}")

    results: list[tuple[float, float]] = []

    for idx, lr in enumerate(args.lrs, start=1):
        logger.info(f"--- lr={lr:.2e} ---")
        config = build_config(args, lr, idx)
        val_loss = train(config)
        results.append((lr, val_loss))
        print(f"lr={lr:.0e}  val_loss={val_loss:.4f}")

    best_lr, best_loss = min(results, key=lambda x: x[1])

    print("\n=== LR Sweep Results ===")
    print(f"{'LR':<12} {'Val Loss':>10}")
    print("-" * 24)
    for lr, val_loss in results:
        marker = " <- best" if lr == best_lr else ""
        print(f"{lr:<12.2e} {val_loss:>10.4f}{marker}")
    print(f"\nbest lr: {best_lr:.0e}  val_loss={best_loss:.4f}")


if __name__ == "__main__":
    main()
