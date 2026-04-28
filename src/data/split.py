"""
Train/val/test split utilities.
"""

import logging
import random
from src.utils.config import get_config


logger = logging.getLogger(__name__)

config = get_config("data/data")["split"]

TRAIN_FRAC: float = config["train"]
VAL_FRAC: float   = config["val"]
SPLIT_SEED: int = config["seed"]

def create_splits(
    svgs: list[str],
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    seed: int = SPLIT_SEED,
) -> tuple[list[str], list[str], list[str]]:
    """Shuffle and split SVG strings into (train, val, test) lists."""
    data = list(svgs)
    random.seed(seed)
    random.shuffle(data)

    n = len(data)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    train = data[:n_train]
    val   = data[n_train : n_train + n_val]
    test  = data[n_train + n_val :]

    logger.info(
        f"Splits — train: {len(train)}, val: {len(val)}, test: {len(test)}"
    )
    return train, val, test
