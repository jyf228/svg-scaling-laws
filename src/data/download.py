"""
Download SVG dataset(s) from HuggingFace.
"""

import logging
from pathlib import Path

from datasets import load_dataset, concatenate_datasets, DatasetDict
from src.utils.config import get_config
from src.utils.io_utils import write_data_to_disk

logger = logging.getLogger(__name__)

config = get_config("data/data")

DATASETS: dict = config["datasets"]


def download_datasets(
    dataset_names: list[str] = ("svg-icons-simple",)
) -> dict[str, list[str]]:
    """Download SVG datasets and return a dict mapping dataset name to a list of SVG strings."""
    results: dict[str, list[str]] = {}

    for name in dataset_names:
        if name not in DATASETS:
            raise ValueError(
                f"Unknown dataset '{name}'. Choose from: {list(DATASETS)}"
            )
        metadata = DATASETS[name]
        logger.info(f"Downloading {name}...")

        ds = load_dataset(metadata["path"])

        # Merge all splits into one flat dataset for now
        if isinstance(ds, DatasetDict):
            all_splits = list(ds.values())
            flat_ds = concatenate_datasets(all_splits)
        else:
            flat_ds = ds

        # Subsample large datasets
        limit = metadata["max_samples"]
        if limit and len(flat_ds) > limit:
            logger.info(
                f"Subsampling {name} to {limit} samples from {len(flat_ds)} total."
            )
            flat_ds = flat_ds.shuffle(seed=42).select(range(limit))

        svg_col = metadata["svg_col"]
        raw_values = flat_ds[svg_col]
        svg_strings: list[str] = [
            svg for svg in raw_values if isinstance(svg, str) # Filter out non-string/null entries
        ]

        dropped = len(raw_values) - len(svg_strings)
        if dropped:
            logger.warning(
                f"Dropped {dropped} empty SVG rows from {name}."
            )

        logger.info(f"Loaded {len(svg_strings)} SVG strings.")
        results[name] = svg_strings

        # Persist to disk as a plain text file
        raw_dir = config["data_paths"]["raw_dir"]
        out_path = Path(raw_dir) / f"{name}.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        write_data_to_disk(svg_strings, out_path)
        logger.info(f"Saved raw {name} to {out_path}.")

    return results
