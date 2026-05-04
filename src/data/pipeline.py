"""
End-to-end data preparation pipeline for download, cleaning, and tokenizing SVG datasets.
"""

import logging

from src.eval.render import render_by_complexity
from src.data.split import create_splits
from src.data.download import download_datasets
from src.data.preprocess import process_svgs
from src.data.tokenizer import SVGTokenizer
from src.data.dataset import SVGDataset
from src.eval.dataset_metrics import (
    compute_dataset_stats,
    print_stats_table,
    plot_sequence_length_histogram,
)
from src.utils.config import get_config

logger = logging.getLogger(__name__)

config = get_config("data/data")


def pipeline(datasets: list[str], use_stats: bool, render: bool) -> None:
    # Step 1: Download datasets
    logger.info("**** Step 1: Download Datasets ****")
    datasets = download_datasets(datasets)
    # Combine SVGs from all downloaded datasets
    all_svgs: list[str] = []
    for svgs in datasets.values():
        all_svgs.extend(svgs)
    logger.info(f"Total SVGs downloaded: {len(all_svgs)}")

    # Step 2: Clean / Normalize
    logger.info("**** Step 2: Clean / Normalize SVGs ****")
    cleaned = process_svgs(all_svgs)

    # Step 3: Split into train/val/test
    logger.info("**** Step 3: Split Train/Val/Test ****")
    train_svgs, val_svgs, test_svgs = create_splits(cleaned)

    # Step 4: Train tokenizer
    logger.info("**** Step 4: Train Tokenizer ****")
    tokenizer = SVGTokenizer()
    tokenizer.train(train_svgs)
    tokenizer.save()

    # Step 5: Tokenize each split
    logger.info("**** Step 5: Tokenize & Save Datasets ****")
    splits = {"train": train_svgs, "val": val_svgs, "test": test_svgs}
    processed_dir = config["data_paths"]["processed_dir"]

    seq_lengths: dict[str, list[int]] = {}
    for split_name, svgs in splits.items():
        logger.info(f"Processing {split_name} set...")
        ds = SVGDataset.from_svgs(svgs, tokenizer)
        # Save token lengths for computing dataset metrics
        seq_lengths[split_name] = ds.seq_lengths
        out = f"{processed_dir}{split_name}.bin"
        ds.save(out)

    # Warning if training set tokens < 100M
    num_train_tokens = sum(seq_lengths["train"])
    if num_train_tokens < 100_000_000:
        logger.warning(
            f"Training set has {num_train_tokens} tokens - 100M+ tokens is recommended."
        )

    # Step 6: Dataset metrics & histogram
    if use_stats:
        logger.info("**** Step 6: Compute Dataset Metrics ****")
        stats = compute_dataset_stats(
            seq_lengths=seq_lengths,
            vocab_size=tokenizer.actual_vocab_size,
            n_before_filter=len(cleaned),
        )
        print_stats_table(stats)
        plot_sequence_length_histogram(stats)

    # Step 7: Render complexity samples
    if render:
        logger.info("**** Step 7: Render SVG Samples ****")
        render_by_complexity(train_svgs)
