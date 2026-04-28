"""
Dataset metrics for evaluation and report.
"""

import logging
import os
import statistics
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt

from src.utils.config import get_config

logger = logging.getLogger(__name__)

MAX_TOKEN_LENGTH = get_config("data/data")["cleaning"]["max_token_length"]
PLOT_COLORS = {"train": "#2166ac", "val": "#d6604d", "test": "#4dac26"}


def _percentile(data: list[int | float], p: float) -> float:
    s = sorted(data)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _length_stats(lengths: list[int]) -> dict:
    return {
        "count": len(lengths),
        "total": sum(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "mean": statistics.mean(lengths),
        "median": statistics.median(lengths),
        "stdev": statistics.stdev(lengths) if len(lengths) > 1 else 0.0,
        "p25": _percentile(lengths, 25),
        "p75": _percentile(lengths, 75),
        "p95": _percentile(lengths, 95),
        "p99": _percentile(lengths, 99),
    }


def compute_dataset_stats(
    seq_lengths: dict[str, list[int]],
    vocab_size: int,
    n_before_filter: Optional[int] = None,
) -> dict:
    """
    Compute all dataset metrics required for the report from pre-computed
    per-SVG token lengths (collected during SVGDataset.from_svgs).

    Args:
        seq_lengths: Dict mapping split name to list of per-SVG token lengths
        vocab_size: Tokenizer vocabulary size
        n_before_filter: Total SVG count before token-length filtering (sum across splits)

    Returns
        A dict with keys:
            "vocab_size"     : int
            "token_counts"   : {split: int}
            "filter"         : {"before", "after", "dropped", "max_token_length"} or None
            "seq_len_stats"  : {split: stats_dict}
            "seq_lengths"    : {split: list[int]}
    """
    token_counts = {split: sum(lengths) for split, lengths in seq_lengths.items()}

    n_after = sum(len(lengths) for lengths in seq_lengths.values())
    filter_info = None
    if n_before_filter is not None:
        filter_info = {
            "before": n_before_filter,
            "after": n_after,
            "dropped": n_before_filter - n_after,
            "max_token_length": MAX_TOKEN_LENGTH,
        }

    return {
        "vocab_size": vocab_size,
        "token_counts": token_counts,
        "filter": filter_info,
        "seq_len_stats": {k: _length_stats(v) for k, v in seq_lengths.items()},
        "seq_lengths": seq_lengths,
    }


def print_stats_table(stats: dict) -> None:
    """Print a summary of the computed stats."""
    print("\nDataset Metrics\n")
    print(f"  Vocabulary size: {stats['vocab_size']:,}")

    print("\n  Token counts per split:")
    for split, count in stats["token_counts"].items():
        print(f"    {split:<8} {count:>14,} tokens")

    if stats["filter"] is not None:
        f = stats["filter"]
        pct = 100 * f["dropped"] / f["before"] if f["before"] else 0
        print(f"\n  SVG filtering (max_token_length={f['max_token_length']}):")
        print(f"    Before : {f['before']:,} files")
        print(f"    After  : {f['after']:,} files")
        print(f"    Dropped: {f['dropped']:,} files ({pct:.1f}%)")


def plot_sequence_length_histogram(
    stats: dict,
    out_dir: str | os.PathLike = get_config("evaluation/evaluation")["figures_path"],
    histogram_bins: int = 60,
) -> None:
    """Save one sequence-length histogram PDF per split (train/val/test)."""
    matplotlib.use("Agg")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split, lengths in stats["seq_lengths"].items():
        fig, ax = plt.subplots(figsize=(6, 3.5))

        ax.hist(
            lengths,
            bins=histogram_bins,
            color=PLOT_COLORS.get(split),
            alpha=0.8,
            edgecolor="white",
            linewidth=0.3,
        )

        if MAX_TOKEN_LENGTH is not None:
            ax.axvline(
                MAX_TOKEN_LENGTH,
                color="#b2182b",
                linestyle="--",
                linewidth=1.2,
                label=f"filter threshold ({MAX_TOKEN_LENGTH})",
            )
            ax.legend(fontsize=9)

        ax.set_xlabel("Sequence length (tokens)", fontsize=10)
        ax.set_ylabel("Number of SVGs", fontsize=10)
        ax.set_title(f"SVG Sequence Length Distribution — {split}", fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()

        out_path = out_dir / f"seq_len/seq_len_{split}.pdf"
        fig.savefig(out_path, format="pdf", bbox_inches="tight", dpi=300)
        plt.close(fig)
        logger.info(f"Histogram saved to {out_path}")
