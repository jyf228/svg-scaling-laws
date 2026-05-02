"""
Render SVG samples to PNG for evaluation and report figures.
"""

import logging
import re
from pathlib import Path

import cairosvg
from src.utils.config import get_config

logger = logging.getLogger(__name__)

FIGURES_PATH = get_config("eval/eval")["figures_path"]

# SVG elements that count toward visual complexity
COMPLEXITY_TAGS = re.compile(
    r"<(path|circle|rect|ellipse|line|polyline|polygon|text|use|image|g)[\s/>]",
    re.IGNORECASE,
)


def _complexity(svg: str) -> int:
    """Count the number of rendering elements in an SVG string."""
    return len(COMPLEXITY_TAGS.findall(svg))


def render_by_complexity(
    svgs: list[str],
    out_dir: str | Path = f"{FIGURES_PATH}render_samples",
) -> None:
    """
    Pick one SVG each at low, medium, and high complexity and render them to PNG.

    Complexity is measured by the number of SVG drawing elements. The corpus is sorted 
    by complexity and the low/medium/high samples are drawn from the 10th, 50th, and 90th 
    percentiles respectively.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Score every SVG for visual complexity
    scored = sorted(enumerate(svgs), key=lambda t: _complexity(t[1]))
    samples = len(scored)

    # Pick indices at 10th / 50th / 90th percentile of the sorted list
    targets = {
        "low":    scored[max(0, int(0.10 * samples))],
        "medium": scored[int(0.50 * samples)],
        "high":   scored[min(samples - 1, int(0.90 * samples))],
    }

    for level, (orig_idx, svg) in targets.items():
        complexity = _complexity(svg)
        out_path = out_dir / f"sample_{level}_complexity.png"
        try:
            cairosvg.svg2png(bytestring=svg.encode(), write_to=str(out_path), output_width=512, output_height=512)
            logger.info(
                f"Rendered {level}-complexity sample (idx={orig_idx}, elements={complexity}). Saved to {out_path}",
            )
        except Exception as exc:
            logger.warning(f"Could not render {level}-complexity sample (idx={orig_idx}): {exc}")


def render_generated_samples(
    svgs: list[str],
    out_dir: str | Path,
    width: int = 512,
    height: int = 512,
) -> None:
    """
    Render a list of SVGs to PNGs file.
    SVGs that fail to render are skipped with a warning.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    for i, svg in enumerate(svgs):
        out_path = out_dir / f"{i:04d}.png"
        try:
            cairosvg.svg2png(
                bytestring=svg.encode(),
                write_to=str(out_path),
                output_width=width,
                output_height=height,
            )
            n_ok += 1
        except Exception as exc:
            logger.warning(f"Could not render sample {i}: {exc}")

    logger.info(f"Rendered {n_ok}/{len(svgs)} samples. Saved to {out_dir}.")
