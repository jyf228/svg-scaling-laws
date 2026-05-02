#!/usr/bin/env python3
"""
Sample generation from a trained SVG transformer.
"""

import argparse
import json
import logging
from pathlib import Path

import torch

from src.data.tokenizer import SVGTokenizer
from src.eval.metrics import compute_perplexity, evaluate_samples
from src.eval.render import render_generated_samples
from src.generation.sampler import Sampler
from src.model.transformer import Transformer
from src.training.config import TrainConfig
from src.utils.config import get_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate SVG samples from a trained checkpoint.")
    p.add_argument("--run", required=True, help="Run name (e.g. large_01_mup). Checkpoint loaded from experiments/{run}/checkpoint.pt.")
    p.add_argument("--n_unconditional", type=int, default=10, help="Number of unconditional samples to generate.")
    p.add_argument(
        "--temperatures",
        nargs="+",
        type=float,
        default=[0.5, 0.8, 1.0],
        help="Temperature values to use for sampling.",
    )
    p.add_argument("--top_k", type=int, default=50, help="Top-k filtering (0 to disable).")
    p.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling (1.0 to disable).")
    p.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens per sample.")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    return p.parse_args()


def load_model(checkpoint_path: str | Path, device: str) -> tuple[Transformer, TrainConfig]:
    """
    Load a model from a checkpoint.
    Reconstructs TrainConfig from the saved model_config dict and handles µP if needed.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_config_dict = ckpt["model_config"]

    # Reconstruct TrainConfig from the saved dict
    config = TrainConfig(**{**model_config_dict, "device": device})
    model = Transformer(config).to(device)

    # Note: set_base_shapes is not needed for inference — mup multipliers only
    # affect the optimizer and have no effect on the forward pass.

    model.load_state_dict(ckpt["model"])
    model.eval()
    logger.info(
        f"Loaded checkpoint from {checkpoint_path} "
        f"(step={ckpt.get('step', '?')}, val_loss={ckpt.get('val_loss', '?'):.4f})"
    )
    return model, config


def load_prefixes(path: str | Path) -> list[tuple[str, str]]:
    """
    Load prefix prompts from a text file.
    Returns list of (label, svg_prefix) tuples.
    """
    path = Path(path)
    if not path.exists():
        logger.warning(f"Prefixes file not found: {path}. Skipping conditioned generation.")
        return []

    prefixes = []
    lines = path.read_text().splitlines()
    label = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Treat lines starting with # as labels for the following prompt
        if line.startswith("#"):
            label = line.lstrip("#").strip()
        else:
            prefixes.append((label or f"prefix_{len(prefixes)+1:02d}", line))
            label = None

    logger.info(f"Loaded {len(prefixes)} prefix prompts from {path}")
    return prefixes


def generate_and_save(
    sampler: Sampler,
    tokenizer: SVGTokenizer,
    prompt_ids: list[int],
    label: str,
    out_dir: Path,
    temperatures: list[float],
    top_k: int,
    top_p: float,
    max_new_tokens: int,
) -> list[str]:
    """
    Generate samples at each temperature for a single prompt.
    Saves .svg files and returns decoded SVG strings.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    svgs = []
    for temp in temperatures:
        temp_str = f"{temp:.1f}".replace(".", "p")
        token_ids = sampler.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temp,
            top_k=top_k if top_k > 0 else None,
            top_p=top_p if top_p < 1.0 else None,
        )
        svg = tokenizer.decode(token_ids, skip_special_tokens=True)
        svgs.append(svg)

        svg_path = out_dir / f"{label}_t{temp_str}.svg"
        svg_path.write_text(svg)

        logger.info(f"  [{label} at T={temp}] {len(token_ids)} tokens")

    return svgs


def main() -> None:
    args = parse_args()

    data_cfg = get_config("data/data")
    tokenizer = SVGTokenizer.from_pretrained(data_cfg["data_paths"]["tokenizer_dir"])
    bos_id = tokenizer.bos_token_id

    checkpoint_path = Path("experiments") / args.run / "checkpoint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    model, config = load_model(checkpoint_path, args.device)
    sampler = Sampler(model, tokenizer, device=args.device)

    run_name = args.run
    out_dir = Path("reports/figures/generated") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_svgs: list[str] = []

    # Generate unconditional samples
    logger.info(f"Generating {args.n_unconditional} unconditional samples...")
    for i in range(1, args.n_unconditional + 1):
        label = f"unconditional_{i:02d}"
        svgs = generate_and_save(
            sampler, tokenizer,
            prompt_ids=[bos_id],
            label=label,
            out_dir=out_dir,
            temperatures=args.temperatures,
            top_k=args.top_k,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )
        sample_svgs.extend(svgs)

    # Render generated sample SVGs to PNG
    logger.info("Rendering generated samples to PNG...")
    render_generated_samples(sample_svgs, out_dir=out_dir / "png")

    # Render a grid of generated samples
    logger.info("Rendering sample grid...")
    # Stick to one temperature for the grid (the middle one if multiple provided)
    grid_temp = sorted(args.temperatures)[len(args.temperatures) // 2]
    grid_temp_str = f"{grid_temp:.1f}".replace(".", "p")
    grid_svgs = [
        svg
        for path, svg in [
            (p, p.read_text())
            for p in sorted(out_dir.glob(f"*_t{grid_temp_str}.svg"))
        ]
    ]

    # Calculate quantitative metrics
    logger.info("Computing quantitative metrics...")
    metrics = evaluate_samples(sample_svgs)

    # Save metrics to JSON
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
