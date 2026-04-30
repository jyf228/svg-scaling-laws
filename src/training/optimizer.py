"""
Optimizer construction for training.
"""

import logging
import inspect

import torch
from mup import MuAdamW

from src.training.config import TrainConfig


logger = logging.getLogger(__name__)


def build_optimizer(model: torch.nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    """
    Build the optimizer for a model. 
    Uses mup optimizer if configured for µP training, otherwise AdamW.
    """
    if config.optimizer == "mup":
        return _build_mup_optimizer(model, config)
    return _build_adamw_optimizer(model, config)


def _build_adamw_optimizer(model: torch.nn.Module, config: TrainConfig) -> torch.optim.AdamW:
    """
    Standard AdamW optimizer.
    Adapted from nanoGPT's optimizer configuration:
    https://github.com/karpathy/nanoGPT/blob/master/model.py#L263
    """
    # Start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
        
    # Filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

    # Create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    n_decay = sum(p.numel() for p in decay_params)
    n_nodecay = sum(p.numel() for p in nodecay_params)
    logger.info(f"Number decayed parameter tensors: {len(decay_params)}, with {n_decay:,} parameters")
    logger.info(f"Number non-decayed parameter tensors: {len(nodecay_params)}, with {n_nodecay:,} parameters")

    # Create AdamW optimizer and use the fused version if it is available
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and config.device == "cuda"
    extra_args = dict(fused=True) if use_fused else dict()
    if use_fused:
        logger.info("Using fused AdamW kernel.")

    return torch.optim.AdamW(
        optim_groups,
        lr=config.learning_rate,
        betas=config.betas,
        **extra_args,
    )


def _build_mup_optimizer(model: torch.nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    """
    µP-aware optimizer using MuAdamW.
    """
    param_dict = {n: p for n, p in model.named_parameters() if p.requires_grad}

    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

    optim_groups = [
        {"params": decay_params,   "weight_decay": config.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    n_decay = sum(p.numel() for p in decay_params)
    n_nodecay = sum(p.numel() for p in nodecay_params)
    logger.info(f"Number decayed parameter tensors (µP): {len(decay_params)}, with {n_decay:,} parameters")
    logger.info(f"Number non-decayed parameter tensors (µP): {len(nodecay_params)}, with {n_nodecay:,} parameters")

    return MuAdamW(
        optim_groups,
        lr=config.learning_rate,
        betas=config.betas,
    )
