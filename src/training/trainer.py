"""
Training loop for SVG scaling laws experiments.
"""

from contextlib import nullcontext
import logging
import math
from pathlib import Path
import time

import numpy as np
import torch
import wandb

from src.model.transformer import Transformer
from src.training.optimizer import build_optimizer
from src.training.logger import MetricsLogger

from src.utils.config import get_config
from src.training.config import TrainConfig


logger = logging.getLogger(__name__)

data_config = get_config("data/data")["data_paths"]
TRAIN = f"{data_config['processed_dir']}train.bin"
VAL = f"{data_config['processed_dir']}val.bin"


def _get_batch(
    path: str,
    batch_size: int,
    block_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a random batch from a tokenized .bin file.

    The memmap is recreated on every call to avoid a memory leak as discussed in demo-rnns.ipynb:
    https://colab.research.google.com/drive/1R1T7PlKjuUISzgulTQ8wX4RwUBB4crDo?usp=sharing
    """
    data = np.memmap(path, dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + block_size + 1].astype(np.int64)) for i in ix])

    if "cuda" in str(device):
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)

    return x, y


def _get_lr(step: int, config: TrainConfig) -> float:
    """
    Cosine learning rate schedule with linear warmup.
    Adapted from nanoGPT's implementation of the learning rate schedule.
    """
    lr = config.learning_rate
    min_lr = config.min_lr
    warmup_iters = config.warmup_steps
    lr_decay_iters = config.total_steps

    # 1. Linear warmup for the first `warmup_iters` steps
    if step < warmup_iters:
        return lr * (step + 1) / (warmup_iters + 1)
    
    # 2. If step > lr_decay_iters, return min learning rate
    if step > lr_decay_iters:
        return min_lr
    
    # 3. In between, use cosine decay down to min learning rate
    decay_ratio = (step - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (lr - min_lr)


@torch.no_grad()
def estimate_loss(model: Transformer, config: TrainConfig, ctx) -> dict[str, float]:
    out = {}
    model.eval()
    for path, split in [(TRAIN, "train"), (VAL, "val")]:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = _get_batch(path, config.micro_batch_size, config.block_size, config.device)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def _save_checkpoint(model: Transformer, config: TrainConfig, step: int, val_loss: float) -> None:
    run_dir = Path(config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "checkpoint.pt"
    torch.save(
        {
            "step":       step,
            "val_loss":   val_loss,
            "model":      model.state_dict(),
            "model_config": vars(model.config),
        },
        ckpt_path,
    )
    wandb.save(str(ckpt_path), base_path=str(run_dir))
    logger.info(f"Checkpoint saved -> {ckpt_path}  (val_loss={val_loss:.4f})")


def train(config: TrainConfig) -> None:
    """
    Training loop.
    Adapted from nanoGPT's training loop.
    """
    torch.manual_seed(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    ptdtype = torch.bfloat16
    ctx = nullcontext() if config.device == 'cpu' else torch.amp.autocast(device_type=config.device, dtype=ptdtype)
    
    # Initialize model, optimizer, and metrics logger
    model = Transformer(config).to(config.device)
    optimizer = build_optimizer(model, config)
    metrics = MetricsLogger(config)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    batch_size = config.batch_size_tokens // config.block_size  # full effective batch size
    grad_accum_steps = max(1, batch_size // config.micro_batch_size)  # number of steps to accumulate gradients before an optimizer step
    config.micro_batch_size = batch_size // grad_accum_steps

    tokens_per_step = config.micro_batch_size * config.block_size * grad_accum_steps  # number of tokens processed per optimizer step 
    n_train_tokens = len(np.memmap(TRAIN, dtype=np.uint16, mode="r"))
    total_steps = n_train_tokens // tokens_per_step  # total iterations for 1 epoch through the training data
    config.total_steps = total_steps
    if config.warmup_steps is None:
        config.warmup_steps = total_steps // 10
    config.eval_interval = min(config.eval_interval, max(1, total_steps // 10))

    logger.info(
        f"Training: steps={total_steps}  micro_batch={config.micro_batch_size}  grad_accum={grad_accum_steps}  tokens/step={tokens_per_step}  n_train_tokens={n_train_tokens}"
    )

    model.train()
    accum_loss = 0.0
    epoch_start = time.perf_counter()
    tokens_processed = 0
    for step in range(total_steps):
        # Get and set the learning rate for this step
        lr = _get_lr(step, config)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Forward backward update, with gradient accumulation to simulate larger batch size
        for _ in range(grad_accum_steps):
            x, y = _get_batch(TRAIN, config.micro_batch_size, config.block_size, config.device)
            # Compute loss and backprop
            with ctx:
                _, loss = model(x, y)
            loss = loss / grad_accum_steps   # scale the loss to account for gradient accumulation
            loss.backward()
            accum_loss += loss.item()

        # Gradient clipping
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        train_loss = accum_loss
        accum_loss = 0.0
        tokens_processed += tokens_per_step

        if step % config.log_interval == 0:
            elapsed = time.perf_counter() - epoch_start
            tok_per_sec = tokens_processed / elapsed if elapsed > 0 else 0.0
            gpu_mem_mb = (
                torch.cuda.max_memory_allocated() / 1e6
                if config.device == "cuda" else 0.0
            )
            metrics.log(
                step,
                train_loss=round(train_loss, 6),
                lr=lr,
                tokens_per_sec=round(tok_per_sec, 1),
                gpu_mem_mb=round(gpu_mem_mb, 1),
            )
            logger.info(
                f"step {step}/{total_steps}  loss={train_loss:.4f}  lr={lr:.2e}  "
                f"tok/s={tok_per_sec:.0f}  gpu_mem={gpu_mem_mb:.0f}MB"
            )

        if (step > 0 and step % config.eval_interval == 0) or (step == total_steps - 1):
            losses = estimate_loss(model, config, ctx)
            metrics.log(step, train_loss_eval=round(losses["train"], 6), val_loss=round(losses["val"], 6))
            logger.info(f"step {step}  train_loss_eval={losses['train']:.4f}  val_loss={losses['val']:.4f}")

    # Final evaluation for the scaling plot
    epoch_time_s = time.perf_counter() - epoch_start
    final_losses = estimate_loss(model, config, ctx)
    final_val_loss = final_losses["val"]
    gpu_mem_mb = (
        torch.cuda.max_memory_allocated() / 1e6
        if config.device == "cuda" else 0.0
    )
    logger.info(
        f"Training complete. Final val_loss={final_val_loss:.4f}  "
        f"epoch_time={epoch_time_s:.1f}s  gpu_mem={gpu_mem_mb:.0f}MB"
    )
    metrics.log(
        total_steps,
        train_loss_eval=round(final_losses["train"], 6),
        val_loss=round(final_val_loss, 6),
        epoch_time_s=round(epoch_time_s, 2),
        gpu_mem_mb=round(gpu_mem_mb, 1),
    )
    metrics.log_final(final_val_loss, n_params=n_params)

    _save_checkpoint(model, config, step=total_steps, val_loss=final_val_loss)
    metrics.close()
    return final_val_loss
