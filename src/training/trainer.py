"""
Training loop for SVG scaling laws experiments.
"""

from contextlib import nullcontext
import dataclasses
import logging
import math
from pathlib import Path
import time

import numpy as np
import torch
import wandb
from mup import set_base_shapes

from src.model.transformer import Transformer
from src.training.optimizer import build_optimizer
from src.training.logger import TrainLogger

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
        absolute_lr = lr * (step + 1) / (warmup_iters + 1)
    # 2. If step > lr_decay_iters, return min learning rate
    elif step > lr_decay_iters:
        absolute_lr = min_lr
    # 3. In between, use cosine decay down to min learning rate
    else:
        decay_ratio = (step - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        absolute_lr = min_lr + coeff * (lr - min_lr)

    return absolute_lr / lr


@torch.no_grad()
def _estimate_loss(model: Transformer, config: TrainConfig, ctx) -> dict[str, float]:
    """Estimate the loss on training and validation sets."""
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


def _save_checkpoint(
    model: Transformer, 
    config: TrainConfig, 
    step: int, 
    val_loss: float,
    epoch: int | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    """Save a checkpoint to disk and upload to wandb."""
    run_dir = Path(config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    # Always overwrite the latest checkpoint
    ckpt_path = run_dir / "checkpoint.pt"
    payload = {
        "step": step,
        "epoch": epoch,
        "val_loss": val_loss,
        "model": model.state_dict(),
        "model_config": vars(model.config),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
    }
    torch.save(payload, ckpt_path)

    if epoch is not None:
        epoch_ckpt = run_dir / f"checkpoint_epoch_{epoch:02d}.pt"
        torch.save(payload, epoch_ckpt)
        logger.info(f"Checkpoint saved at {epoch_ckpt} (epoch={epoch}, val_loss={val_loss:.4f})")
    else:
        logger.info(f"Checkpoint saved at {ckpt_path} (val_loss={val_loss:.4f})")
    wandb.save(str(ckpt_path), base_path=str(run_dir))


def _load_checkpoint(path: str | Path, model: Transformer, optimizer: torch.optim.Optimizer | None = None) -> dict:
    """Load model (and optionally optimizer) weights from a checkpoint. Returns the checkpoint dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    epoch = ckpt.get("epoch", 0) or 0
    step  = ckpt.get("step",  0) or 0
    val_loss = ckpt.get("val_loss", float("inf"))
    logger.info(f"Resumed from {path} (epoch={epoch}, step={step}, val_loss={val_loss:.4f})")
    return ckpt


def _set_base_shapes_mup(model: Transformer, config: TrainConfig, d_model_base: int, rescale_params: bool = True) -> None:
    """Set base shapes for µP training."""

    def _mup_config(base: TrainConfig, d_model: int) -> TrainConfig:
        """Builds a new TrainConfig for µP shape construction."""
        # Preserve the ratio between d_ff and d_model when changing d_model.
        ff_model_ratio = base.d_ff / base.d_model
        return dataclasses.replace(
            base,
            d_model=d_model,
            d_ff=int(round(d_model * ff_model_ratio)),
            # Keep architecture consistent with the target model for µP shape inference.
            n_head=base.n_head,
            use_mup=False,
        )

    # Delta should differ in width and remain divisible by n_head.
    delta_width = d_model_base + max(1, config.n_head)

    # Instantiate a base model
    base_cfg = _mup_config(config, d_model_base)
    base_model = Transformer(base_cfg)

    # Instantiate a "delta" model that differs from the base only in d_model
    delta_cfg = _mup_config(config, delta_width)
    delta_model = Transformer(delta_cfg)
    
    set_base_shapes(model, base_model, delta=delta_model, rescale_params=rescale_params)
    if rescale_params:
        model.reinit_weights_mup()  # reinitialize weights after setting base shapes
    logger.info(
        f"µP base shapes set (d_model_base={d_model_base}, d_model_delta={delta_width}, n_head={config.n_head})"
    )

    # We can delete after setting base shapes since they're not used for training
    del base_model, delta_model


def train(config: TrainConfig, resume_from: str | Path | None = None) -> None:
    """Training loop. Adapted from nanoGPT's training loop."""
    torch.manual_seed(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    ptdtype = torch.bfloat16
    ctx = nullcontext() if config.device == 'cpu' else torch.amp.autocast(device_type=config.device, dtype=ptdtype)
    
    # Initialize model, optimizer, and metrics logger
    model = Transformer(config).to(config.device)

    if config.use_mup:
        _set_base_shapes_mup(model, config, config.d_model_base)

    optimizer = build_optimizer(model, config)
    # Snapshot the (μP-scaled) base LR for each param group immediately after
    # optimizer construction. The scheduler will multiply these by a ratio so
    # that per-group μP scaling is preserved rather than overwritten absolutely.
    base_lrs = [group["lr"] for group in optimizer.param_groups]

    # Optionally resume from a checkpoint
    start_epoch = 1
    if resume_from is not None:
        ckpt = _load_checkpoint(resume_from, model, optimizer)
        start_epoch = (ckpt.get("epoch") or 0) + 1
        logger.info(f"Resuming training from epoch {start_epoch}")
        # torch.save does not persist infshape objects attached to parameter
        # tensors, so we must re-apply base shapes after loading.
        if config.use_mup:
            _set_base_shapes_mup(model, config, config.d_model_base, rescale_params=False)

    metrics = TrainLogger(config)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    batch_size = config.batch_size_tokens // config.block_size  # full effective batch size
    grad_accum_steps = max(1, batch_size // config.micro_batch_size)  # number of steps to accumulate gradients before an optimizer step
    config.micro_batch_size = batch_size // grad_accum_steps

    tokens_per_step = config.micro_batch_size * config.block_size * grad_accum_steps  # number of tokens processed per optimizer step 
    n_train_tokens = len(np.memmap(TRAIN, dtype=np.uint16, mode="r"))
    steps_per_epoch = n_train_tokens // tokens_per_step
    total_steps = steps_per_epoch * config.n_epochs
    config.total_steps = total_steps
    if config.warmup_steps is None:
        config.warmup_steps = steps_per_epoch // 10
    config.eval_interval = min(config.eval_interval, max(1, steps_per_epoch // 10))

    logger.info(
        f"Training: epochs={config.n_epochs}  steps/epoch={steps_per_epoch}  total_steps={total_steps}  "
        f"micro_batch={config.micro_batch_size}  grad_accum={grad_accum_steps}  "
        f"tokens/step={tokens_per_step}  n_train_tokens={n_train_tokens}"
    )

    model.train()
    global_step = 0
    best_val_loss = float("inf")
    train_start = time.perf_counter()
    tokens_processed = 0

    for epoch in range(start_epoch, config.n_epochs + 1):
        logger.info(f"--- Epoch {epoch}/{config.n_epochs} ---")
        epoch_start = time.perf_counter()
        accum_loss = 0.0

        for local_step in range(steps_per_epoch):
            # Get and set the learning rate for this step.
            # We apply the schedule as a multiplier on the μP-scaled base LRs
            # per https://github.com/microsoft/mup#current-limitations.
            lr_multiplier = _get_lr(global_step, config)
            for group, base_lr in zip(optimizer.param_groups, base_lrs):
                group["lr"] = base_lr * lr_multiplier
            lr = lr_multiplier * config.learning_rate  # for logging

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

            if global_step % config.log_interval == 0:
                elapsed = time.perf_counter() - train_start
                tok_per_sec = tokens_processed / elapsed if elapsed > 0 else 0.0
                gpu_mem_mb = (
                    torch.cuda.max_memory_allocated() / 1e6
                    if config.device == "cuda" else 0.0
                )
                metrics.log(
                    global_step,
                    train_loss=round(train_loss, 6),
                    lr=lr,
                    tokens_per_sec=round(tok_per_sec, 1),
                    gpu_mem_mb=round(gpu_mem_mb, 1),
                    epoch=epoch,
                )
                logger.info(
                    f"epoch {epoch}  step {local_step}/{steps_per_epoch} (global {global_step})  "
                    f"loss={train_loss:.4f}  lr={lr:.2e}  tok/s={tok_per_sec:.0f}"
                )

            if global_step % config.eval_interval == 0:
                losses = _estimate_loss(model, config, ctx)
                metrics.log(global_step, train_loss_eval=round(losses["train"], 6), val_loss=round(losses["val"], 6), epoch=epoch)
                logger.info(f"epoch {epoch}  step {global_step}  train_loss_eval={losses['train']:.4f}  val_loss={losses['val']:.4f}")

            global_step += 1

        epoch_time_s = time.perf_counter() - epoch_start
        epoch_losses = _estimate_loss(model, config, ctx)
        epoch_val_loss = epoch_losses["val"]
        gpu_mem_mb = (
            torch.cuda.max_memory_allocated() / 1e6
            if config.device == "cuda" else 0.0
        )
        logger.info(
            f"Epoch {epoch} complete. val_loss={epoch_val_loss:.4f}  "
            f"time={epoch_time_s:.1f}s  gpu_mem={gpu_mem_mb:.0f}MB"
        )
        metrics.log(
            global_step,
            train_loss_eval=round(epoch_losses["train"], 6),
            val_loss=round(epoch_val_loss, 6),
            epoch_time_s=round(epoch_time_s, 2),
            gpu_mem_mb=round(gpu_mem_mb, 1),
            epoch=epoch,
        )
        _save_checkpoint(model, config, step=global_step, val_loss=epoch_val_loss, epoch=epoch, optimizer=optimizer)
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss

    total_time_s = time.perf_counter() - train_start
    logger.info(f"Training complete. Best val_loss={best_val_loss:.4f}  total_time={total_time_s:.1f}s")
    metrics.log_final(best_val_loss, n_params=n_params)
    metrics.close()
    return best_val_loss
