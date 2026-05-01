import logging
from dataclasses import asdict

import wandb

from src.training.config import TrainConfig

logger = logging.getLogger(__name__)


class TrainLogger:
    """Log training metrics to wandb."""

    def __init__(self, config: TrainConfig) -> None:
        self._run = wandb.init(
            project=config.wandb_project,
            name=config.run_name,
            config=asdict(config),
            dir=config.run_dir,
        )
        logger.info("wandb run: %s", self._run.url)

    def log(self, step: int, **kwargs) -> None:
        wandb.log({"step": step, **kwargs}, step=step)

    def log_final(self, val_loss: float, n_params: int) -> None:
        wandb.log({"final_val_loss": val_loss})
        wandb.summary["final_val_loss"] = val_loss
        wandb.summary["n_params"] = n_params

    def close(self) -> None:
        wandb.finish()
