from dataclasses import dataclass

@dataclass(kw_only=True)
class TrainConfig:
    """
    All hyperparameters used during training.
    """
    # Model Architecture (shared/core.yaml + model/{size}.yaml)
    n_layer: int
    n_head: int
    d_model: int
    d_ff: int
    vocab_size: int
    block_size: int
    bias: bool = False
    dropout: float = 0.0

    # Training (training/base.yaml)
    optimizer: str = "adamw"
    batch_size_tokens: int = 524288
    micro_batch_size: int 
    total_steps: int = 0    # Set during training
    learning_rate: float
    min_lr: float = 0.0     # Set based on learning rate
    warmup_steps: int | None = None     # Set based on total steps during training if not provided
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    betas: tuple[float, float] = (0.9, 0.95)
    seed: int = 1337

    # Device
    device: str = "cpu"

    # Logging / evaluation (training/base.yaml)
    log_interval: int = 10
    eval_interval: int = 500
    eval_iters: int = 50

    # Run IDs
    run_name: str = ""
    run_dir: str = ""
    wandb_project: str = "svg-scaling-laws"

    def __post_init__(self) -> None:
        """
        Set derived fields.
        """
        if not self.run_dir and self.run_name:
            self.run_dir = f"experiments/{self.run_name}"
        if self.min_lr == 0.0:
            self.min_lr = self.learning_rate / 10
        # YAML lists become Python lists; normalise betas to tuple
        if not isinstance(self.betas, tuple):
            self.betas = tuple(self.betas)
