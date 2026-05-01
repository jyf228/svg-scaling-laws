# svg-scaling-laws

## Setup

```bash
conda env create -f environment.yaml
conda activate svg-scaling-laws
```

```bash
# Login to W&B
wandb login
```

## Run Data Processing Pipeline

```bash
python prepare_data.py --datasets svg-icons-simple --stats --render
```

## Learning Rate Sweep

```bash
python sweep.py --model tiny --device cuda      # use the default learning rates
python sweep.py --model tiny --device cuda --lrs 1e-3 3e-4 1e-4     # set your own learning rates
python sweep.py --model tiny --device cuda --mup   # use μP reparameterization
```

## Train a Model

```bash
python train.py --model tiny --device cuda --run_name tiny_run_01
python train.py --model tiny --device cuda --run_name tiny_run_01 --mup     # use μP reparameterization
```
