# svg-scaling-laws

## Setup

```bash
conda env create -f environment.yaml
conda activate svg-scaling-laws
```

## Run Data Processing Pipeline

```bash
python prepare_data.py --datasets svg-icons-simple --stats --render
```

## Learning Rate Sweep

```bash
python sweep.py --model tiny --device cuda
```

## Train a Model

```bash
python train.py --model tiny --device cuda
```
