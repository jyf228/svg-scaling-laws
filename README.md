# svg-scaling-laws

## Setup

```bash
conda env create -f environment.yaml
conda activate svg-scaling-laws

# Login to W&B
wandb login
```

## Run Data Processing Pipeline

```bash
python prepare_data.py --datasets svg-icons-simple svg-emoji-simple svg-stack-simple --stats --render
```

## Learning Rate Sweep

```bash
python lr_sweep.py --model tiny --device cuda      # use the default learning rates
python lr_sweep.py --model tiny --device cuda --lrs .001 .05 .01     # set your own learning rates
python lr_sweep.py --model tiny --device cuda --mup   # use μP reparameterization
```

## Train a Model

```bash
python train.py --model tiny --device cuda --learning_rate 3e-3 --run_name tiny_run_01
python train.py --model tiny --device cuda --learning_rate 3e-3 --run_name tiny_run_01 --mup     # use μP reparameterization
```

## Train a Family of Models

`scripts/train_model_family.sh` can be used to train a family of models. Edit it to specify the model sizes and command line args according to what you need.

```bash
chmod +x scripts/train_model_family.sh
./scripts/train_model_family.sh
```

## Sample Generation

```bash
python generate.py --run xl_01 --temperatures 0.5 --top_k 50 --top_p 0.9
```
