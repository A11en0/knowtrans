#!/bin/bash

#SBATCH -J train
#SBATCH -p a40
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --output=experiments/SKC/downstream-train.out

# Set PYTHONPATH
export PYTHONPATH=$(pwd)

export GPU_NUM=0
export SUB_SIZE=-1

# Avoiding warning
export TOKENIZERS_PARALLELISM=false

# Set environment variables
export WANDB_MODE=offline
export WANDB_PROJECT=KnowPrep-7B-weight

export VLLM_NO_DEPRECATION_WARNING=1

export CUDA_VISIBLE_DEVICES=0


export WANDB_NAME=few-shot-weight
llamafactory-cli train scripts/downstram-train/SKC.yaml
