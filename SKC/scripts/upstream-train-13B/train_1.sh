#!/bin/bash

#SBATCH -J train
#SBATCH -p a40-high
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --output=./jellyfish-rag/logs/train-1

# Set PYTHONPATH
export PYTHONPATH=$(pwd)

export GPU_NUM=0
export SUB_SIZE=-1

# Avoiding warning
export TOKENIZERS_PARALLELISM=false

# Set environment variables
export WANDB_MODE=offline
export WANDB_PROJECT=jellyfish-13B

export VLLM_NO_DEPRECATION_WARNING=1

export CUDA_VISIBLE_DEVICES=0


# Set wandb and run scripts
export WANDB_NAME=jellyfish-13B-beers
llamafactory-cli train jellyfish-rag/scripts/upstream-train-13B/beers.yaml

# Set wandb and run scripts
export WANDB_NAME=jellyfish-13B-buy
llamafactory-cli train jellyfish-rag/scripts/upstream-train-13B/buy.yaml

