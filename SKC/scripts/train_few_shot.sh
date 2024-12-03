#!/bin/bash

#SBATCH -J train
#SBATCH -p a40
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --output=./jellyfish-rag/logs/train

# Set PYTHONPATH
export PYTHONPATH=$(pwd)

export GPU_NUM=0
export SUB_SIZE=-1

# Avoiding warning
export TOKENIZERS_PARALLELISM=false

# Set environment variables
export WANDB_MODE=offline
export WANDB_PROJECT=Mistral-jellyfish-lora

export VLLM_NO_DEPRECATION_WARNING=1

export CUDA_VISIBLE_DEVICES=0

# Set wandb and run scripts

export WANDB_NAME=jellyfish-rag-few-shot
llamafactory-cli train jellyfish-rag/scripts/few_shot.yaml

python jellyfish-rag/run_infer.py \
    --file_path "sotab2" \
    --save_path "jellyfish-rag/results/lora-7B/" \
    --model_weights "/home/yhge/pre-train/Jellyfish-7B" \
    --lora_weights "" \
    --enable_vllm 
    
