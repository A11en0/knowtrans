#!/bin/bash

#SBATCH -J 00
#SBATCH -p a40-high
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --output=./jellyfish-rag/logs/00

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


export WANDB_NAME=knowprep-7B-weight-r64
# llamafactory-cli train jellyfish-rag/scripts/upstream-train-7B/rank/few-shot-merge-weight-r64.yaml

python jellyfish-rag/run_infer.py \
    --file_path "abt_buy" \
    --save_path "jellyfish-rag/results/7B/few-shot-train20-weight-r64" \
    --base_data_dir "data/Jellyfish-rag/test" \
    --model_weights "/share/home/12351018/pre-train/Jellyfish-7B" \
    --lora_weights "jellyfish-rag/outputs/7B/few-shot-train20-weight-r64" \
    --model_type "mistral" \
    --overwrite    



