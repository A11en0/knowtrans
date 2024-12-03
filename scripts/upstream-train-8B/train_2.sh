#!/bin/bash

#SBATCH -J train
#SBATCH -p a40-high
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --output=./jellyfish-rag/logs/train-2

# Set PYTHONPATH
export PYTHONPATH=$(pwd)

export GPU_NUM=0
export SUB_SIZE=-1

# Avoiding warning
export TOKENIZERS_PARALLELISM=false

# Set environment variables
export WANDB_MODE=offline
export WANDB_PROJECT=llama3-8B

export VLLM_NO_DEPRECATION_WARNING=1

export CUDA_VISIBLE_DEVICES=0


# Set wandb and run scripts
export WANDB_NAME=llama3-8B-dblp_acm
llamafactory-cli train jellyfish-rag/scripts/upstream-train-8B/dblp_acm.yaml

# Set wandb and run scripts
export WANDB_NAME=llama3-8B-dblp_scholar
llamafactory-cli train jellyfish-rag/scripts/upstream-train-8B/dblp_scholar.yaml

# Set wandb and run scripts
export WANDB_NAME=llama3-8B-restaurant
llamafactory-cli train jellyfish-rag/scripts/upstream-train-8B/restaurant.yaml

# Set wandb and run scripts
export WANDB_NAME=llama3-8B-fodors_zagat
llamafactory-cli train jellyfish-rag/scripts/upstream-train-8B/fodors_zagat.yaml

# python jellyfish-rag/run_infer.py \
#     --file_path "abt_buy,ae,beers_DC,beers,cms,enterprise,flights,flipkart,hospital_DC,oa,phone,rayyan_DC,rayyan,sotab2,walmart_amazon" \
#     --save_path "jellyfish-rag/results/lora-embedding-k512-8B" \
#     --model_weights "/home/yhge/pre-train/Jellyfish-8B" \
#     --lora_weights "jellyfish-rag/outputs/lora-embedding-k512-8B" \
#     --enable_vllm
