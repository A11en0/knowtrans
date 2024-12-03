#!/bin/bash

#SBATCH -J train
#SBATCH -p a40-high
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
export WANDB_PROJECT=llama3-8B

export VLLM_NO_DEPRECATION_WARNING=1

export CUDA_VISIBLE_DEVICES=0





# python jellyfish-rag/run_infer.py \
#     --file_path "abt_buy,ae,beers_DC,beers,cms,enterprise,flights,flipkart,hospital_DC,oa,phone,rayyan_DC,rayyan,sotab2,walmart_amazon" \
#     --save_path "jellyfish-rag/results/lora-embedding-k512-8B" \
#     --model_weights "/home/yhge/pre-train/Jellyfish-8B" \
#     --lora_weights "jellyfish-rag/outputs/lora-embedding-k512-8B" \
#     --enable_vllm
