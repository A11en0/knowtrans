#!/bin/bash

#SBATCH -J train
#SBATCH -p a40-high
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --output=./jellyfish-rag/logs/weight

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
export WANDB_NAME=jellyfish-13B-few-shot-merge-weight
llamafactory-cli train jellyfish-rag/scripts/upstream-train-13B/few-shot-merge-weight.yaml

python jellyfish-rag/run_infer.py \
    --file_path "abt_buy,ae,beers_DC,beers,cms,flights,flipkart,hospital_DC,oa,phone,rayyan_DC,rayyan,walmart_amazon" \
    --save_path "jellyfish-rag/results/13B/few-shot-train20-weight" \
    --base_data_dir "data/Jellyfish-rag/test" \
    --model_weights "/share/home/12351018/pre-train/Jellyfish-13B" \
    --lora_weights "jellyfish-rag/outputs/13B/few-shot-train20-weight" \
    --model_type "jellyfish-13b" \
    --overwrite