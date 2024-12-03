#!/bin/bash

#SBATCH -J train500
#SBATCH -p a40
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --output=./jellyfish-rag/logs/train500.log

# Set PYTHONPATH
export PYTHONPATH=$(pwd)

export GPU_NUM=0
export SUB_SIZE=-1

# Avoiding warning
export TOKENIZERS_PARALLELISM=false

# Set environment variables
export WANDB_MODE=offline
export WANDB_PROJECT=knowprep-7B-scalability

export VLLM_NO_DEPRECATION_WARNING=1

export CUDA_VISIBLE_DEVICES=0

source activate dpmoe

export WANDB_NAME=knowprep-7B-weight-train500
llamafactory-cli train jellyfish-rag/scripts/upstream-train-7B/scalability/few-shot-train500.yaml

python jellyfish-rag/run_infer.py \
    --file_path "abt_buy,ae,beers_DC,beers,cms,flights,flipkart,oa,phone,rayyan_DC,rayyan,walmart_amazon" \
    --save_path "jellyfish-rag/results/7B/few-shot-train500" \
    --lora_weights "jellyfish-rag/outputs/7B/few-shot-train500" \
    --base_data_dir "data/Jellyfish-rag/test" \
    --model_weights "/share/home/12351018/pre-train/Jellyfish-7B" \
    --model_type "mistral" \
    --overwrite \
    --enable_vllm \

    