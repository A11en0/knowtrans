#!/bin/bash

#SBATCH -J train
#SBATCH -p a40-high
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --output=./jellyfish-rag/logs/0

# Set PYTHONPATH
export PYTHONPATH=$(pwd)

export GPU_NUM=0
export SUB_SIZE=-1

# Avoiding warning
export TOKENIZERS_PARALLELISM=false

# Set environment variables
export WANDB_MODE=offline
export WANDB_PROJECT=KnowPrep-7B-LR

export VLLM_NO_DEPRECATION_WARNING=1

export CUDA_VISIBLE_DEVICES=0


export WANDB_NAME=knowprep-7B-weight-1e-5
llamafactory-cli train jellyfish-rag/scripts/upstream-train-7B/lr/few-shot-merge-weight-1e-5.yaml

export WANDB_NAME=knowprep-7B-weight-8e-5
llamafactory-cli train jellyfish-rag/scripts/upstream-train-7B/lr/few-shot-merge-weight-8e-5.yaml

export WANDB_NAME=knowprep-7B-weight-6e-6
llamafactory-cli train jellyfish-rag/scripts/upstream-train-7B/lr/few-shot-merge-weight-6e-6.yaml


python jellyfish-rag/run_infer.py \
    --file_path "abt_buy,ae,beers_DC,beers,cms,flights,flipkart,hospital_DC,oa,phone,rayyan_DC,rayyan,walmart_amazon" \
    --save_path "jellyfish-rag/results/7B/few-shot-train20-weight-1e-5" \
    --base_data_dir "data/Jellyfish-rag/test" \
    --model_weights "/share/home/12351018/pre-train/Jellyfish-7B" \
    --lora_weights "jellyfish-rag/outputs/7B/few-shot-train20-weight-1e-5" \
    --model_type "mistral" \
    --overwrite

python jellyfish-rag/run_infer.py \
    --file_path "abt_buy,ae,beers_DC,beers,cms,flights,flipkart,hospital_DC,oa,phone,rayyan_DC,rayyan,walmart_amazon" \
    --save_path "jellyfish-rag/results/7B/few-shot-train20-weight-8e-5" \
    --base_data_dir "data/Jellyfish-rag/test" \
    --model_weights "/share/home/12351018/pre-train/Jellyfish-7B" \
    --lora_weights "jellyfish-rag/outputs/7B/few-shot-train20-weight-8e-5" \
    --model_type "mistral" \
    --overwrite

python jellyfish-rag/run_infer.py \
    --file_path "abt_buy,ae,beers_DC,beers,cms,flights,flipkart,hospital_DC,oa,phone,rayyan_DC,rayyan,walmart_amazon" \
    --save_path "jellyfish-rag/results/7B/few-shot-train20-weight-6e-6" \
    --base_data_dir "data/Jellyfish-rag/test" \
    --model_weights "/share/home/12351018/pre-train/Jellyfish-7B" \
    --lora_weights "jellyfish-rag/outputs/7B/few-shot-train20-weight-6e-6" \
    --model_type "mistral" \
    --overwrite


