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

python jellyfish-rag/run_infer.py \
    --file_path "abt_buy,ae,beers_DC,beers,cms,flights,flipkart,oa,phone,rayyan_DC,rayyan,walmart_amazon,sotab3" \
    --save_path "jellyfish-rag/results/7B/few-shot-train20-weight-6e-6" \
    --base_data_dir "data/Jellyfish-rag/test" \
    --model_weights "/home/yhge/pre-train/Jellyfish-7B" \
    --lora_weights "jellyfish-rag/outputs/7B/few-shot-train20" \
    --model_type "mistral" \
    --enable_vllm \
    --overwrite
    