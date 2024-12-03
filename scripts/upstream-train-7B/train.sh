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
export WANDB_PROJECT=KnowPrep-7B-weight

export VLLM_NO_DEPRECATION_WARNING=1

export CUDA_VISIBLE_DEVICES=0


export WANDB_NAME=few-shot-weight
llamafactory-cli train jellyfish-rag/scripts/upstream-train-7B/few-shot-merge-weight.yaml

# python jellyfish-rag/run_infer.py \
#     --file_path "abt_buy,ae,beers_DC,beers,cms,flights,flipkart,hospital_DC,oa,phone,rayyan_DC,rayyan,walmart_amazon" \
#     --save_path "jellyfish-rag/results/7B/jellyfish-7B-few-shot-train20-merge-weight" \
#     --base_data_dir "data/Jellyfish-rag/test" \
#     --model_weights "/home/yhge/pre-train/Jellyfish-7B" \
#     --lora_weights "jellyfish-rag/outputs/7B/few-shot-train20-weight" \
#     --model_type "mistral" \
#     --overwrite


# # Set wandb and run scripts
# export WANDB_NAME=mistral-7B-adult
# llamafactory-cli train jellyfish-rag/scripts/upstream-train-7B/adult.yaml

# # Set wandb and run scripts
# export WANDB_NAME=mistral-7B-amazon_google
# llamafactory-cli train jellyfish-rag/scripts/upstream-train-7B/amazon_google.yaml

# # Set wandb and run scripts
# export WANDB_NAME=mistral-7B-sm
# llamafactory-cli train jellyfish-rag/scripts/upstream-train-7B/sm.yaml

# export WANDB_NAME=mistral-7B-few-shot
# llamafactory-cli train jellyfish-rag/scripts/upstream-train-7B/mistral-few-shot-merge.yaml

# python jellyfish-rag/run_infer.py \
#     --file_path "abt_buy,ae,beers_DC,beers,cms,flights,flipkart,hospital_DC,oa,phone,rayyan_DC,rayyan,walmart_amazon" \
#     --save_path "jellyfish-rag/results/7B/few-shot-train20-merge" \
#     --base_data_dir "data/Jellyfish-rag/mini_test" \
#     --model_weights "jellyfish-rag/outputs/7B/upstream-task/full_model" \
#     --lora_weights "jellyfish-rag/outputs/7B/few-shot-train20-merge" \
#     --model_type "mistral" \
#     --enable_vllm \
#     --overwrite    

# --base_data_dir "data/Jellyfish-rag/mini_test" \
# python jellyfish-rag/run_infer.py \
#     --file_path "abt_buy,ae,beers_DC,beers,cms,flights,flipkart,hospital_DC,oa,phone,rayyan_DC,rayyan,walmart_amazon" \
#     --save_path "jellyfish-rag/results/7B/few-shot-train20-merge-jellyfish-fulltest" \
#     --base_data_dir "data/Jellyfish-rag/test" \
#     --model_weights "/home/yhge/pre-train/Jellyfish-7B" \
#     --lora_weights "jellyfish-rag/outputs/7B/few-shot-train20-merge" \
#     --model_type "mistral" \
#     --enable_vllm \
#     --overwrite        

# export WANDB_NAME=mistral-7B-few-shot
# llamafactory-cli train jellyfish-rag/scripts/upstream-train-7B/few-shot.yaml

# export WANDB_NAME=mistral-7B-few-shot-merge
# llamafactory-cli train jellyfish-rag/scripts/upstream-train-7B/jellyfish-few-shot-merge.yaml

# python jellyfish-rag/run_infer.py \
#     --file_path "abt_buy,ae,beers_DC,beers,cms,flights,flipkart,hospital_DC,oa,phone,rayyan_DC,rayyan,walmart_amazon" \
#     --save_path "jellyfish-rag/results/7B/jellyfish-7B" \
#     --base_data_dir "data/Jellyfish-rag/test" \
#     --model_weights "/home/yhge/pre-train/Jellyfish-7B" \
#     --lora_weights "" \
#     --model_type "mistral" \
#     --enable_vllm \
#     --overwrite

# python jellyfish-rag/run_infer.py \
#     --file_path "abt_buy,ae,beers_DC,beers,cms,flights,flipkart,hospital_DC,oa,phone,rayyan_DC,rayyan,walmart_amazon" \
#     --save_path "jellyfish-rag/results/7B/jellyfish-7B-few-shot" \
#     --base_data_dir "data/Jellyfish-rag/test" \
#     --model_weights "/home/yhge/pre-train/Jellyfish-7B" \
#     --lora_weights "jellyfish-rag/outputs/7B/few-shot-train20" \
#     --model_type "mistral" \
#     --enable_vllm \
#     --overwrite

# python jellyfish-rag/run_infer.py \
#     --file_path "abt_buy,ae,beers_DC,beers,cms,flights,flipkart,hospital_DC,oa,phone,rayyan_DC,rayyan,walmart_amazon" \
#     --save_path "jellyfish-rag/results/7B/jellyfish-7B-few-shot-train20-merge" \
#     --base_data_dir "data/Jellyfish-rag/test" \
#     --model_weights "/home/yhge/pre-train/Jellyfish-7B" \
#     --lora_weights "jellyfish-rag/outputs/7B/merged/jellyfish-few-shot-train20-merge/" \
#     --model_type "mistral" \
#     --enable_vllm \
#     --overwrite

# export WANDB_NAME=mistral-7B-few-shot
# llamafactory-cli train jellyfish-rag/scripts/upstream-train-7B/few-shot-merge-weight.yaml

# python jellyfish-rag/run_infer.py \
#     --file_path "abt_buy,ae,beers_DC,beers,cms,flights,flipkart,hospital_DC,oa,phone,rayyan_DC,rayyan,walmart_amazon" \
#     --save_path "jellyfish-rag/results/7B/jellyfish-7B-few-shot-train20-merge-weight" \
#     --base_data_dir "data/Jellyfish-rag/test" \
#     --model_weights "/home/yhge/pre-train/Jellyfish-7B" \
#     --lora_weights "jellyfish-rag/outputs/7B/few-shot-train20-weight" \
#     --model_type "mistral" \
#     --overwrite




