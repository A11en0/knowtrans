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


# # Set wandb and run scripts
# export WANDB_NAME=llama3-8B-adult
# llamafactory-cli train jellyfish-rag/scripts/upstream-train-8B/adult.yaml

# # Set wandb and run scripts
# export WANDB_NAME=llama3-8B-amazon_google
# llamafactory-cli train jellyfish-rag/scripts/upstream-train-8B/amazon_google.yaml

# # Set wandb and run scripts
# export WANDB_NAME=llama3-8B-sm
# llamafactory-cli train jellyfish-rag/scripts/upstream-train-8B/sm.yaml

# # Set wandb and run scripts
# export WANDB_NAME=llama3-8B-few-shot
# llamafactory-cli train jellyfish-rag/scripts/upstream-train-8B/few-shot.yaml

# python jellyfish-rag/run_infer.py \
#     --file_path "abt_buy,ae,beers_DC,beers,cms,flights,flipkart,hospital_DC,oa,phone,rayyan_DC,rayyan,walmart_amazon" \
#     --save_path "jellyfish-rag/results/8B/few-shot-train20" \
#     --base_data_dir "data/Jellyfish-rag/mini_test" \
#     --model_weights "/home/yhge/pre-train/Llama-3-8B-Instruct" \
#     --lora_weights "jellyfish-rag/outputs/8B/few-shot-train20" \
#     --model_type "llama3" \
#     --enable_vllm \
#     --overwrite

# export WANDB_NAME=llama3-8B-few-shot
# llamafactory-cli train jellyfish-rag/scripts/upstream-train-8B/llama3-few-shot-merge.yaml

# python jellyfish-rag/run_infer.py \
#     --file_path "abt_buy,ae,beers_DC,beers,cms,flights,flipkart,hospital_DC,oa,phone,rayyan_DC,rayyan,walmart_amazon" \
#     --save_path "jellyfish-rag/results/8B/few-shot-train20-merge" \
#     --base_data_dir "data/Jellyfish-rag/mini_test" \
#     --model_weights "jellyfish-rag/outputs/8B/upstream-task/full_model" \
#     --lora_weights "jellyfish-rag/outputs/8B/few-shot-train20-merge" \
#     --model_type "llama3" \
#     --enable_vllm \
#     --overwrite 

# --base_data_dir "data/Jellyfish-rag/mini_test" \
# python jellyfish-rag/run_infer.py \
#     --file_path "abt_buy,ae,beers_DC,beers,cms,flights,flipkart,hospital_DC,oa,phone,rayyan_DC,rayyan,walmart_amazon" \
#     --save_path "jellyfish-rag/results/8B/few-shot-train20-merge-jellyfish-fulltest" \
#     --base_data_dir "data/Jellyfish-rag/test" \
#     --model_weights "/home/yhge/pre-train/Jellyfish-8B" \
#     --lora_weights "jellyfish-rag/outputs/8B/few-shot-train20-merge" \
#     --model_type "llama3" \
#     --enable_vllm \
#     --overwrite 

# export WANDB_NAME=jellyfish-8B-few-shot
# llamafactory-cli train jellyfish-rag/scripts/upstream-train-8B/jellyfish-few-shot-merge.yaml

# # 合并后 few-shot FT on jellyfish
# python jellyfish-rag/run_infer.py \
#     --file_path "abt_buy,ae,beers_DC,beers,cms,flights,flipkart,hospital_DC,oa,phone,rayyan_DC,rayyan,walmart_amazon" \
#     --save_path "jellyfish-rag/results/8B/jellyfish-few-shot-train20-merge" \
#     --base_data_dir "data/Jellyfish-rag/test" \
#     --model_weights "/home/yhge/pre-train/Jellyfish-8B" \
#     --lora_weights "jellyfish-rag/outputs/8B/jellyfish-few-shot-train20-merge" \
#     --model_type "llama3" \
#     --enable_vllm \
#     --overwrite
                                
# # 合并后 few-shot FT on jellyfish
# python jellyfish-rag/run_infer.py \
#     --file_path "abt_buy,ae,beers_DC,beers,cms,flights,flipkart,hospital_DC,oa,phone,rayyan_DC,rayyan,walmart_amazon" \
#     --save_path "jellyfish-rag/results/8B/jellyfish-8B" \
#     --base_data_dir "data/Jellyfish-rag/test" \
#     --model_weights "/home/yhge/pre-train/Jellyfish-8B" \
#     --lora_weights "" \
#     --model_type "llama3" \
#     --enable_vllm \
#     --overwrite                     

# export WANDB_NAME=jellyfish-8B-few-shot-weight
# llamafactory-cli train jellyfish-rag/scripts/upstream-train-8B/jellyfish-few-shot-merge.yaml

# few-shot FT on jellyfish
python jellyfish-rag/run_infer.py \
    --file_path "abt_buy,ae,beers_DC,beers,cms,flights,flipkart,oa,phone,rayyan_DC,rayyan,walmart_amazon" \
    --save_path "jellyfish-rag/results/8B/few-shot-train20-weight" \
    --base_data_dir "data/Jellyfish-rag/test" \
    --model_weights "/home/yhge/pre-train/Jellyfish-8B" \
    --lora_weights "jellyfish-rag/outputs/8B/few-shot-train20-weight" \
    --model_type "llama3" \
    --overwrite 
    