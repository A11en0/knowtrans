#! /bin/bash

#SBATCH -J AKB--train
#SBATCH -p a40
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --output=./experiments/results/DP/AKB-train.out

source activate dpmoe

export OPENAI_API_KEY='sk-...'

python src/run_DP.py \
    --task=abt_buy \
    --mode='pipeline' \
    --train_dataset='data/train20/abt_buy.json' \
    --train_version='' \
    --test_dataset='data/test/abt_buy.json' \
    --test_version='' \
    --infer_mode='direct' \
    --component='rules' \
    --save_suffix='-train20(Knowtrans-7B)' \
    --model ./pre-train/Jellyfish-7B \
    --lora ./experiments/outputs/7B/Knowtrans-7B \
    --save_path Knowtrans-7B-pipeline \