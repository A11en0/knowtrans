#! /bin/bash

#SBATCH -J Knowtrans-7B--infer
#SBATCH -p a40
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --output=./experiments/results/DP/Knowtrans-7B--infer.out

source activate dpmoe

tasks=(
    "flights"
    "rayyan"
    "beers"
    "flipkart"
    "phone"
    "cms"
    "abt_buy"
    "walmart_amazon"
    "sotab3"
    "ae"
    "oa"
    "hospital_DC"
    "rayyan_DC"
    "beers_DC"
)

# 循环遍历每个任务
for task in "${tasks[@]}"; do

    echo "Processing task: $task"

        python src/run_DP.py \
        --task=$task \
        --mode='test' \
        --test_dataset="data/test/${task}.json" \
        --test_version='best' \
        --infer_mode='direct' \
        --save_suffix='-test_BestD(Knowtrans-7B)' \
        --model ./pre-train/jellyfish-7B \
        --lora ./experiments/outputs/7B/Knowtrans-7B \
        --save_path Knowtrans-7B-test 
done
exit