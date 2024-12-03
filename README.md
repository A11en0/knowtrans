# KNOWTRANS: Boosting Transferability of Data Preparation LLMs via Knowledge Augmentation

Yuhang Ge, Fengyu Li, Yuren Mao, Yanbo Yang, Congcong Ge, Zhaoqiang Chen, Jiang Long, Yunjun Gao*

This repo contains code for "KNOWTRANS: Boosting Transferability of Data Preparation LLMs via Knowledge Augmentation." Please see our paper `KnowTrans-Full-Version.pdf` for technique details.

# Abstract

Data Preparation (DP), which involves tasks such as data cleaning, imputation and integration, is a fundamental process in data-driven applications. Recently, Large Language Models (LLMs) fine-tuned for DP tasks, i.e., DP-LLMs, have achieved state-of-the-art performance. However, transferring DP-LLMs to novel datasets and tasks typically requires a substantial amount of labeled data, which is impractical in many real-world scenarios. To address this issue, we propose a knowledge augmentation framework for data preparation, dubbed KNOWTRANS. This framework allows DP-LLMs to be transferred to novel datasets and tasks with a few data, significantly decreasing the dependence on extensive labeled data. KNOWTRANS comprises two components: Selective Knowledge Concentration and Automatic Knowledge Bridging. The first component re-uses knowledge from previously learned tasks, while the second automatically integrates additional knowledge from external sources. Extensive experiments on 13 datasets demonstrate the effectiveness of KNOWTRANS. KNOWTRANS boosts the performance of the state-of-the-art DP-LLM, Jellyfish-7B, by an average of 4.93%, enabling it to outperform both GPT-4 and GPT-4o.

## Installation

1. Install the LLM training framework [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

2. Install our modified peft module in `SKC/peft` with:

    ```python
    pip install -e peft
    ```

3. Setup your OPENAI_API_KEY with the following command:

    ```bash
    export OPENAI_API_KEY=YOUR_KEY
    ```

<!-- 
## Dataset Preparation
1. Transfer instance to string.


2. Export instruction datasets.
    ```python
    python experiments/run_DP.py \
        --task={task} \
        --mode='export' \
        --test_dataset={dataset} \
        --test_version='' \
        --save_suffix='test'
    ```
-->

## Running

KNOWTRANS consists of two components: SKC for the training stage and AKB for the inference stage.

### SKC (Training Stage)

1. Download datasets from [Jellyfish Benchmark](https://huggingface.co/datasets/NECOUDBFM/Jellyfish-Instruct). If prepare your own datasets, we also provide a dataset preprocessing script `python src/data_utils/prepare.py`. 
   
2. Run upstream training scripts in `scripts/upstream-train-7B/train.sh` with LLaMA-Factory.

3. Run downstream training script in `scripts/train-few-shot.sh` with LLaMA-Factory.

### AKB (Inference Stage)

4. Run AKB
   
 - Generation:
    ```python
    python src/run_DP.py \
        --task={dataset_name} \
        --mode='train' \
        --train_dataset={train_dataset} \
        --train_version='' \
        --test_dataset={test_dataset} \
        --test_version='' \
        --infer_mode='direct' \
        --component='rules' \
        --save_suffix='-train' \
        --model={upstream_model_path} \
        --lora={lora_path} \
        --save_path={save_path}
    ```

- Refinement:
    ```python
    python src/run_DP.py \
        --task={dataset_name} \
        --mode='error' \
        --train_dataset={train_dataset} \
        --train_version='' \
        --test_dataset={test_dataset} \
        --test_version='1' \
        --infer_mode='direct' \
        --save_suffix='-train_1' \
        --model={upstream_model_path} \
        --lora={lora_path} \
        --save_path={save_path}
    ```

5. Export instruction datasets with knowledge.
    ```python
    python experiments/run_DP.py \
        --task={task} \
        --mode='export' \
        --test_dataset={dataset} \
        --test_version='' \
        --save_suffix='test'
    ```

    (optional) If In-context Learning is necessary, add "--export_as_demo true" parameter to generate damo datasets based   on train.json and use "ICL_method". 

6. Inference on the test dataset
   ```python
   python src/run_infer.py \
    --file_path="abt_buy,ae,beers_DC,beers,cms,flights,flipkart,oa,phone,rayyan_DC,rayyan,walmart_amazon,sotab3" \
    --save_path={save_path} \
    --base_data_dir="data/test" \
    --model_weights={backbone_model_path}\
    --lora_weights={lora_path} \
    --model_type="mistral" \
    --enable_vllm \
    --overwrite
   ```

## Comments

Our codebase is based on the following repo. Thanks for open-sourcing!

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [APE](https://github.com/keirp/automatic_prompt_engineer)
- [PEFT](https://github.com/huggingface/peft)
