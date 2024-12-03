# KNOWTRANS: Boosting Transferability of Data Preparation LLMs via Knowledge Augmentation

Yuhang Ge, Fengyu Li, Yuren Mao, Yanbo Yang, Congcong Ge, Zhaoqiang Chen, Jiang Long, Yunjun Gao*

This repo contains code for "KNOWTRANS: Boosting Transferability of Data Preparation LLMs via Knowledge Augmentation". Please see our paper for technique details.

# Abstract

Data Preparation (DP), which involves tasks such as data cleaning, imputation and integration, is a fundamental process in data-driven applications. Recently, Large Language Models (LLMs) fine-tuned for DP tasks, i.e., DP-LLMs, have achieved state-of-the-art performance. However, transferring DP-LLMs to novel datasets and tasks typically requires a substantial amount of labeled data, which is impractical in many real-world scenarios. To address this issue, we propose a knowledge augmentation framework for data preparation, dubbed KNOWTRANS. This framework allows DP-LLMs to be transferred to novel datasets and tasks with a few data, significantly decreasing the dependence on extensive labeled data. KNOWTRANS comprises two components: Selective Knowledge Concentration and Automatic Knowledge Bridging. The first component re-uses knowledge from previously learned tasks, while the second automatically integrates additional knowledge from external sources. Extensive experiments on 13 datasets demonstrate the effectiveness of KNOWTRANS. KNOWTRANS boosts the performance of the stateof-the-art DP-LLM, Jellyfish-7B, by an average of 4.93%, enabling it to outperform both GPT-4 and GPT-4o.

## Installation

KNOWTRANS consists of two components: SKC for training stage and AKB for inference stage.

First, you need first to install the LLM training framework [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

Then, you need to install our modified peft module in `SKC/peft` with:

```
pip install -e SKC/peft
```

Last, in order to run AKB component, you need setup your OPENAI_API_KEY with the following command:

```
export OPENAI_API_KEY=YOUR_KEY
```
<!-- 
## Dataset Preparation
1. Transfer instance to string.
   
   `python AKB/data_utils/prepare.py`

2. Export instruction datasets.
    ```python
    python experiments/run_DP.py \
        --task={task} \
        --mode='export' \
        --test_dataset={dataset} \
        --test_version='' \
        --save_suffix='test'
    ```

3. (optional) If In-context Learning is necessary, add "--export_as_demo true" parameter to generate damo datasets based on train.json and use "ICL_method".  -->

## Running

### SKC (Training Stage)

1. Download datasets from [Jellyfish Benchmark](https://huggingface.co/datasets/NECOUDBFM/Jellyfish-Instruct).
   
2. Run upstream training scripts in `scripts/upstream-train-7B/train.sh` with LLaMA-Factory.

3. Run downstream training script in `scripts/train-few-shot.sh` with LLaMA-Factory.

### AKB (Inference Stage)

4. Run AKB
   
 - Generation:
    ```python
    python experiments/run_DP.py \
        --task={dataset_name} \
        --mode='train' \
        --train_dataset='{train_dataset}' \
        --train_version='' \
        --test_dataset='{test_dataset}' \
        --test_version='' \
        --infer_mode='direct' \
        --component='rules' \
        --save_suffix='-train' \
        --model {upstream_model_path} \
        --lora {lora_path} \
        --save_path {save_path}
    ```

- Refinemnt:
    ```python
    python experiments/run_DP.py \
        --task={dataset_name} \
        --mode='error' \
        --train_dataset='{train_dataset}' \
        --train_version='' \
        --test_dataset='{test_dataset}' \
        --test_version='1' \
        --infer_mode='direct' \
        --save_suffix='-train_1' \
        --model {upstream_model_path} \
        --lora {lora_path} \
        --save_path {save_path}
    ```

## Comments

Our codebase is based on the following repo. Thanks for open-sourcing!

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [APE](https://github.com/keirp/automatic_prompt_engineer)
- [PEFT](https://github.com/huggingface/peft)
