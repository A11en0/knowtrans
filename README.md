# KNOWTRANS: Boosting Transferability of Data Preparation LLMs via Knowledge Augmentation

Yuhang Ge, Fengyu Li, Yuren Mao, Yanbo Yang, Congcong Ge, Zhaoqiang Chen, Jiang Long, Yunjun Gao*

This repo contains code for "KNOWTRANS: Boosting Transferability of Data Preparation LLMs via Knowledge Augmentation." Please see our paper `KnowTrans-Full-Version.pdf` for technique details.

# Abstract

Data Preparation (DP), which involves tasks such as data cleaning, imputation and integration, is a fundamental process in data-driven applications. Recently, Large Language Models (LLMs) fine-tuned for DP tasks, i.e., DP-LLMs, have achieved state-of-the-art performance. However, transferring DP-LLMs to novel datasets and tasks typically requires a substantial amount of labeled data, which is impractical in many real-world scenarios. To address this issue, we propose a knowledge augmentation framework for data preparation, dubbed KNOWTRANS. This framework allows DP-LLMs to be transferred to novel datasets and tasks with a few data, significantly decreasing the dependence on extensive labeled data. KNOWTRANS comprises two components: Selective Knowledge Concentration and Automatic Knowledge Bridging. The first component re-uses knowledge from previously learned tasks, while the second automatically integrates additional knowledge from external sources. Extensive experiments on 13 datasets demonstrate the effectiveness of KNOWTRANS. KNOWTRANS boosts the performance of the state-of-the-art DP-LLM, Jellyfish-7B, by an average of 4.93%, enabling it to outperform both GPT-4 and GPT-4o.

## Installation

1. Install the LLM training framework [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

2. Install our modified peft module with:

    ```python
    pip install -e peft
    ```

3. Setup your OPENAI_API_KEY with the following command:

    ```bash
    export OPENAI_API_KEY=YOUR_KEY
    ```

## Running

KNOWTRANS consists of two components: Selective Knowledge Concentration (SKC) for the training stage and Automatic Knowledge Bridging (AKB) for the inference stage.

### SKC (Training Stage)

1. Download datasets from [Jellyfish Benchmark](https://huggingface.co/datasets/NECOUDBFM/Jellyfish-Instruct). If prepare your own datasets, we also provide a dataset preprocessing script `python src/data_utils/prepare.py`. Put all the dataset file in './data' and register them in dataset_info.json following LLaMA-Factory.

2. Add extra finetuning_args in LLaMA-Factory:
    ```python
    use_wlora: bool = field(
        default=False,
        metadata={"help": "Whether use_wlora."},
    ) 
    lora_ckpt_dir: str = field(
        default="./",
        metadata={"help": "Path to the adapters of the saved LoRA model."},        
    )
    ```

2. Run upstream training scripts in `scripts/upstream-train-7B/train.sh` with LLaMA-Factory.

3. Run downstream training script in `scripts/downstream-train/train.sh` with LLaMA-Factory.

### AKB (Inference Stage)

### 4. Run AKB

The `main` function takes the following arguments:

- `task` - The specific task or dataset to be performed.

- `train_version` - The version identifier for the training dataset rules. For example, 'best' is 'rules_best' in template dict.

- `test_version` - The version identifier for the testing dataset rules.

- `mode` - The operating mode of the program. Options: `'train'`, `'test'`, `'export'`, `'error'`, `'pipeline'`.

- `train_dataset` - Path to the labeled training dataset.

- `test_dataset` - Path to the labeled testing dataset.

- `component` - The component to optimize. Default is `'rules'`.

- `save_suffix` - Suffix for the saved result file name, appended after the task name.

- `model` - Path to the evaluation/testing model.

- `lora` - Path to the evaluation/testing LoRA model.

- `infer_mode` - Inference mode.

- `ICL_method` - The In-Context Learning (ICL) method configuration.

- `demo_dataset` - Path to the demonstration dataset.

- `metric` - Evaluation metric.

- `save_path` - Path to save output files.

- `export_as_demo` - Activate to export only as a demonstration dataset. Note: Cannot be used for testing purposes.

- Pipeline:
    ```python
    python src/run_DP.py \
        --task={dataset_name} \
        --mode='pipeline' \
        --train_dataset={train_dataset} \
        --train_version='' \
        --test_dataset={test_dataset} \
        --test_version='' \
        --infer_mode='direct' \
        --component='rules' \
        --save_suffix='' \
        --model={upstream_model_path} \
        --lora={lora_path} \
        --save_path={save_path}
    ```

Or run the Knowledge Generation and Knowledge Refinement seperately

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
    python src/run_DP.py \
        --task={task} \
        --mode='export' \
        --test_dataset={dataset} \
        --test_version='' \
        --save_suffix='test'
    ```

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

   (optional) If In-context Learning is necessary, add "--export_as_demo true" parameter to generate damo datasets based on train.json in task dataset and use ICL_method='{"sim":"em-3"}' to choose the top-3 similar instance as demo.

## Comments

Our codebase is based on the following repo. Thanks for open-sourcing!

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [APE](https://github.com/keirp/automatic_prompt_engineer)
- [PEFT](https://github.com/huggingface/peft)
