import os
import json
from collections import defaultdict


def ft_data_prepare(save_dir, data_path, k):
    # save the ft_datas to local json file
    with open(data_path, "r", encoding="utf-8") as f:
        # read json file
        results = json.load(f)

    dataset_groups = defaultdict(list)
    for result in results:
        # add system prompt
        sys_prompt = "You are an AI assistant that follows instruction extremely well. User will give you a question. Your task is to answer as faithfully as you can."

        ft_data = {
            "system": sys_prompt,
            "instruction": result["instruction"],
            "input": result["input"],
            "output": result["output"]
        }

        # 根据 dataset 字段分组
        dataset_groups[result.get("dataset", "default")].append(ft_data)

    # 将每个数据集保存到单独的文件中
    for dataset, data in dataset_groups.items():
        file_path = f"{save_dir}/train/k{k}/{dataset}.json"
        
        # check file_path if exist
        if not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"数据已保存到 {save_dir} 目录下的多个文件中。")

    generate_yaml_config(dataset_groups)


def generate_yaml_config(dataset_groups):
    import yaml
    # load template
    with open("jellyfish-rag/scripts/template.yaml", "r") as f:
        config = yaml.safe_load(f)

    for dataset in dataset_groups.keys():
        dataset = dataset.lower()
        # config["dataset"] = f"jellyfish-rag-k{k}-{dataset}-7B"
        config["dataset"] = f"{dataset}-k{k}-
        
        jellyfish-rag-k{k}-{dataset}-7B"
        config["output_dir"] = f"jellyfish-rag/outputs/{dataset}-lora-k{k}-7B"

        yaml_file_path = f"jellyfish-rag/scripts/k{k}/{dataset}.yaml"
        with open(yaml_file_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

    print("YAML配置文件已生成。")


if __name__=="__main__":
    k = 8
    save_dir = "data/Jellyfish-rag"
    data_path = f"{save_dir}/retrieval_results_k{k}.json"    
    ft_data_prepare(save_dir, data_path, k=8)