import os
import yaml


def generate_yaml_config(datanames):    
    # load template
    with open("jellyfish-rag/scripts/template.yaml", "r") as f:
        config = yaml.safe_load(f)

    for dataset in datanames:
        dataset = dataset.lower()
        config["dataset"] = f"upstream-task-{dataset}"
        config["num_train_epochs"] = 3.0
        config["model_name_or_path"] = f"/home/yhge/pre-train/Llama-3-8B-Instruct"
        config["output_dir"] = f"jellyfish-rag/outputs/8B/upstream-task/upstream-task-{dataset}"
        config["template"] = f"llama3"
        
        yaml_file_path = f"jellyfish-rag/scripts/upstream-train-8B/{dataset}.yaml"
        
        # check file_path if exist
        if not os.path.exists(yaml_file_path):
            os.makedirs(os.path.dirname(yaml_file_path), exist_ok=True)

        with open(yaml_file_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
    
    print(f"YAML配置文件已生成。")


datanames = ['adult', 'amazon_google', 'beers', 'buy', 'dblp_acm', 'dblp_scholar', 'fodors_zagat', 'hospital', 'itunes_amazon', 'restaurant', 'sm']
generate_yaml_config(datanames)    