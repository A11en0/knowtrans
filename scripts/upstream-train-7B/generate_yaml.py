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
        config["model_name_or_path"] = f"/home/yhge/pre-train/Mistral-7B-OpenOrca"
        config["output_dir"] = f"upstream-task-{dataset}"
        config["template"] = f"mistral"
        
        yaml_file_path = f"jellyfish-rag/scripts/upstream-train-7B/{dataset}.yaml"
        
        # check file_path if exist
        if not os.path.exists(yaml_file_path):
            os.makedirs(os.path.dirname(yaml_file_path), exist_ok=True)

        with open(yaml_file_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
    
    # config["dataset"] = f"k{args.k}-{args.model}-all"
    # config["num_train_epochs"] = args.epoch
    # config["model_name_or_path"] = f"/home/yhge/pre-train/Jellyfish-{args.model}/"
    # config["output_dir"] = f"jellyfish-rag/outputs/lora-emebdding-k{args.k}-{args.model}"
    
    # yaml_file_path = f"jellyfish-rag/scripts/train_{args.model}/k{args.k}.yaml"

    # check file_path if exist
    # if not os.path.exists(yaml_file_path):
    #     os.makedirs(os.path.dirname(yaml_file_path), exist_ok=True)

    # with open(yaml_file_path, "w") as f:
    #     yaml.dump(config, f, default_flow_style=False)
    
    print(f"YAML配置文件已生成。")


datanames = ['adult', 'amazon_google', 'beers', 'buy', 'dblp_acm', 'dblp_scholar', 'fodors_zagat', 'hospital', 'itunes_amazon', 'restaurant', 'sm']
generate_yaml_config(datanames)    