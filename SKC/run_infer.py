import os
import time
import torch
from tqdm import tqdm, trange
from openai import OpenAI

import random
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from transformers import set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

import json
import argparse
from peft import PeftModel
from utils import eval_results, load_dataset
from model_utils import load_model_and_tokenizer, batch_generate

rand_seed = 42
set_seed(rand_seed)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


print("GPU_NUM", os.environ["CUDA_VISIBLE_DEVICES"])
tp_size = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))

def process_requests(data_list, engine, lora_weights, tokenizer, enable_vllm, subsize, generation_config, batch_size=None, is_moe=False, args=None): 
    datanames, prompts, preds, labels = [], [], [], []
    
    for i in trange(0, len(data_list), batch_size, desc="Predicting batches", position=1, leave=False):
        if i > subsize and subsize != -1:
            break
        
        batch_datas = data_list[i : i + batch_size]
        datanames.extend([data['dataset'] if data.__contains__('dataset') else None for data in batch_datas])
        prompts.extend([data['instruction'] for data in batch_datas])
        preds.extend(batch_generate(batch_datas, generation_config, engine, lora_weights, tokenizer, enable_vllm, is_moe, args))
        labels.extend([data['output'] for data in batch_datas])
        
    return datanames, prompts, preds, labels

def main(args):
    generation_config = {
        "n": 1, 
        "temperature": 0.35,   # 0.3
        "max_tokens": 4096,  # 2048
        "top_p": 0.9, 
        "top_k": 10, 
        "num_beams": 1,    # don't work for poly
        "max_new_tokens": 50, 
        "do_sample": True,  # try do sample
        "use_beam_search": False,
        "best_of": 1  # Add best_of parameter
        # "repetition_penalty": 1.3   # harmful to performance !!!
    }
    
    print("random seed: ", args.seed)
    print("subsize: ", args.subsize)
    set_seed(args.seed)
    
    if args.n > 1:   # consistency
        generation_config['n'] = args.n
        generation_config['do_sample'] = False
        generation_config['use_beam_search'] = False
    
    if args.beam_search:
        generation_config['use_beam_search'] = True        
        generation_config['do_sample'] = False
        generation_config['temperature'] = 0
        generation_config['num_beams'] = 4
        generation_config['n'] = 4

    if args.greedy_search:
        generation_config['do_sample'] = False
        generation_config['temperature'] = 0
    
    args.generation_config = generation_config
    
    # make directory
    print("save path: ", args.save_path)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    # save logs
    with open(os.path.join(args.save_path, 'logs.txt'), 'w') as f:
        json.dump(vars(args), f)    
    
    # load model
    engine, tokenizer = load_model_and_tokenizer(args.model_weights, args.lora_weights, args.enable_vllm, tp_size, args.print_param_status)          
    
    # load datasets
    data_lists = load_dataset(args.model_type, tokenizer, args.file_path, shuffle=args.shuffle, insert_knowledge=args.insert_knowledge, reasoning=args.reasoning, few_shot=args.few_shot, BASE_DATA_DIR=args.base_data_dir)
    
    for file_name, data_list in data_lists.items():
        file_name_for_save = file_name.split(".")[0]
        
        # 添加标识符来区分reasoning和few-shot
        if args.save_type:
            file_name_for_save += f"/{args.save_type}"

        save_list_path = os.path.join(args.save_path, f"{file_name_for_save}")
        
        if not os.path.exists(save_list_path): 
            os.makedirs(save_list_path) 
        
        result_path = os.path.join(save_list_path, "results.json")
        
        print(f"Datasets: {file_name}")
        if os.path.exists(result_path) and not args.overwrite: 
            with open(result_path, 'r') as f: 
                save_list = json.load(f)
        else: 
            datanames, prompts, preds, labels = process_requests(data_list, engine, args.lora_weights, tokenizer, args.enable_vllm, args.subsize, generation_config, args.batch_size, args.is_moe, args)

            save_list = []
            for _, (dataname, prompt, pred, label) in enumerate(zip(datanames, prompts, preds, labels)):
                sample = {
                    "dataset": dataname, 
                    "instruction": prompt,
                    "prediction": pred, 
                    "label": label
                }
                save_list.append(sample)
            
            with open(os.path.join(result_path), "w") as f:
                json.dump(save_list, f, ensure_ascii=False)

        eval_results(save_list, file_name, data_lists, save_path=save_list_path, print_results=True, lower=True)
        
                        
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="An example program with command-line arguments.")
    parser.add_argument('--file_path', default="few-shot/abt_buy,", help='The file path of test datasets.')
    parser.add_argument("--save_path", default="project/results/", help="The name of the person.")
    parser.add_argument("--folder", default="jellyfish", help="The name of the person.")
    parser.add_argument("--split", default="test", help="The split of the dataset.")
    parser.add_argument("--model_weights", default="/home/yhge/pre-train/Mistral-7B-Instruct-v0.2", help="The name of the person.")
    parser.add_argument("--lora_weights", nargs='+', type=str, default=["project/outputs/lora_EM_all", "project/outputs/lora_DI_all"], help="List of LoRA weights.")
    parser.add_argument("--overwrite", action="store_true", help="if overwrite the saved results.")
    parser.add_argument("--enable_vllm", action="store_true", help="if overwrite the saved results.")
    parser.add_argument("--seed", default="42", type=int, help="The number of seed.")
    parser.add_argument("--subsize", default="-1", type=int, help="The number of sub-samples.")
    parser.add_argument("--is_moe", action="store_true", help="Whether use the moe architecture.")
    parser.add_argument("--greedy_search", action="store_true", help="Whether use the greedy search.")
    parser.add_argument("--beam_search", action="store_true", help="Whether use the beam search.")
    parser.add_argument("--n", default="1", type=int, help="The number of outputs.")
    parser.add_argument("--batch_size", default="8", type=int, help="The number of samples in a batch.")
    parser.add_argument("--print_param_status", action="store_true", help="Whether print the parameters.")
    parser.add_argument("--model_type", default="mistral", help="The model type.")
    parser.add_argument("--base_data_dir", default="mistral", help="The model type.")
    parser.add_argument("--shuffle", action="store_true", help="Whether to shuffle the dataset.")
    parser.add_argument("--insert_knowledge", action="store_true", help="Whether to insert additional knowledge into the instructions.")
    parser.add_argument("--reasoning", action="store_true", help="Whether to add reasoning to the instructions.") 
    parser.add_argument("--few_shot", action="store_true", help="Whether to apply few-shot learning.")
    parser.add_argument("--save_type", default="", type=str, help=".")
    
    args = parser.parse_args()
    main(args)
    


