import random
import fire
from automatic_prompt_engineer import ape, data, template
from experiments.data.load_data import load_dataset
# from experiments.evaluation.instruction_induction.exec_accuracy import exec_accuracy_evaluator
import yaml
import re
import argparse
import json
from typing import Literal
import math
import os
import copy



def run(task, component, train_version, train_dataset, test_version, test_dataset, save_suffix, model, lora, infer_mode, ICL_method, demo_dataset, metric, save_path):
    train_data, template= load_dataset(task, train_version, train_dataset, infer_mode, ICL_method, demo_dataset, component)
    if len(train_data) <= 20:
        prompt_gen_data = train_data.copy()
        eval_data = train_data.copy()
    else:
        prompt_gen_data, eval_data = data.create_split(
            train_data, max(20, int(len(train_data)*0.2)))
    

    
    with open(f'./experiments/configs/default.yaml') as f:
        conf = yaml.safe_load(f)
        conf['evaluation']['num_samples'] = min(len(eval_data),conf['evaluation']['num_samples'])
        conf['generation']['num_demos'] = min(len(train_data), conf['generation']['num_demos'])
        if model:
            conf['evaluation']['model']['gpt_config']['model'] = model
        if metric:
            conf['evaluation']['method'] = metric
        if lora:
            conf['evaluation']['model']['gpt_config']['lora'] = lora

       
    res, _ = ape.find_prompts(task_type=template.task_type,
                              eval_template=template,
                                prompt_gen_data=prompt_gen_data,
                                eval_data=eval_data,
                                conf=conf,
                                few_shot_data=prompt_gen_data,
                                demos_template=template._demo_template,
                                prompt_gen_template=template.prompt_gen_template(component),
                                seed_prompt = template.get_component(component),
                                component=component,
                                )

    print('Finished finding prompts.')
    prompts, scores = res.sorted()
    print('Prompts:')
    for prompt, score in list(zip(prompts, scores)):#[:10]
        print(f'  {score}: {prompt}')

    print('--'*50)
    print(f'Evaluating on test data on {prompts[0]}')
    template.update(component, prompts[0])
    infer(task, component, template, test_dataset, test_version, save_suffix, conf, infer_mode, ICL_method, None, save_path or 'train', res)
    return 



def error(task, component, train_dataset, train_version, test_dataset, test_version, save_suffix, model, lora, infer_mode, ICL_method, save_path):
    train_data, template = load_dataset(task, train_version, train_dataset, infer_mode, ICL_method, None, component)
    
    with open(f'./experiments/configs/default.yaml') as f:
        conf = yaml.safe_load(f)
        conf['evaluation']['num_samples'] = min(len(train_data),conf['evaluation']['num_samples'])
        if model:
            conf['evaluation']['model']['gpt_config']['model'] = model
        if lora:
            conf['evaluation']['model']['gpt_config']['lora'] = lora

    train_res = infer(task, component, template, train_dataset, train_version, save_suffix, conf, infer_mode, ICL_method, save_path or 'train')
    print(f"[INFO] train errors is computed: {train_res.sorted()[1][0]} about {len(train_res.errors)} instance")


    for i in range(2):
        if len(train_res.errors) == 0:
            # raise ValueError(f"Train set has no error!")
            print(f"[Warning] the train set has no error now!")
            break
        eval_res = ape.refine_prompts(task=task,
                                    task_type=template.task_type,
                                    component=component,
                                    eval_template=template,
                                    demos_template=template._demo_template,
                                        error_data=train_res.errors,
                                        eval_data=train_data,
                                        origin_prompt=template,
                                        conf=conf,
                                        )

        print('Finished finding prompts.')
        prompts, scores = eval_res.sorted()
        print('Prompts:')
        for prompt, score in list(zip(prompts, scores)):#[:10]
            print(f'  {score}: {prompt}')
        
        train_res = eval_res
        
        # template.update(component, prompts[-1])
        # infer(task, component, template, test_dataset, test_version, save_suffix, conf, infer_mode, ICL_method, None, save_path+f"-{i+1}l", train_res)

        template.update(component, prompts[0])  # select the bset prompt
        # conf['refine']['num_subsamples'] = int(conf['refine']['num_subsamples'] / 2)
        # infer(task, component, template, test_dataset, test_version, save_suffix, conf, infer_mode, ICL_method, None, save_path+f"-{i+1}h", train_res)

        

    print('--'*50)
    print(f'Evaluating on test data on {prompts[0]}')

    # test_data, template = load_dataset(task, test_version, test_dataset, infer_mode, ICL_method)
    infer(task, component, template, test_dataset, test_version, save_suffix, conf, infer_mode, ICL_method, None, save_path or 'train', train_res)
    return 


def test(task, component, test_version, test_dataset, save_suffix, model, lora, infer_mode, ICL_method, demo_dataset, save_path='test'):
    test_data, template = load_dataset(task, test_version, test_dataset, infer_mode, ICL_method, demo_dataset)

    with open(f'./experiments/configs/default.yaml') as f:
        conf = yaml.safe_load(f)
        if model:
            conf['evaluation']['model']['gpt_config']['model'] = model
        if lora:
            conf['evaluation']['model']['gpt_config']['lora'] = lora
        conf['evaluation']['num_samples'] = len(test_data) # test full dataset
    
    infer(task, component, template, test_dataset, test_version, save_suffix, conf, infer_mode, ICL_method, demo_dataset, save_path)


def infer(task, component, template: template.Template, test_dataset, test_version, save_suffix, conf, infer_mode, ICL_method, demo_dataset=None, save_path=None, eval_res=None):
    test_data, _ = load_dataset(task, test_version, test_dataset, infer_mode, ICL_method, demo_dataset, component)
    infer_conf = copy.deepcopy(conf)
    infer_conf['evaluation']['num_samples'] = len(test_data)
    # conf['evaluation']['num_samples'] = 100
    test_res = ape.evaluate_prompts(task_type=template.task_type,
                                prompts=[template.get_component(component)],
                                eval_template=template,
                                eval_data=test_data,
                                few_shot_data=test_data, 
                                demos_template=template._demo_template,
                                conf=infer_conf,
                                )
    test_score = test_res.sorted()[1][0]
    print(f'Test score: {test_score}')

    if save_path is None:
        return test_res
    # {re.search(r"\/([^\/]+)\.json$", test_dataset)}
    if not os.path.exists(f'./experiments/results/DP/{save_path}'):
        os.makedirs(f'./experiments/results/DP/{save_path}')
    with open(f'./experiments/results/DP/{save_path}/{task}{save_suffix}-error.json', "w", encoding='utf-8') as json_file:
        json.dump(test_res.errors, json_file, indent=4, ensure_ascii=False)
    with open(f'experiments/results/DP/{save_path}/{task}{save_suffix}.txt', 'w', encoding='utf-8') as f:
        f.write(f"Test model: {infer_conf['evaluation']['model']['gpt_config']['model']} {infer_conf['evaluation']['model']['gpt_config'].get('lora', None)}\n")
        f.write(f'Test dataset: {test_dataset}\n')
        f.write(f'Test score: {test_score}\n')
        f.write(f"Prompt: {template.eval_template()}\n\n\n")
        if eval_res:
            prompts, scores = eval_res.sorted()
            for prompt, score in zip(prompts, scores):
                f.write(f"  {score}: {prompt}\n")


def export(task, version, dataset, save_suffix, infer_mode, ICL_method, demo_dataset, export_as_demo):
    dataset, template = load_dataset(task, version, dataset, infer_mode, ICL_method, demo_dataset)
    data = template.export(template, dataset, export_as_demo)

    json_data = data
    
    with open(f"./data/{task}/{save_suffix}.json", "w", encoding='utf-8') as json_file:
        json.dump(json_data, json_file, indent=4)

    print(f'Finished exporting dataset {task}{version}.')




def main():
    parser = argparse.ArgumentParser(description="A simple CLI calculator")
    
    parser.add_argument("--task", help="The operation to perform")
    parser.add_argument("--train_version", type=str, help="version for _template")
    parser.add_argument("--test_version", type=str, help="version for _template")
    parser.add_argument("--train_pattern", type=str, default=None, help="version for _template")
    parser.add_argument("--test_pattern", type=str, default=None, help="version for _template")
    parser.add_argument("--mode", choices=['train','test', 'export', 'error'], help="version for _template")
    parser.add_argument("--train_dataset", type=str, help="The second operand")
    parser.add_argument("--test_dataset", type=str, help="The second operand")
    parser.add_argument("--component", type=str, default=None, help="The second operand")
    parser.add_argument("--save_suffix", type=str, help="The second operand")
    parser.add_argument("--model", type=str, default=None, help="eval/test model")
    parser.add_argument("--lora", type=str, default=None, help="eval/test model")
    parser.add_argument("--infer_mode", type=str, choices=['direct','reason'], default='direct', help="eval/test model")
    parser.add_argument("--ICL_method", type=str, default="{}", help="eval/test model")
    parser.add_argument("--demo_dataset", type=str, default=None, help="The second operand")
    parser.add_argument("--metric", type=str, choices=['f1_score', 'likelihood'], default='f1_score', help="metric for evaluation")
    parser.add_argument("--save_path", type=str, default=None, help="save_path")
    parser.add_argument("--export_as_demo", type=bool, default=False, help="export only as demo dataset, cannot be used as test!")
    
    args = parser.parse_args()

    args.ICL_method = json.loads(args.ICL_method)

    # if not args.test_dataset:
    #     args.test_data
    
    if args.mode == "train":
        run(args.task, args.component, args.train_version, args.train_dataset, args.test_version, args.test_dataset, args.save_suffix, args.model, args.lora, args.infer_mode, args.ICL_method, args.demo_dataset, args.metric, args.save_path)
    elif args.mode == "test":
        test(args.task, args.component, args.test_version, args.test_dataset, args.save_suffix, args.model, args.lora, args.infer_mode, args.ICL_method, args.demo_dataset, args.save_path)
    elif args.mode == "export":
        export(args.task, args.test_version, args.test_dataset, args.save_suffix, args.infer_mode, args.ICL_method, args.demo_dataset, args.export_as_demo)
    elif args.mode == "error":
        error(args.task, args.component, args.train_dataset, args.train_version, args.test_dataset, args.test_version, args.save_suffix, args.model, args.lora, args.infer_mode, args.ICL_method, args.save_path)
    else:
        print("Error: Invalid operation")
        
    
if __name__ == '__main__':
    # fire.Fire(run)
    main()
