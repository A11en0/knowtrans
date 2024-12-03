import os
import json
import random
import re
from automatic_prompt_engineer.template import Template,_template
# induce_data_path = os.path.dirname(__file__)os.path.join(, 'raw/induce/')
# eval_data_path = os.path.join(os.path.dirname(__file__), 'raw/execute/')

# Get a list of tasks (by looking at the names of the files in the induced directory)
tasks = [f.split('.')[0] for f in os.listdir(os.path.dirname(__file__))]

def load_data(task, version_suffix, path):
    # assert task+'_test' in tasks, f'Task {task} not found!'

    # def load_dataset(task, suffix):
    # path = os.path.join('./experiments/data/EM', task+suffix+'.json')

    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    return data
    
    # inputs = []
    # outputs = []
    # instructions = []
    # for item in data:
    #     # item['input'] = pattern.search(item['instruction']).group(1).strip()
    #     # inputs.append(item['input'])
    #     instructions.append(item['instruction'])
    #     inputs.append(item['entity'])
    #     outputs.append(item['output'])

    # return (inputs, outputs), instructions, data[0]['task_type']
    

def load_dataset(task, version, dataset, infer_mode, ICL_method, demo_dataset=None, component=None):
    version_suffix = f'_{version}' if version is not None and len(version) else version
    data = load_data(task, version_suffix, dataset)

    template = Template(
        task = task,
        task_type = data[0]['task_type'],
        task_description = _template[task].get('task_description' + version_suffix, _template[task].get('task_description')),
        rules = _template[task].get('rules' + version_suffix, _template[task].get('rules')),
        question = _template[task].get('question' + version_suffix, _template[task].get('question')),
        infer_mode = infer_mode,
        ICL_method=ICL_method,
        demo_file=demo_dataset,
        component=component,
        )
    
    return data, template



    
