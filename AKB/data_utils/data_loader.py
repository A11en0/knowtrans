import pandas as pd
import os
from typing import Literal, Optional
from fuzzywuzzy import fuzz
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split  
import json
import re
import csv
import random
from datetime import datetime
from dateutil import parser
import numpy as np
import copy
from collections import OrderedDict  
from sklearn.metrics import jaccard_score
from scipy.spatial.distance import euclidean
import heapq
from multiprocessing import Pool, Array
# from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pickle
# from transformers import AutoTokenizer
from functools import partial
# import faiss
import math


task_type_map = {
    'ED': 0,
    'DI': 1,
    'SM': 2,
    'EM': 3,
    "CTA": 4,
    "AVE": 5,
    "DC": 6,
}

def serialize_(entities):
    table = ""
    for entity in entities:
        row_values = []
        for pair in entity:
            attr_name, attr_value = pair
            if pd.isna(attr_value) or attr_value=="" or attr_value=="?":  
                row_values.append(f"{attr_name}: \"nan\"")  # 将空值转换为空字符串
            else:
                row_values.append(f"{attr_name}: \"{attr_value}\"")
        table += "[" + ", ".join(row_values) + "]"
    return table

def get_input(dataset_dir, task_type, columns, query_column=None, split='test'):
    # table_path = os.path.join(dataset_dir, 'table.csv')
    # if os.path.exists(table_path):
    #     raise ValueError(f"Table is not found: {table_path}")
    
    def read_EM_csv(input_csv, tableA_csv, tableB_csv, dataset_path=dataset_dir):  
        input_csv = os.path.join(dataset_path, input_csv)
        tableA_csv = os.path.join(dataset_path, tableA_csv)
        tableB_csv = os.path.join(dataset_path, tableB_csv)
        # output_json = os.path.join(dataset_path, output_json)
        json_datas = []

            
        with open(input_csv, 'r', encoding='utf-8') as f:  
            reader = csv.DictReader(f)  
            rows = list(reader)  

        with open(tableA_csv, 'r', encoding='utf-8') as f:  
            reader = csv.DictReader(f)  
            tableA_header = reader.fieldnames  
            tableA_data = {row['id']: row for row in reader}  
    
        with open(tableB_csv, 'r', encoding='utf-8') as f:  
            reader = csv.DictReader(f)  
            tableB_header = reader.fieldnames  
            tableB_data = {row['id']: row for row in reader}  
    
        nonlocal columns
        if columns is None:
            columns = [tableA_header, tableB_header]
            print(f"[Warning] All columns are read:\n{columns}") 
        
        for row in rows:  
            ltable_id = row['ltable_id']  
            rtable_id = row['rtable_id']  
            label = row['label']  

            # 根据ltable_id和rtable_id获取tableA和tableB的数据  
            tableA_row = tableA_data.get(ltable_id)  
            tableB_row = tableB_data.get(rtable_id) 
    
            if tableA_row is None or tableB_row is None:
                print(f"[Warning] one pair {ltable_id} {rtable_id} is None")  
                continue
                # raise  ValueError("Existed None row")


            tableA_list = []
            for header in columns[0]:    
                tableA_list.append([header.replace('_', ' ').lower(), tableA_row[header]]) 

            tableB_list = []
            for header in columns[1]:  
                tableB_list.append([header.replace('_', ' ').lower(), tableB_row[header]]) 

            # tableA_list.append(['modelno', 'nan'])
            # tableB_list.append(['modelno', 'nan'])
            # print(f"[ERROR] add modelno")
                                                                                                        
            json_datas.append({
                "entityA": tableA_list,
                "entityB": tableB_list,
                "label": "Yes" if label=='1' else "No"
                })

        return json_datas
        
    def read_DI_csv(table_csv, query_column, dataset_path=dataset_dir):
        table_csv = os.path.join(dataset_path, table_csv)
        json_datas = []
        with open(table_csv, 'r') as f:  
            reader = csv.DictReader(f)  
            nonlocal columns
            if columns is None:
                columns = reader.fieldnames
                print(f"[Warning] All columns are read:\n{columns}") 

            for row in reader:
                # skip nan ground truth
                if row[query_column] is None or row[query_column] == 'nan' or row[query_column] == '':
                    continue
                masked_row = row.copy()
                masked_row[query_column] = "nan"
                try:
                    json_datas.append({
                        # "entity": [(key, value) for key, value in masked_row.items()],
                        "entity": [(attr, masked_row[attr]) for attr in columns],
                        "column": query_column, 
                        "label": row[query_column]
                    })
                except:
                    print(f"[ERROR]mask fail in line 284: {row}")

        return json_datas

    def read_ED_csv(table_csv, dataset_path=dataset_dir):
        table_csv = os.path.join(dataset_path, table_csv)
        json_datas = []
        read_idx_file = True

        if read_idx_file:
            table_df = pd.read_csv(os.path.join(dataset_path, 'table.csv'), dtype=str)
        
        with open(table_csv, 'r') as f:
            reader = csv.DictReader(f)
            nonlocal columns

            for row in reader:
                if read_idx_file:
                    table_row = table_df.iloc[int(row['row_id'])]
                    # print(table_row)
                    # if int(table_row['row_id']) != int(row['row_id']):
                    #     raise ValueError(f"row_id is not match : get {table_row['row_id']} expected {row['row_id']}")
                    row.update(table_row)

                if columns is None:
                    if read_idx_file:
                        columns = table_df.columns.to_list()
                    else:
                        columns = reader.fieldnames
                    print(f"[Warning] All columns are read:\n{columns}") 
                
                json_datas.append({
                    "entity": [(attr,row[attr]) for attr in columns],
                    "column": None, 
                    "label": "No" if row['is_clean']=='1' else "Yes",
                    'attribute': row['col_name'],
                    'value': str(row[row['col_name']]),
                    })
        return json_datas
    
    def read_SM_csv(table_csv, dataset_path=dataset_dir):
        table_csv = os.path.join(dataset_path, table_csv)
        json_datas = []
        ltable_prefix = 'ltable.'
        rtable_prefix = 'rtable.'

        with open(table_csv, 'r') as f:  
            reader = csv.DictReader(f)
            nonlocal columns
            if columns is None:
                columns = reader.fieldnames
                print(f"[Warning] All columns are read:\n{columns}")       

            ltable_cols = [ltable_prefix+col for col in columns]
            rtable_cols = [rtable_prefix+col for col in columns]

            for row in reader:
                # try:
                    json_datas.append({
                        "entityA": [(col[len(ltable_prefix):], row[col]) for col in ltable_cols],
                        "entityB": [(col[len(rtable_prefix):], row[col]) for col in rtable_cols],
                        "label": "Yes" if row['label']=='1' else "No"
                    })
                # except:
                #     print(f"[ERROR]mask fail in line 363: {row}")

        return json_datas
            
    def read_CTA_csv(table_csv, dataset_path=dataset_dir):
        table_csv = os.path.join(dataset_path, table_csv)
        json_datas = []
        with open(table_csv, 'r') as f:  
            reader = csv.DictReader(f)  
            nonlocal columns
            if columns is None:
                columns = reader.fieldnames
                print(f"[Warning] All columns are read:\n{columns}") 

            for row in reader:
                json_datas.append({
                    "entity": row['dataSample'],
                    "label": row['field'],
                    "table_type": row.get('table_type', None),
                })

        return json_datas

    def read_CTA_pkl(table_csv, dataset_path=dataset_dir):
        table_csv = os.path.join(dataset_path, table_csv)
        table_pkl = table_csv.replace('.csv', '.pkl')
        json_datas = []
        with open(table_pkl, 'rb') as f:
            table_data = pickle.load(f)
            for item in table_data:
                json_datas.append({
                    "entity": item[1],
                    "label_list": [sotab_label.get(value) for value in item[2]],
                    "table_type": item[3],
                })
                if 'sotab2' in table_pkl:
                    json_datas[-1]['label'] = ', '.join([f"Column {index+1}: {sotab_label.get(value)}" for index, value in enumerate(item[2])])
                else:
                    json_datas[-1]['label'] = str(item[3]) if str(item[3])!="MusicRecording" else "Music Recording"

        return json_datas

    def read_AVE_csv(table_csv, dataset_path=dataset_dir):
        table_csv = os.path.join(dataset_path, table_csv)
        json_datas = []
        with open(table_csv, 'r') as f:  
            reader = csv.DictReader(f)  
            nonlocal columns
            if columns is None:
                columns = reader.fieldnames
                print(f"[Warning] All columns are read:\n{columns}") 

            for row in reader:
                json_datas.append({
                    "entity": row['input'],
                    "column": row['target_key'],
                    "label": row['target_value'],
                })

        return json_datas
    
    def read_DC_csv(table_csv, dataset_path=dataset_dir):
        table_csv = os.path.join(dataset_path, table_csv)
        json_datas = []

        if read_idx_file:
            table_df = pd.read_csv(os.path.join(dataset_path, 'table.csv'), dtype=str)
        
        with open(table_csv, 'r') as f:
            reader = csv.DictReader(f)
            nonlocal columns

            for row in reader:
                if read_idx_file:
                    table_row = table_df.iloc[int(row['row_id'])]
                    row.update(table_row)

                if columns is None:
                    if read_idx_file:
                        columns = table_df.columns.to_list()
                    else:
                        columns = reader.fieldnames
                    print(f"[Warning] All columns are read:\n{columns}") 
                
                json_datas.append({
                    "entity": [(attr,row[attr]) for attr in columns],
                    "column": None, 
                    "label": str(row['clean_value']),
                    'attribute': row['col_name'],
                    'value': row[row['col_name']],
                    })
        return json_datas
    
    jsons = []

    sp_file = split+'.csv'
    if task_type == 'EM':
        json_test = read_EM_csv(sp_file, 'tableA.csv', 'tableB.csv')
    elif task_type == 'DI':
        json_test = read_DI_csv(sp_file, query_column)
    elif task_type == 'ED':
        json_test = read_ED_csv(sp_file)
    elif task_type == 'SM':
        json_test = read_SM_csv(sp_file)
    elif task_type == "CTA":
        json_test = read_CTA_pkl(sp_file)
    elif task_type == "AVE":
        json_test = read_AVE_csv(sp_file)
    elif task_type == "DC":
        json_test = read_DC_csv(sp_file)
    else:
        raise ValueError("Invalid read_task_type: {}".format(task_type))
    jsons.extend(json_test)
    return jsons

def arrest_input(instance, dataset_name, task_type, entity_name=None):
    tmp_prompt = ""
    if task_type == 'EM':
        tmp_prompt += f"{entity_name.capitalize()} A: " + serialize_([instance['entityA']]) + "\n"
        tmp_prompt += f"{entity_name.capitalize()} B: " + serialize_([instance['entityB']]) + "\n"
    elif task_type == 'SM':
        tmp_prompt += f"Attribute A is " + serialize_([instance['entityA']]) + "\n"
        tmp_prompt += f"Attribute B is " + serialize_([instance['entityB']]) + "\n"
    elif task_type == 'ED':
        serialized_instance = serialize_([instance['entity']])
        tmp_prompt += f"Record " + serialized_instance + "\n"
    elif task_type == 'DI':
        serialized_instance = serialize_([instance['entity']])
        tmp_prompt += f"Record: " + serialized_instance + "\n"
    elif task_type == "CTA":
        if dataset_name.endswith('2'): # step 2: column-wise
            tmp_prompt += f"Column: " + instance['entity'] + "\n"
        else:
            tmp_prompt += f"Classify this table: " + instance['entity'] + "\n"
    elif task_type == "AVE":
        serialized_instance = instance['entity']
        tmp_prompt += f"Product title: [" + serialized_instance + "]\n"
    elif task_type == 'DC':
        serialized_instance = serialize_([instance['entity']])
        tmp_prompt += f"Record " + serialized_instance + "\n"

    return tmp_prompt

def load_dataset(dataset_name, task_type, columns=None, query_column=None, entity_name="entity", split="test", random_seed=42):
    random.seed(random_seed)
    dataset_dir = os.path.join('./data', dataset_name)
    data = get_input(dataset_dir, task_type, columns, query_column)

    json_data = []
    for item in data:
        json_data.append({
            "dataset": dataset_name,
            "task_type": task_type_map[task_type],
            "entity": arrest_input(item, dataset_name, task_type, entity_name),
            "output":item["label"]
        })
        if 'value' in item: # for calculation the recall of DC
            json_data[-1]['value'] = item['value']
            json_data[-1]['attribute'] = item['attribute']
        if 'label_list' in item: # for calculation the recall of DC
            json_data[-1]['label_list'] = item['label_list']
        if 'table_type' in item: # for step-2 of CTA
            json_data[-1]['table_type'] = item['table_type']
    
    with open(os.path.join(dataset_dir, split+".json"), 'w', encoding='utf-8') as f:  
        json.dump(json_data, f, ensure_ascii=False, indent=4)
        print(f"[INFO] {dataset_name}'s {split}({len(json_data)}) has been built.") 
