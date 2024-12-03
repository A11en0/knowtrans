import os
import json
import random
import numpy as np

from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from constants import ENTERPRICE_OLD_INSTRUCTION, ENTERPRICE_NEW_INSTRUCTION, ABT_BUY_OLD_KNOWLEDGE, ABT_BUY_NEW_KNOWLEDGE, ABT_BUY_OLD_INSTRUCTION, ABT_BUY_NEW_INSTRUCTION, ENTERPRICE_OLD_KNOWLEDGE, ENTERPRICE_NEW_KNOWLEDGE, WALMART_AMAZON_OLD_INSTRUCTION, WALMART_AMAZON_NEW_INSTRUCTION, WALMART_AMAZON_OLD_KNOWLEDGE, WALMART_AMAZON_NEW_KNOWLEDGE



def load_dataset(model_type, tokenizer, file_name, shuffle=False, insert_knowledge=None, reasoning=None, few_shot=False, BASE_DATA_DIR=''):
    # BASE_DATA_DIR = "data/Jellyfish-rag/test"
    data_lists = dict()
    file_names = file_name.split(",") if ',' in file_name else [file_name]
    
    for _file in file_names:
        if _file:
            with open(os.path.join(BASE_DATA_DIR, _file) + ".json", 'r') as f:
                data_list = json.load(f)
            
            data_list = process_data_list(model_type, tokenizer, data_list, _file, reasoning, insert_knowledge, few_shot)
            
            if shuffle:
                random.shuffle(data_list)
            
            # data_lists.append((_file, data_list))
            data_lists[_file] = data_list
    
    return data_lists

def process_data_list(model_type, tokenizer, data_list, file_name, reasoning, insert_knowledge, few_shot):
    if model_type == "mistral": 
        for item in data_list:
            item['instruction'] = item['instruction'].replace("[INST]:", "[INST]")
            # add system prompt
            sys_prompt = "You are an AI assistant that follows instruction extremely well. User will give you a question. Your task is to answer as faithfully as you can."
            item['instruction'] = f"{sys_prompt}\n\n[INST]\n\n{item['instruction']}\n\n[DEMOS]{item['input']}\n[/INST]\n\n"
    elif model_type == 'llama3': 
        for item in data_list:
            prompt = f"{item['instruction']}\n\n[DEMOS]{item['input']}"
            messages = [
                    {
                        "role": "system", 
                        "content": "You are an AI assistant that follows instruction extremely well. User will give you a question. Your task is to answer as faithfully as you can."
                    },
                    {"role": "user", "content": prompt}
            ]

            item['instruction'] = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, return_tensors="pt")
    elif model_type == "jellyfish-13b": 
        for item in data_list:
            # add system prompt
            sys_prompt = "You are an AI assistant that follows instruction extremely well. User will give you a question. Your task is to answer as faithfully as you can."
            item['instruction'] = f"{sys_prompt}\n\n### Instruction:\n\n{item['instruction']}\n\n[DEMOS]{item['input']}\n\n### Response:\n\n"
            
    if few_shot:
        data_list = apply_few_shot(data_list, file_name)
    else:
        for item in data_list:
            item['instruction'] = item['instruction'].replace("[DEMOS]", "")

    if reasoning:
        data_list = apply_reasoning(data_list, file_name)

    if insert_knowledge:
        data_list = insert_additional_knowledge(data_list, file_name)
    
    return data_list

def apply_few_shot(data_list, file_name):
    few_shot_examples = get_few_shot_examples(file_name)
    
    for item in data_list:
        few_shot_prompt = create_few_shot_prompt(few_shot_examples)
        item['instruction'] = item['instruction'].replace("[DEMOS]", few_shot_prompt)

    return data_list

def get_few_shot_examples(file_name):
    if file_name == "enterprise":
        return [
            {"instruction": "判断两个企业是否相同: A公司: 阿里巴巴集团控股有限公司, B公司: 阿里巴巴(中国)有限公司", "output": "不相同"},
            {"instruction": "判断两个企业是否相同: A公司: 腾讯科技(深圳)有限公司, B公司: 腾讯科技(深圳)有限公司", "output": "相同"},
            {"instruction": "判断两个企业是否相同: A公司: 百度在线网络技术(北京)有限公司, B公司: 北京百度网讯科技有限公司", "output": "不相同"}
        ]
    elif file_name == "abt-buy":
        return [
            {"instruction": "Product A: [name: \"samsung s3 black multimedia player yps3jab\", description: \"samsung s3 black multimedia player yps3jab 4 gb internal flash memory 1.8 ' tft lcd display touch-sensitive led controls multi-formats support dnse 2.0 sound engine fm tuner and recorder with presets up to 25 hours audio playback up to 4 hours video playback black finish\"]\nProduct B: [name: \"samsung 4gb portable mltimdia plyr blk yps-s3jab / xaa\", description: \"nan\"]\nAre Product A and Product B the same Product?\nChoose your answer from: [Yes, No]", "output": "Yes"},
            {"instruction": "Product A: [name: \"sony white 8 ' portable dvd player dvpfx820w\", description: \"sony dvp-fx820 white 8 ' portable dvd player dvpfx820w swivel & flip screen with dual sensor for remote control control buttons on screen bezel 12 bit video dac with 108 mhz processing removable ,rechargeable battery & car adapter included white finish\"]\nProduct B: [name: \"toshiba sd-p71s portable dvd player\", description: \"toshiba sd-p71s 7 'portable dvd player\"]\nAre Product A and Product B the same Product?\nChoose your answer from: [Yes, No]", "output": "No"},
            {"instruction": "Product A: [name: \"sony xplod 10-disc add-on cd/mp3 changer cdx565mxrf\", description: \"sony xplod 10-disc add-on cd/mp3 changer cdx565mxrf cd/cd-r/cd-rw and mp3 playback mp3 decoding d-bass 12-second advanced electronic shock protection fm modulator 9 modulation frequencies wireless remote\"]\nProduct B: [name: \"sony cdx-565mxrf 10-disc cd/mp3 changer\", description: \"nan\"]\nAre Product A and Product B the same Product?\nChoose your answer from: [Yes, No]", "output": "Yes"},
        ]
    elif file_name == 'walmart-amazon':
        return [
            {"instruction": "Product A: [name: \"d-link dgs-1005g 5-port gigabit desktop switch\", modelno: \"dgs1005g\"]\nProduct B: [name: \"d-link dgs-1005g 5-port gigabit desktop switch\", modelno: \"dgs-1005g\"]\nAre Product A and Product B the same Product?\nChoose your answer from: [Yes, No]", "output": "Yes"},
            {"instruction": "Product A: [name: \"nzxt phantom crafted series atx full tower steel chassis black\", modelno: \"nzxt phantom\"]\nProduct B: [name: \"nzxt crafted series atx full tower steel chassis - phantom white\", modelno: \"phantom white\"]\nAre Product A and Product B the same Product?\nChoose your answer from: [Yes, No]", "output": "No"},
            {"instruction": "Product A: [name: \"at t prepaid gophone samsung a187 with bluetooth blue\", modelno: \"a187\"]\nProduct B: [name: \"samsung a107 prepaid gophone at t\", modelno: \"a107\"]\nAre Product A and Product B the same Product?\nChoose your answer from: [Yes, No]", "output": "Yes"},
        ]
    else:
        return []  # 对于其他数据集,暂时不提供少样本示例

def create_few_shot_prompt(examples):
    prompt = ""
    for example in examples:
        # prompt += f"### Instruction: {example['instruction']}\n### Response: {example['output']}\n\n"
        prompt += f"[INST]:\n{example['instruction']}\n {example['output']}\n[/INST]\n\n"
    # prompt += f"现在,请回答以下问题:\n输入: {instruction}\n输出:"
    return prompt

def apply_reasoning(data_list, file_name):
    if file_name == "enterprise":
        for item in data_list:
            item['instruction'] = item['instruction'].replace(ENTERPRICE_OLD_INSTRUCTION, ENTERPRICE_NEW_INSTRUCTION)
    if file_name == "abt-buy":
        for item in data_list:
            item['instruction'] = item['instruction'].replace(ABT_BUY_OLD_INSTRUCTION, ABT_BUY_NEW_INSTRUCTION)
    if file_name == "walmart-amazon":
        for item in data_list:
            item['instruction'] = item['instruction'].replace(WALMART_AMAZON_OLD_INSTRUCTION, WALMART_AMAZON_NEW_INSTRUCTION)
    return data_list

def insert_additional_knowledge(data_list, file_name):
    '''
    根据不同的文件名，对数据集进行不同的处理
    '''
    if file_name == "enterprise":
        for item in data_list:
            item['instruction'] = item['instruction'].replace(ENTERPRICE_OLD_KNOWLEDGE, ENTERPRICE_NEW_KNOWLEDGE)
    
    if file_name == "abt-buy":
        for item in data_list:
            item['instruction'] = item['instruction'].replace(ABT_BUY_OLD_KNOWLEDGE, ABT_BUY_NEW_KNOWLEDGE)

    if file_name == "walmart-amazon":
        for item in data_list:
            item['instruction'] = item['instruction'].replace(WALMART_AMAZON_OLD_KNOWLEDGE, WALMART_AMAZON_NEW_KNOWLEDGE)
    
    # 可以在这里添加其他文件类型的处理逻辑
    return data_list

def eval_results(save_list, file_name, input_data, save_path=None, print_results=True, lower=False):
    label_sets = []
    for item in save_list:
        if lower:
            label_sets.append(item['label'].lower())
        else:
            label_sets.append(item['label'])
    label_sets = list(set(label_sets))

    label_list, pred_list, error_save_list = [], [], []
    for item in save_list:
        pred, label = item['prediction'], item['label']
        
        if item['dataset'] == 'sotab2':
            if "\n\n" in pred:
                pred = pred.split("\n\n")[1]
        else:
            pred = pred.replace(" { ", "").strip()

            if lower:
                pred, label = pred.lower(), label.lower()

            if "final answer:" in pred.lower():
                pred = pred.lower().split("final answer:")[-1].strip()

        if pred not in label_sets:
            pred = random.choice(label_sets)
            
        if pred != label:
            error_save_list.append({
                "instruction": item['instruction'], 
                "output": item['prediction'], 
                "prediction": pred, 
                "label": label
            })

        pred_list.append(pred)
        label_list.append(label)

    if save_path: 
        with open(os.path.join(save_path, 'errors.json'), "w") as f:
            json.dump(error_save_list, f, ensure_ascii=False)
    
    if len(label_sets) > 2:
        if any(['ae' in file_name, 'oa' in file_name]): 
            results = {
                "F1-score": ave_f1_score(label_list, pred_list)
            } 
        elif 'sotab2' in file_name: 
            results = {
                "Micro-f1": cta_f1_score(label_list, pred_list, input_data['sotab2'])
            } 
        elif 'hospital_DC' in file_name: 
            results = {
                "F1-score": dc_f1_score(label_list, pred_list, input_data['hospital_DC'])
            } 
        elif 'rayyan_DC' in file_name: 
            results = {
                "F1-score": dc_f1_score(label_list, pred_list, input_data['rayyan_DC'])
            } 
        elif 'beers_DC' in file_name: 
            results = {
                "F1-score": dc_f1_score(label_list, pred_list, input_data['beers_DC'])
            } 
        else: 
            results = {
                "Accuracy": accuracy_score(label_list, pred_list),
                "Micro-f1": f1_score(label_list, pred_list, average='micro'),
                "Micro-Precision": precision_score(label_list, pred_list, average='micro'),
                "Micro-Recall": recall_score(label_list, pred_list, average='micro'),
                "Macro-f1": f1_score(label_list, pred_list, average='macro'),
                "Macro-Precision": precision_score(label_list, pred_list, average='macro'),
                "Macro-Recall": recall_score(label_list, pred_list, average='macro'),
                "Weighted-F1": f1_score(label_list, pred_list, average='weighted'),
                "Classification Report": classification_report(label_list, pred_list) 
            } 
    else:
        results = {
            "Accuracy": accuracy_score(label_list, pred_list),
            "Binary-f1": f1_score(label_list, pred_list, average='binary', pos_label='yes'),
            "Binary-Precision": precision_score(label_list, pred_list, average='binary', pos_label='yes'),
            "Binary-Recall": recall_score(label_list, pred_list, average='binary', pos_label='yes'),
            "Micro-f1": f1_score(label_list, pred_list, average='micro'),
            "Micro-Precision": precision_score(label_list, pred_list, average='micro'),
            "Micro-Recall": recall_score(label_list, pred_list, average='micro'),
            "Macro-f1": f1_score(label_list, pred_list, average='macro'),
            "Macro-Precision": precision_score(label_list, pred_list, average='macro'),
            "Macro-Recall": recall_score(label_list, pred_list, average='macro'),
            "Weighted-F1": f1_score(label_list, pred_list, average='weighted'),
            "Classification Report": classification_report(label_list, pred_list) 
        }        

    with open(os.path.join(save_path, 'metrics.txt'), "w") as file:
        for metric, value in results.items():
            file.write(f"{metric}: {value}\n")

    if print_results:
        print(f"================= results: {file_name} =================")
        for k, v in results.items():
            print(f"{k}: {v}")

    return pred_list, label_list


def ave_f1_score(y_true, y_pred):
    # AVE specific F1-score
    NN = sum(1 for x, y in zip(y_true, y_pred) if y == 'n/a' and x == 'n/a') # no predicted value when the ground truth does not contain an attribute value
    NV = sum(1 for x, y in zip(y_true, y_pred) if y == 'n/a' and x != 'n/a' and x!=y) # incorrect predicted value when the ground truth does not contain an attribute value
    VN = sum(1 for x, y in zip(y_true, y_pred) if x != 'n/a' and y == 'n/a') # no predicted value when the ground truth contains an attribute value
    VC = sum(1 for x, y in zip(y_true, y_pred) if x != 'n/a' and y != 'n/a' and x==y) # correct predicted value thatexactly matches the attribute value in the ground truth
    VW = sum(1 for x, y in zip(y_true, y_pred) if x != 'n/a' and y != 'n/a' and x!=y) # incorrect predicted value that does not match the attribute value inthe ground truth
    P = VC / (NV + VC + VW)
    R = VC / (VN + VC + VW)
    F1 = 2 * P * R / (P + R)
    return round(100 * F1, 4)
    # return round(100 * F1 + VC/(VC+VW) + R, 4)    
    # avg_f1 = round(F1 * 100)
    # avg_acc = round(VC / (VC + VW) * 100)
    # avg_recall = round(R * 100)
    # return avg_f1 + avg_acc / 100000 + avg_recall / 100000000 

def dc_f1_score(y_true, y_pred, input_data):
    dirty = [item['output']!=item['value'] for item in input_data]
    correct = [DC_mapping(y, x) for x, y in zip(y_true, y_pred)]
    C = sum(correct)
    D = sum(dirty)
    DC = sum(1 for x, y in zip(correct, dirty) if x and y)
    All = len(y_pred)
    P = C / All # The number of correctly repaired tuples over the total number of repaired tuples in data, which assesses the correctness of repairing;
    R = DC / D #  the number of correctly repaired tuples over the total number of dirty tuples, which assesses the completeness
    F1 = 2*P*R / (P + R) 
    return round(100 * F1, 4)

def DC_mapping(response, output):
    try:
        # 尝试将 response 和 output 解析为 JSON 对象进行比较
        return json.loads(response) == json.loads(output)
    except json.JSONDecodeError:
        pass

    # 去除 response 的首尾引号（如果存在）
    def simplify(s):
        if s.startswith('"') and s.endswith('"'):
            s = s[1:-1]
        if s.endswith("."):
            s = s[:-1]
        if s in ["", "null", "n/a", "nan"]:
            return "nan"
    
    return simplify(response) == simplify(output)

def cta_f1_score(y_true, y_pred, input_data):
    label_list = [item['label_list'] for item in input_data]
    origin_pred = y_pred.copy()
    y_pred = []
    y_true = []
    for x, y in zip(origin_pred, label_list):
        y_pred.extend(CTA_processing(x, y))
        y_true.extend(y)
        # f1 = f1_score(y_true, y_pred, average='micro', labels=list(set(label_list)))

    types = list(set(y_true))
    types = types + ["-"]
    num_classes = len(types) # 33
    y_tests = [types.index(y) for y in y_true]
    y_preds = [types.index(y) for y in y_pred]
    #Confusion matrix
    cm = np.zeros(shape=(num_classes,num_classes))
    for i in range(len(y_tests)):
        cm[y_preds[i]][y_tests[i]] += 1
    
    report = {}
    for j in range(len(cm[0])):
        report[j] = {}
        report[j]['FN'] = 0
        report[j]['FP'] = 0
        report[j]['TP'] = cm[j][j]

        for i in range(len(cm)):
            if i != j:
                report[j]['FN'] += cm[i][j]
        for k in range(len(cm[0])):
            if k != j:
                report[j]['FP'] += cm[j][k]

        precision = report[j]['TP'] / (report[j]['TP'] + report[j]['FP'])
        recall = report[j]['TP'] / (report[j]['TP'] + report[j]['FN'])
        f1 = 2*precision*recall / (precision + recall)
        
        if np.isnan(f1):
            f1 = 0
        if np.isnan(precision):
            f1 = 0
        if np.isnan(recall):
            f1 = 0

        report[j]['p'] =  precision
        report[j]['r'] =  recall
        report[j]['f1'] = f1
    
    all_fn = 0
    all_tp = 0
    all_fp = 0

    for r in report:
        if r != num_classes-1:
            all_fn += report[r]['FN']
            all_tp += report[r]['TP']
            all_fp += report[r]['FP']
        
    class_f1s = [ report[class_]['f1'] for class_ in report]
    class_p = [ 0 if np.isnan(report[class_]['p']) else report[class_]['p'] for class_ in report]
    class_r = [ 0 if np.isnan(report[class_]['r']) else report[class_]['r'] for class_ in report]
    macro_f1 = sum(class_f1s[:-1]) / (num_classes-1)
    
    p =  sum(class_p[:-1]) / (num_classes-1)
    r =  sum(class_r[:-1]) / (num_classes-1)
    micro_f1 = all_tp / ( all_tp + (1/2 * (all_fp + all_fn) )) 
    
    per_class_eval = {}
    for index, t in enumerate(types[:-1]):
        per_class_eval[t] = {"Precision":class_p[index], "Recall": class_r[index], "F1": class_f1s[index]}
    
    evaluation = {
        "Micro-F1": micro_f1,
        "Macro-F1": macro_f1,
        "Precision": p,
        "Recall": r
    }
    return round(100 * evaluation['Micro-F1'], 4)


def CTA_processing(table_preds, table_labels):
        table_number = len(table_labels)

        if "Class:" in table_preds:
            table_preds = table_preds.split("Class:")[1]
        if "column 1:" in table_preds:
            table_preds = table_preds.split("column 1:")[1]

        #Break predictions into either \n or ,
        if ":" in table_preds or "-" in table_preds:
            if ":" in table_preds:
                separator = ":"
                start = 0 # 1
                end = table_number+1
            else:
                separator = "-"  
                start = 1
                end = table_number+1
        else:
            separator = ","
            start = 0
            end = table_number
            
        col_preds = table_preds.split(separator)[start:end]
        labeled_preds = []
        for id, label in enumerate(table_labels):
            pred = col_preds[id] if id < len(col_preds) else "-"
            # Remove break lines
            if "\n" in pred:
                pred = pred.split('\n')[0].strip()
            # Remove commas
            if "," in pred:
                pred = pred.split(",")[0].strip()
            # Remove paranthesis
            if '(' in pred:
                pred = pred.split("(")[0].strip()
            #Remove points
            if '.' in pred:
                pred = pred.split(".")[0].strip()
            # Lower-case prediction
            pred = pred.strip().lower()
            
            if pred in _sotab_map.get(label, []):
                labeled_preds.append(label)
            else:
                print(f"For test example out of label space prediction: {pred}")
                labeled_preds.append("-")

        return labeled_preds

_sotab_map = {
    "locality of address": ["addressLocality", "locality of address"],
    "postal code": ["postalCode", "postal code"],
    "region of address": ["addressRegion", "region of address"],
    "country": ["Country", "country"],
    "price range": ["priceRange", "price range"],
    "name of hotel": ["Hotel/name", "name of hotel", "name of hotels", "name of aparthotel"],
    "telephone": ["telephone", ],
    "fax number": ["faxNumber", "fax number"],
    "date": ["Date", "date", "end date"],
    "name of restaurant": ["Restaurant/name", "name of restaurant"],
    "payment accepted": ["paymentAccepted", "payment accepted"],
    "day of week": ["DayOfWeek", "day of week"],
    "review": ["Review", "review"],
    "organization": ["Organization", "organization", "name of organization"],
    "date and time": ["DateTime", "date and time"],
    "music artist": ["MusicArtistAT", "music artist"],
    "music album": ["MusicAlbum/name", "music album", "name of album", "name of music album"],
    "name of music recording": ["MusicRecording/name", "name of music recording", "music recording", "name of song"],
    "photograph": ["Photograph", "photograph"],
    "coordinate": ["CoordinateAT", "coordinate"],
    "name of event": ["Event/name", "event name", "name of event"],
    "event attendance mode": ["EventAttendanceModeEnumeration", "event attendance mode"],
    "event status": ["EventStatusType", "event status"],
    "currency": ["currency", ],
    "email": ["email", "email address"],
    "time": ["Time", "time", "check-in time", "check-out time", "time of check-in", "time of check-out"],
    "location feature": ["LocationFeatureSpecification", "location feature", "description of hotel amenities", "description of hotel amenities", "amenities", "hotel amenities", "amenities of hotel room", "hotel features"],
    "duration": ["Duration", "duration", "duration of music recording or video"],
    "description of event": ["Event/description", "description of event", "event description", "descriptions of events"],
    "description of restaurant": ["Restaurant/description", "description of restaurant"],
    "description of hotel": ["Hotel/description", "description of hotel", "description of hotels"],
    "rating": ["Rating", "rating"],
}
