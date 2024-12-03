from AKB import llm, data, evaluate, template
import numpy as np
import re
from sklearn.metrics import f1_score, recall_score, accuracy_score
import torch
import json
import random
from sentence_transformers import SentenceTransformer
import faiss
from itertools import chain

special_output_token = '[[[[OUTPUT]]]]'


# def get_query(prompt, eval_template, input_, output_, demo_data, demos_template):
#     """
#     Returns the text sent to the LLM for likelihood evaluation.
#     Parameters:
#         prompt: The prompt.
#         eval_template: The template for the evaluation queries.
#         input_: The input.
#         output_: The output.
#     Returns:
#         The query for the LLM and the range of the output text in the form of (start_idx, end_idx).
#     """
#     demos = demos_template.fill(demo_data)
#     query = eval_template.fill(prompt=prompt,
#                                input=input_,
#                                output=output_,
#                                full_demo=demos)
#     # query_without_output = eval_template.fill(prompt=prompt,
#     #                                           input=input_,
#     #                                           output=special_output_token,
#     #                                           full_demo=demos)

#     # first_idx = query_without_output.find(special_output_token)
#     # output_idx = first_idx, first_idx + len(output_)
#     return query

def get_query(eval_template, prompt, item):
    query = eval_template.fill(
        prompt=prompt,
        input=item['input'].strip() if eval_template.task_type in [5] else item['entity'], # ED and AVE has instance-wise question
        output=item['output'],
        comma=template._sotab_comma.get(item.get('table_type',None), None), # CTA 2-step
        attribute=item.get('attribute', None), # ED 
        value=item.get('value', None), 
    )
    return query


def f1_evaluator(task_type, prompts, eval_template, eval_data, demos_template, few_shot_data, config):
    """
    For each prompt, evaluate the f1-score of the data (output) given the prompt.
    Parameters:
        prompts: A list of prompts.
        eval_template: The template for the evaluation queries.
        eval_data: The data to use for evaluation.
        config: The configuration dictionary.
    Returns:
        A f1EvaluationResult object.
    """
    queries = []
    input_data = []
    # output_indices = []
    subsampled_data = data.subsample_data(
            eval_data, config['num_samples'])
    
    for prompt in prompts:    
        # for d in zip(*subsampled_data):
        for item in subsampled_data:
            # if eval_template.task_type in [0]: # ED has instance-wise question
            #     input_ = item['input'].strip()
            # else:
            #     input_ = item['entity']
            # output_ = item['output']
            # demo_data = data.subsample_data(
            #     few_shot_data, config['num_few_shot'])
            # query = get_query(
            #     prompt, eval_template, input_, output_, demo_data, demos_template)
            # query = re.sub(r'Output:.*', 'Output: ', query, flags=re.DOTALL)
            query = get_query(eval_template, prompt, item)
            queries.append(query)
            input_data.append(item)
            # output_indices.append(output_idx)
    model = llm.model_from_config(config['model'])
    outputs = model.generate_text(queries, 1)
    # model.cleanup()
    # del model
    # torch.cuda.empty_cache()

    res = F1EvaluationResult(task_type, prompts, outputs, input_data, config['num_samples'], queries)

    return res


class F1EvaluationResult(evaluate.EvaluationResult):
    """
    A class for storing the results of a f1-score evaluation. Supports
    sorting prompts by various statistics of the f1-score.
    """

    def __init__(self, task_type, prompts, outputs, input_data, num_samples, queries):
        self.prompts = prompts
        self.task_type = task_type
        self.binary_classification = True if task_type in [0,2,3] else False
        self.multiple_classification = True if task_type in [1,4] else False
        print(f"[INFO] task is {'not' if not self.binary_classification else ''} binary_classification task!")
        self.scores, self.errors = self._compute_avg_f1(prompts, outputs, input_data, num_samples, queries)
        print(f"[DEBUG] the {len(self.errors)}")
        
            
    def preprocess(self, s):
        # 定义需要删除的短语
        patterns = ['however', 'therefore', 'answer:', 'answer:', '\nsince', '\nbased on', 'conclude that', '\nso', 'given table:', 'classified columns:']
        
        # 构建正则表达式，将所有模式都加入到正则表达式中，删除短语之前的所有内容
        pattern = re.compile(rf".*({ '|'.join(patterns) }).*", re.IGNORECASE)
        
        # 查找所有匹配的短语
        matches = list(re.finditer(pattern, s))
        
        # 如果有匹配，保留最后一个匹配及其之后的部分
        if matches:
            last_match = matches[-1]
            s = s[last_match.start(1):]

        if self.task_type == 4:
            if 'Column 1' in s:
                s = s[s.index('Column 1'):]
            if 'column 1' in s:
                s = s[s.index('column 1'):]
            if 'column1' in s:
                s = s[s.index('column1'):]
            if 'Column1' in s:
                s = s[s.index('Column1'):]
        
        return s.strip()  # 去掉前后的空格

    def label_mapping(self, s):
        s = s.strip().lower()
        s = self.preprocess(s)
        if self.binary_classification:
            if s.startswith("yes"):
                return 1
            elif s.startswith("no"):    
                return 0
            
            same_substring = ["are the same", "be the same", "indeed the same", "yes", "be considered the same"]
            diff_substring = ["be different", "not the same", "not identical", "no"]

            same_pos = [s.rfind(ss) for ss in same_substring]
            diff_pos = [s.rfind(ss) for ss in diff_substring]

            last_same = max(same_pos)
            last_diff = max(diff_pos)

            if last_same > last_diff:
                return 1
            elif last_diff > last_same:
                return 0
            else:
                print(f'[Warning] unexpected answer\n{s}')
                return random.choice([0,1])
        else:
            return s.lower().strip()
        
    def CTA_processing(self, table_preds, table_labels):
        table_number = len(table_labels)

        # if "Class:" in table_preds:
        #     table_preds = table_preds.split("Class:")[1]
        # if "column 1:" in table_preds:
        #     table_preds = table_preds.split("column 1:")[1]

        #Break predictions into either \n or ,
        if ":" in table_preds or "-" in table_preds:
            if ":" in table_preds:
                separator = ":"
                start = 1
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
        # sotab_set = set(chain.from_iterable(template._sotab_map.values()))
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
            
            if pred in template._sotab_map.get(label, []):
                labeled_preds.append(label)
            elif pred in template._sotab_map.keys() and pred != '-':
                # print(f"[DEBUG] len of sotab is {len(template._sotab_map.keys())}")
                # print(f"For test example out of {label} label space prediction: {pred}")
                labeled_preds.append(pred)
            else:
                labeled_preds.append("-")
                print(f"For test example out of label space prediction: {label} == {pred}")
                # for key,value in template._sotab_map.items():
                #     if key != label:
                #         if pred in value:
                #             labeled_preds.append(key)
                #             break
                # if len(labeled_preds) <= id:
                #     labeled_preds.append("-")

        return labeled_preds
        
    def CTA_mapping(self, y_pred, y_true):
        for idx, pair in enumerate(zip(y_pred, y_true)):
            pred, label = pair
            pred = pred.split(',', 1)[0]
            if pred in template._sotab_map.get(label, []):
                y_pred[idx] = label
            elif pred in template._sotab_map or pred == 'i don\'t know':
                y_pred[idx] = pred
            else:
                print(f"[Warning] pred out of candidate set: {label} == {pred}?")
                y_pred[idx] = "i don't know"
        return y_pred


    def DC_mapping(self, response, output):
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

        
    def _compute_avg_f1(self, prompts, outputs, input_data, num_samples, queries):
        prompt_f1s = []
        print(f"[DEBUG] len of input_data{len(input_data)}")
        print(f"[DEBUG] num_samples{num_samples}")
        ground_truths = [self.label_mapping(item['output']) for item in input_data]
        predictions = [self.label_mapping(s) for s in outputs]
        error_samples = []
        # if self.multiple_classification: # DI, CTA
        #     label_set = list(set(ground_truths))
        #     sbert = SentenceTransformer('/share/home/12351018/pre-train/sentence-transformers')
        #     label_set_embeddings = sbert.encode(label_set).astype('float32')

        #     # 建立faiss索引
        #     dimension = label_set_embeddings.shape[1]  # 向量的维度
        #     index = faiss.IndexFlatL2(dimension)
        #     index.add(label_set_embeddings)
        #     prediction_embeddings = sbert.encode(predictions).astype('float32')
        #     # 检索最相似的 label_set
        #     retrieved_predictions = []
        #     for prediction_embedding in prediction_embeddings:
        #         D, I = index.search(np.array([prediction_embedding]), 1)  # 搜索1个最相似的结果
        #         # 通过索引从 label_set 中获取最相似的结果
        #         most_similar_label = label_set[I[0][0]]
        #         retrieved_predictions.append(most_similar_label)

        #     predictions = retrieved_predictions
            
                    
        for idx, prompt in enumerate(prompts):
            y_true = [gt for gt in ground_truths[idx*num_samples : (idx+1)*num_samples]]
            y_pred = [out for out in predictions[idx*num_samples : (idx+1)*num_samples]]
            correction = [x==y for x, y in zip(y_true, y_pred)]
            if self.task_type == 1: # DI accuracy
                avg_acc = accuracy_score(y_true, y_pred)
                avg_acc = round(avg_acc*100, 2)
                prompt_f1s.append(avg_acc)

            elif self.task_type == 4 and 'olumn' in input_data[0]['output']: # CTA step-2: table wise
                label_list = [item['label_list'] for item in input_data]
                origin_pred = y_pred.copy()
                y_pred = []
                y_true = []
                for i, (x, y) in enumerate(zip(origin_pred, label_list)):
                    y_pred.extend(self.CTA_processing(x, y))
                    y_true.extend(y)
                    # f1 = f1_score(y_true, y_pred, average='micro', labels=list(set(label_list)))
                    predictions[idx*num_samples + i] = self.CTA_processing(x, y)
                correction = [all(x == y for x, y in zip(label, pred)) for label, pred in zip(label_list, predictions[idx*num_samples : (idx+1)*num_samples])]

                # types = list(set(y_true))
                types = list(template._sotab_map.keys())
                types = types + ["-"]
                assert len(types) == 33
                num_classes = len(types)
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
                    # f1 = 2*precision*recall / (precision + recall)
                    f1 = np.divide(2 * precision * recall, precision + recall, where=(precision + recall) != 0)
                    
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
                avg_f1  = round(evaluation['Micro-F1']*100, 2)
                avg_recall = round(evaluation['Recall']*100)
                avg_acc = round(evaluation['Precision']*100)
                prompt_f1s.append(avg_f1+avg_acc/100000+avg_recall/100000000)

            elif self.binary_classification or self.task_type == 4: # CTA micro-f1 step-1 is depecated
                if self.task_type == 4:
                    # y_pred = [y  if y in template._sotab_map else "i don't know" for y in y_pred]
                    y_pred = self.CTA_mapping(y_pred, y_true)
                    predictions[idx*num_samples:(idx+1)*num_samples] = y_pred
                    correction = [x==y for x, y in zip(y_true, y_pred)]
                    avg_f1 = f1_score(y_true, y_pred, average='micro', labels=list(set(y_true+y_pred) - {"i don't know"}))
                    avg_recall = recall_score(y_true, y_pred, average='micro', labels=list(set(y_true+y_pred) - {"i don't know"}))
                else:
                    avg_f1 = f1_score(y_true, y_pred, average='micro' if self.task_type==4 else 'binary')
                    avg_recall = recall_score(y_true, y_pred, average='micro' if self.task_type==4 else 'binary')
                avg_f1 = round(avg_f1*100, 2)
                avg_recall = round(avg_recall*100)
                avg_acc = accuracy_score(y_true, y_pred)
                avg_acc = round(avg_acc*100)
                prompt_f1s.append(avg_f1+avg_acc/100000+avg_recall/100000000)
                # predictions = y_pred # BUG 破坏了for训练下一轮的计算
                # predictions[idx*num_samples : (idx+1)*num_samples] = y_pred
            
            elif self.task_type == 5: # AVE specific F1-score
                y_pred = [y.split(',')[0] for y in y_pred]
                NN = sum(1 for x, y in zip(y_true, y_pred) if y == 'n/a' and x == 'n/a') # no predicted value when the ground truth does not contain an attribute value
                NV = sum(1 for x, y in zip(y_true, y_pred) if y == 'n/a' and x != 'n/a' and x != y) # incorrect predicted value when the ground truth does not contain an attribute value
                VN = sum(1 for x, y in zip(y_true, y_pred) if x != 'n/a' and y == 'n/a') # no predicted value when the ground truth contains an attribute value
                VC = sum(1 for x, y in zip(y_true, y_pred) if x != 'n/a' and y != 'n/a' and x == y)# correct predicted value thatexactly matches the attribute value in the ground truth
                VW = sum(1 for x, y in zip(y_true, y_pred) if x != 'n/a' and y != 'n/a' and x != y) # incorrect predicted value that does not match the attribute value inthe ground truth
                P = VC / (NV + VC + VW)
                R = VC / (VN + VC + VW)
                # F1 = 2*P*R / (P + R) 
                F1 = np.divide(2 * P * R, P + R, where=(P + R) != 0)
                avg_f1 = round(F1 * 100, 2)
                avg_acc = round(VC/(VC+VW) * 100)
                avg_recall = round(R*100)
                prompt_f1s.append(avg_f1+avg_acc/100000+avg_recall/100000000)

            elif self.task_type == 6: # DC F1-score
                dirty = [item['output']!=item['value'] for item in input_data[idx*num_samples : (idx+1)*num_samples]]
                correct = [self.DC_mapping(y, x) for x, y in zip(y_true, y_pred)]
                correction = correct
                C = sum(correct)
                D = sum(dirty)
                DC = sum(1 for x, y in zip(correct, dirty) if x and y)
                All = len(y_pred)
                P = C / All # The number of correctly repaired tuples over the total number of repaired tuples in data, which assesses the correctness of repairing;
                R = DC / D #  the number of correctly repaired tuples over the total number of dirty tuples, which assesses the completeness
                # F1 = 2*P*R / (P + R) 
                F1 = np.divide(2 * P * R, P + R, where=(P + R) != 0)
                avg_f1 = round(F1 * 100, 2)
                avg_acc = round(C / All * 100)
                avg_recall = round(R * 100)
                prompt_f1s.append(avg_f1+avg_acc/100000+avg_recall/100000000)

            
            # save error when test
            if prompt_f1s[-1] >= prompt_f1s[0]:
                error_samples = []
                for i in range(num_samples):
                    if not correction[i]:
                        error_samples.append({
                            **input_data[i],
                            "prediction": outputs[idx*num_samples + i],
                            "response": predictions[idx*num_samples + i],
                            "correction": correction[i]
                        })

        return prompt_f1s, error_samples

    def sorted(self, method='default'):
        # Sort prompts by score
        sorted_prompts = [p for _, p in sorted(zip(self.scores, self.prompts))]
        sorted_scores = sorted(self.scores)
        # Reverse both and convert to lists
        sorted_prompts = list(reversed(sorted_prompts))
        sorted_scores = list(reversed(sorted_scores))
        return sorted_prompts, sorted_scores

    def in_place(self, method='default'):
        if method == 'default':
            scores = self._agg_likelihoods('mean')
        else:
            scores = self._agg_likelihoods(method)
        return self.prompts, scores

    def __str__(self):
        s = ''
        prompts, scores = self.sorted()
        s += 'log(p): prompt\n'
        s += '----------------\n'
        for prompt, score in list(zip(prompts, scores))[:10]:
            s += f'{score:.2f}: {prompt}\n'
        return s
