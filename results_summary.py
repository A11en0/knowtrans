import os
import re

# 定义一个函数，用于读取每个子目录中的metrics.txt文件并获取F1-score或Binary-F1
def read_metrics_files(dir_path):
    # 获取目录中的所有子目录
    subdirs = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
    
    # 定义一个空列表，用于存储每个子目录中的F1-score或Binary-F1
    f1_scores = []
    
    # 遍历每个子目录
    for subdir in subdirs:
        # 获取metrics.txt文件的路径
        metrics_file_path = os.path.join(dir_path, subdir, 'metrics.txt')
        
        # 如果metrics.txt文件存在
        if os.path.exists(metrics_file_path):
            # 读取文件内容
            with open(metrics_file_path, 'r') as file:
                content = file.read()
            
            # 使用正则表达式匹配F1-score或Binary-F1
            match = re.search(r'(F1-score|Binary-f1|Micro-f1): (\d+\.\d+)', content)
            
            # 如果匹配成功，将F1-score或Binary-F1添加到列表中            
            # if match:
                # f1_scores.append((subdir, float(match.group(2))))

            if match:
                f1_score = float(match.group(2))
                # 对 beers_DC 进行特殊处理
                if subdir in ['beers_DC', 'ae', 'hospital_DC', 'rayyan_DC', 'rayyan', 'oa'] and f1_score > 1:
                    f1_score /= 100
                f1_scores.append((subdir, f1_score))                
    
    # 返回所有子目录中的F1-score或Binary-F1
    return f1_scores

# 读取当前目录中的所有子目录
dir_path = 'jellyfish-rag/results/7B/few-shot-train20' 
f1_scores = read_metrics_files(dir_path) 

# 打印每个子目录中的F1-score或Binary-F1 
average_f1_score = sum(f1_score for _, f1_score in f1_scores) / len(f1_scores) 
for subdir, f1_score in f1_scores: 
    print(f'{subdir}: {f1_score}') 

print(f"Avg F1-score: {average_f1_score}") 
# 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 