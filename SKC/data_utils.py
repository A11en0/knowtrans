import json
import os


def merge_json_files(directory, output_file):
    merged_data = []
    
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            
            # 读取每个JSON文件
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
                # 将数据添加到合并列表中
                if isinstance(data, list):
                    merged_data.extend(data)
                else:
                    merged_data.append(data)
    
    # 将合并后的数据写入输出文件
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(merged_data, outfile, ensure_ascii=False, indent=2)

    print(f"已将{len(merged_data)}个JSON对象合并到{output_file}")

# 使用示例
merge_json_files('data/random_sample', 'data/random_samples.json')