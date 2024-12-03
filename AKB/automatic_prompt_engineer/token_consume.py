import re

def extract_and_sum_numbers(file_path):
    input_sum = 0
    output_sum = 0

    with open(file_path, 'r') as file:
        for line in file:
            # 匹配 [INPUT TOKEN] 和 [OUTPUT TOKEN] 后面的数字
            input_match = re.search(r'\[INPUT TOKEN\]\s+(\d+)', line)
            output_match = re.search(r'\[OUTPUT TOKEN\]\s+(\d+)', line)

            if input_match:
                input_sum += int(input_match.group(1))
            if output_match:
                output_sum += int(output_match.group(1))

    print(f"Sum of numbers following [INPUT TOKEN]: {input_sum}")
    print(f"Sum of numbers following [OUTPUT TOKEN]: {output_sum}")

# 示例文件路径
file_path = 'experiments/results/Analysis/rayyan--train.log'
extract_and_sum_numbers(file_path)