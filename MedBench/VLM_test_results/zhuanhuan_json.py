# 


import os
import json

def reorder_and_clean_jsonl(directory):
    # 定义新的目标字段顺序（去掉了 options）
    target_order = ["question", "answer", "img_path", "other"]
    
    if not os.path.exists(directory):
        print(f"错误: 找不到目录 {directory}")
        return

    for filename in os.listdir(directory):
        if filename.endswith(".jsonl"):
            file_path = os.path.join(directory, filename)
            temp_data = []

            print(f"正在处理: {filename}")

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        
                        original_dict = json.loads(line)
                        
                        # 创建新字典，仅包含 target_order 中的字段
                        reordered_dict = {}
                        for key in target_order:
                            if key in original_dict:
                                reordered_dict[key] = original_dict[key]
                        
                        temp_data.append(reordered_dict)

                # 写回文件
                with open(file_path, 'w', encoding='utf-8') as f:
                    for item in temp_data:
                        # ensure_ascii=False 确保中文正常显示
                        json_line = json.dumps(item, ensure_ascii=False)
                        f.write(json_line + "\n")

            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")

    print("--- 处理完成：字段已重排并删除了 options ---")

if __name__ == "__main__":
    # 你的目标路径
    target_dir = r"D:\xiazai\xiangmu\git\LlmMedEval\MedBench\VLM_test_results\test_result_gpt-5.2\MedBench_VLM_QA"
    reorder_and_clean_jsonl(target_dir)