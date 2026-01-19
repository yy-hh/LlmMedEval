import os
import json
import shutil

# --- 配置路径 ---
# 包含测试结果的文件夹 (jsonl)
result_folder = './test_result_Baichuan-M3'
# 对应的原始数据文件夹
source_data_folder = '/home/usst/scw/LlmMedEval/MedBench/new_test_data'
# 准备复制到的新文件夹
target_folder = '/home/usst/scw/LlmMedEval/MedBench/new_test_data1'


def process_and_copy():
    # 如果目标文件夹不存在，则创建
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        print(f"已创建目标文件夹: {target_folder}")

    print(f"{'结果文件名':<35} | {'null数量':<10} | {'操作状态'}")
    print("-" * 80)

    found_files_count = 0

    # 1. 遍历测试结果文件夹
    for filename in sorted(os.listdir(result_folder)):
        if filename.endswith('.jsonl'):
            result_file_path = os.path.join(result_folder, filename)
            has_null = False
            null_count = 0

            try:
                # 2. 检查结果文件中是否存在 "answer": null
                with open(result_file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip(): continue
                        data = json.loads(line)
                        if data.get("answer") is None and "answer" in data:
                            has_null = True
                            null_count += 1

                # 3. 如果发现 null，执行复制操作
                if has_null:
                    # 假设源文件名与结果文件名一致
                    source_file_path = os.path.join(source_data_folder, filename)
                    dest_file_path = os.path.join(target_folder, filename)

                    status = ""
                    if os.path.exists(source_file_path):
                        shutil.copy2(source_file_path, dest_file_path)  # copy2 保留元数据
                        status = "✅ 已复制源文件"
                        found_files_count += 1
                    else:
                        status = "❌ 未找到对应源文件"

                    print(f"{filename:<35} | {null_count:<10} | {status}")
                else:
                    # 如果没有 null，则跳过
                    pass

            except Exception as e:
                print(f"{filename:<35} | 出错       | {e}")

    print("-" * 80)
    print(f"处理完成！共提取并复制了 {found_files_count} 个文件到 {target_folder}")


if __name__ == "__main__":
    process_and_copy()