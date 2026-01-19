import requests
import json
import os
import logging
from pathlib import Path
import base64, mimetypes
# 配置日志1
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 配置参数
API_TOKEN = os.getenv('API_TOKEN', '')
MODEL_NAME = os.getenv('MODEL_NAME', '')
API_URL = os.getenv('API_URL', '')

def to_data_url(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def eval(question, image_path: str | None = None):
    content = [{"type": "text", "text": question}]
    if image_path:
        data_url = to_data_url(image_path)
        content.append({"type": "image_url", "image_url": {"url": data_url}})
    try:
        headers = {
            'Authorization': f'Bearer {API_TOKEN}',
        }
        
        json_data = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": "you are a helpful assistant."},
                {"role": "user", "content": content},
            ],
        }

        response = requests.post(API_URL, headers=headers, json=json_data)
        response.raise_for_status()  # 检查HTTP请求是否成功
        
        # 解析返回结果，取出 content 字段，并按 </think> 切分思考过程和最终答案
        data=json.loads(response.text)
        content_str= data['choices'][0]['message']['content']

        if "</think>" in content_str:
            parts = content_str.split("</think>", 1)
            content = parts[1].lstrip()
        else:
            content = content_str.strip()
        return content

    except requests.exceptions.RequestException as e:
        logging.error(f"API请求错误: {e}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"API响应JSON解析错误: {e}")
        return None
    except KeyError as e:
        logging.error(f"API响应缺少关键字段: {e}, 响应内容: {response.text}")
        return None
    except Exception as e:
        logging.error(f"eval函数发生未知错误: {e}")
        return None

def process_old_test_datasets():
    #新数据集
    OUTPUT_DIR = f'test_results/test_result_{MODEL_NAME}'
    # 确保输出目录存在
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    test_dataset_dir = Path('test_dataset')
    if not test_dataset_dir.is_dir():
        logging.warning(f"旧测试数据集目录 '{test_dataset_dir}' 不存在，跳过。")
        return

    datasets = [name.name for name in test_dataset_dir.iterdir() if name.is_dir()]
    if not datasets:
        logging.warning(f"在旧测试数据集目录 '{test_dataset_dir}' 中未找到任何数据集。")
        return

    for dataset in datasets:
        input_file_path = test_dataset_dir / dataset / f'{dataset}_test.jsonl'
        output_file_path = Path(OUTPUT_DIR) / f'{dataset}_test.jsonl'

        if not input_file_path.is_file():
            logging.warning(f"旧数据集 '{dataset}' 的输入文件 '{input_file_path}' 不存在，跳过。")
            continue

        logging.info(f"开始处理旧数据集: {dataset}")
        try:
            with open(input_file_path, 'r', encoding='utf-8') as f_in, \
                    open(output_file_path, 'w', encoding='utf-8') as f_out:
                for line_num, line in enumerate(f_in, 1):
                    try:
                        content_data = json.loads(line)
                    except json.JSONDecodeError as e:
                        logging.error(f"旧数据集文件 '{input_file_path}' 第 {line_num} 行JSON解析错误: {e}")
                        continue

                    question = content_data.get('question')
                    options = content_data.get('options')
                    other = content_data.get('other')

                    if not question:
                        logging.warning(f"旧数据集文件 '{input_file_path}' 第 {line_num} 行缺少 'question' 字段，跳过。")
                        continue

                    prompt_content = "注意：禁止输出json格式结果，直接输出文本\n" + question
                    answer = eval(prompt_content)

                    if answer is not None:
                        logging.info(f"旧数据集: {dataset}, 问题: {question[:50]}..., 回答: {answer[:50]}...")
                    else:
                        logging.warning(f"旧数据集: {dataset}, 问题: {question[:50]}..., 未能获取回答。")

                    output_content = {
                        "question": question,
                        "answer": answer,
                        "other": other,
                        "options": options
                    }
                    f_out.write(json.dumps(output_content, ensure_ascii=False) + '\n')
        except IOError as e:
            logging.error(f"处理旧数据集文件 '{input_file_path}' 或 '{output_file_path}' 时发生IO错误: {e}")
        except Exception as e:
            logging.error(f"处理旧数据集 '{dataset}' 时发生未知错误: {e}")

def process_new_test_datasets():
    #新数据集
    OUTPUT_DIR = f'new_test_results/test_result_{MODEL_NAME}'
    # 确保输出目录存在
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    new_test_data_dir = Path('new_test_data')
    if not new_test_data_dir.is_dir():
        logging.warning(f"新测试数据集目录 '{new_test_data_dir}' 不存在，跳过。")
        return

    files = [p for p in new_test_data_dir.iterdir() if p.is_file() and p.suffix == '.jsonl']
    if not files:
        logging.warning(f"在新测试数据集目录 '{new_test_data_dir}' 中未找到任何 jsonl 文件。")
        return

    for input_file_path in files:
        dataset = input_file_path.stem
        output_file_path = Path(OUTPUT_DIR) / f'{dataset}.jsonl'

        logging.info(f"开始处理新数据集: {dataset}")
        try:
            with open(input_file_path, 'r', encoding='utf-8') as f_in, \
                    open(output_file_path, 'w', encoding='utf-8') as f_out:
                for line_num, line in enumerate(f_in, 1):
                    try:
                        content_data = json.loads(line)
                    except json.JSONDecodeError as e:
                        logging.error(f"新数据集文件 '{input_file_path}' 第 {line_num} 行JSON解析错误: {e}")
                        continue

                    question = content_data.get('question')
                    options = content_data.get('options')
                    other = content_data.get('other')

                    if not question:
                        logging.warning(f"新数据集文件 '{input_file_path}' 第 {line_num} 行缺少 'question' 字段，跳过。")
                        continue

                    prompt_content = question
                    answer = eval(prompt_content)

                    if answer is not None:
                        logging.info(f"新数据集: {dataset}, 问题: {question[:50]}..., 回答: {answer[:50]}...")
                    else:
                        logging.warning(f"新数据集: {dataset}, 问题: {question[:50]}..., 未能获取回答。")

                    output_content = {
                        "question": question,
                        "answer": answer,
                        "other": other,
                        "options": options
                    }
                    f_out.write(json.dumps(output_content, ensure_ascii=False) + '\n')
        except IOError as e:
            logging.error(f"处理新数据集文件 '{input_file_path}' 或 '{output_file_path}' 时发生IO错误: {e}")
        except Exception as e:
            logging.error(f"处理新数据集 '{dataset}' 时发生未知错误: {e}")

def process_VLM_datasets():
     #新数据集
    OUTPUT_DIR = f'VLM_test_results/test_result_{MODEL_NAME}'
    # 确保输出目录存在
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    vlm_test_data_dir = Path('VLM_test_data')
    if not vlm_test_data_dir.is_dir():
        logging.warning(f"VLM测试数据集目录 '{vlm_test_data_dir}' 不存在，跳过。")
        return

    files = [p for p in vlm_test_data_dir.iterdir() if p.is_file() and p.suffix == '.jsonl']
    if not files:
        logging.warning(f"在VLMP测试数据集目录 '{vlm_test_data_dir}' 中未找到任何 jsonl 文件。")
        return

    for input_file_path in files:
        dataset = input_file_path.stem
        output_file_path = Path(OUTPUT_DIR) / f'{dataset}.jsonl'

        logging.info(f"开始处理VLMP数据集: {dataset}")
        try:
            with open(input_file_path, 'r', encoding='utf-8') as f_in, \
                    open(output_file_path, 'w', encoding='utf-8') as f_out:
                for line_num, line in enumerate(f_in, 1):
                    try:
                        content_data = json.loads(line)
                    except json.JSONDecodeError as e:
                        logging.error(f"新数据集文件 '{input_file_path}' 第 {line_num} 行JSON解析错误: {e}")
                        continue

                    question = content_data.get('question')
                    options = content_data.get('options')
                    img_list = content_data.get('img_path')
                    if not img_list or not isinstance(img_list, list):
                        logging.warning(f"VLMP数据集文件 '{input_file_path}' 第 {line_num} 行缺少或格式错误的 'img_path' 字段，跳过。")
                        continue
                    #img_path = f"{vlm_test_data_dir}/{img_list[0]}"
                    img_path = [
                        str(vlm_test_data_dir / img) for img in img_list
                    ]
                    other = content_data.get('other')

                    if not question:
                        logging.warning(f"新数据集文件 '{input_file_path}' 第 {line_num} 行缺少 'question' 字段，跳过。")
                        continue

                    # prompt_content = "注意：禁止输出json格式结果，直接输出文本\n" + question
                    answer = eval(question,img_path)
                    print(answer)
                    # if answer is not None:
                    #     logging.info(f"新数据集: {dataset}, 问题: {question[:50]}..., 回答: {answer[:50]}...")
                    # else:
                    #     logging.warning(f"新数据集: {dataset}, 问题: {question[:50]}..., 未能获取回答。")
                    
                    output_content = {
                        "question": question,
                        "answer": answer,
                        "other": other,
                        "img_path": content_data.get('img_path'),
                        "options": options
                    }
                    f_out.write(json.dumps(output_content, ensure_ascii=False) + '\n')
        except IOError as e:
            logging.error(f"处理新数据集文件 '{input_file_path}' 或 '{output_file_path}' 时发生IO错误: {e}")
        except Exception as e:
            logging.error(f"处理新数据集 '{dataset}' 时发生未知错误: {e}")

def run_evaluation():
    #process_old_test_datasets()
    #process_new_test_datasets()
    process_VLM_datasets()
    return


if __name__ == "__main__":
    run_evaluation()
