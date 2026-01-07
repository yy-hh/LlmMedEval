import requests
import json
import os
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 配置参数
API_TOKEN = os.getenv('API_TOKEN', 'your_token')
MODEL_NAME = os.getenv('MODEL_NAME', 'your_model')
API_URL = os.getenv('API_URL', 'your_url')
OUTPUT_DIR = f'test_results/test_result_{MODEL_NAME}'

# 确保输出目录存在
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def eval(prompt):
    try:
        headers = {
            'Authorization': f'Bearer {API_TOKEN}',
        }

        json_data = {
            'model': MODEL_NAME,
            'messages': [
                {
                    'role': 'user',
                    'content': f"{prompt}",
                },
            ]
        }

        response = requests.post(API_URL, headers=headers, json=json_data)
        response.raise_for_status()  # 检查HTTP请求是否成功
        
        response_json = response.json()
        rst = response_json['choices'][0]['message']['content'].split("</think>")
        
        if len(rst) > 1:
            think = rst[0].replace("\n", "")
            content = rst[1][2:]
            logging.debug(f'思考过程：{think}')
            logging.debug(f'结果：{content}')
            return content
        else:
            logging.warning(f"API响应格式不符合预期: {response_json['choices'][0]['message']['content']}")
            return response_json['choices'][0]['message']['content'] # 返回原始内容
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

def run_evaluation():
    test_dataset_dir = Path('test_dataset')
    if not test_dataset_dir.is_dir():
        logging.error(f"测试数据集目录 '{test_dataset_dir}' 不存在。请确保该目录存在并包含数据集。")
        return

    datasets = [name.name for name in test_dataset_dir.iterdir() if name.is_dir()]
    if not datasets:
        logging.warning(f"在 '{test_dataset_dir}' 目录中未找到任何数据集。")
        return

    for dataset in datasets:
        input_file_path = test_dataset_dir / dataset / f'{dataset}_test.jsonl'
        output_file_path = Path(OUTPUT_DIR) / f'{dataset}_test.jsonl'

        if not input_file_path.is_file():
            logging.warning(f"数据集 '{dataset}' 的输入文件 '{input_file_path}' 不存在，跳过。")
            continue

        logging.info(f"开始处理数据集: {dataset}")
        try:
            with open(input_file_path, 'r', encoding='utf-8') as f_in, \
                    open(output_file_path, 'w', encoding='utf-8') as f_out:
                for line_num, line in enumerate(f_in, 1):
                    try:
                        content_data = json.loads(line)
                    except json.JSONDecodeError as e:
                        logging.error(f"文件 '{input_file_path}' 第 {line_num} 行JSON解析错误: {e}")
                        continue

                    question = content_data.get('question')
                    options = content_data.get('options')
                    other = content_data.get('other')
                    
                    if not question:
                        logging.warning(f"文件 '{input_file_path}' 第 {line_num} 行缺少 'question' 字段，跳过。")
                        continue

                    prompt_content = "注意：禁止输出json格式结果，直接输出文本/n" + question
                    answer = eval(prompt_content)

                    if answer is not None:
                        logging.info(f"数据集: {dataset}, 问题: {question[:50]}..., 回答: {answer[:50]}...")
                    else:
                        logging.warning(f"数据集: {dataset}, 问题: {question[:50]}..., 未能获取回答。")

                    output_content = {
                        "question": question,
                        "answer": answer,
                        "other": other,
                        "options": options
                    }
                    f_out.write(json.dumps(output_content, ensure_ascii=False) + '\n')
        except IOError as e:
            logging.error(f"处理文件 '{input_file_path}' 或 '{output_file_path}' 时发生IO错误: {e}")
        except Exception as e:
            logging.error(f"处理数据集 '{dataset}' 时发生未知错误: {e}")
    logging.info("所有数据集处理完成。")

if __name__ == "__main__":
    run_evaluation()
