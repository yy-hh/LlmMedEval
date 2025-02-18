import requests
import json
import os

def get_subdirectories(directory):
    return [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
def remove_think_content(text):
    # Split the text at the '</think>' tag
    parts = text.split('</think>')
    # Strip leading whitespace from the part after the tag
    return parts[-1].lstrip()

# 获取 test_data 目录下的所有子文件夹名称
directory_path = 'test_data'
datasets = get_subdirectories(directory_path)

url = "https://dify-srv02.weicha88.com/v1/chat-messages"
headers = {
    'Authorization': 'Bearer app-9PKbpgfpkXsMLg2uV6th9mvn',
}


# data = {
#     "inputs": {},
#     "query": "任务：给定病历或者医学影像报告，要求从中抽取临床发现事件的四个属性:主体词、解剖部位、描述词、发生状态。\n主体词：指患者的电子病历中的疾病名称或者由疾病引发的症状，也包括患者的一般情况如饮食，二便，睡眠等。\n描述词：对主体词的发生时序特征、轻重程度、形态颜色等多个维度的刻画，也包括疾病的起病缓急、突发。\n解剖部位：指主体词发生在患者的身体部位，也包括组织，细胞，系统等，也包括部位的方向和数量。\n发生状态：“不确定”或“否定”，肯定的情况不标注发生状态。\n\n任务示例如下：\n\n任务示例1：\n输入报告：\n遂入住我院行“经腹广泛性全子宫切除术+双侧附件切除术+盆腔淋巴结清扫术”，手术顺利，术后病理诊断：1、宫内膜中分化宫内膜样腺癌，浸及肌层2/3，脉管中见癌栓，神经未见癌侵犯，阴道壁残端，宫颈、颈管未见癌累及，双侧附件未见癌转移。\n输出事件：\n[{'主体词': '癌', '发生状态': '', '描述词': ['浸及'], '解剖部位': ['肌层2/3']}, {'主体词': '癌栓', '发生状态': '', '描述词': [], '解剖部位': ['脉管']}, {'主体词': '癌', '发生状态': '否定', '描述词': ['侵犯'], '解剖部位': ['神经']}, {'主体词': '腺癌', '发生状态': '', '描述词': ['中分化', '内膜样', '浸及肌层2/3'], '解剖部位': ['宫内膜']}, {'主体词': '癌', '发生状态': '否定', '描述词': ['累及'], '解剖部位': ['阴道壁残端', '宫颈', '宫颈管']}, {'主体词': '癌', '发生状态': '否定', '描述词': ['转移'], '解剖部位': ['双侧附件']}]\n\n任务示例2：\n输入报告：\n患者遂至我院就诊，查彩超（123456）示：膀胱实性占位（考虑膀胱ca），建议进一步检查。门诊医生建议住院治疗，患者拒绝。\n输出事件：\n[{'主体词': '占位', '发生状态': '', '描述词': ['实性'], '解剖部位': ['膀胱']}, {'主体词': '癌', '发生状态': '不确定', '描述词': [], '解剖部位': ['膀胱']}]\n\n任务示例3：\n输入报告：\n今为行ir间期治疗第2次静脉化疗入院，门诊血常规（2017.01.09）提示wbc 1.98(*10^9/L)，plt 217(*10^9/L)，rbc 2.54(*10^12/L)，hb 80(g/l)，anc 0.83(*10^9/L)；门诊以“急性普通b淋巴细胞淋巴细胞白血病l11型（中危，缓解期）”收入院。\n输出事件：\n[{'主体词': '白血病', '发生状态': '', '描述词': ['急性', '普通', 'l11型', '中危', '缓解期'], '解剖部位': ['b淋巴细胞']}]\n\n任务示例4：\n输入报告：\n患者自发病以来，精神可，食欲可，睡眠可，大便如常，小便如常，体重无明显变化。\n输出事件：\n[{'主体词': '精神', '发生状态': '', '描述词': ['可'], '解剖部位': []}, {'主体词': '食欲', '发生状态': '', '描述词': ['可'], '解剖部位': []}, {'主体词': '睡眠', '发生状态': '', '描述词': ['可'], '解剖部位': []}, {'主体词': '大便', '发生状态': '', '描述词': ['如常'], '解剖部位': []}, {'主体词': '小便', '发生状态': '', '描述词': ['如常'], '解剖部位': []}, {'主体词': '体重改变', '发生状态': '否定', '描述词': ['明显'], '解剖部位': []}]\n\n任务示例5：\n输入报告：\n患者无腹痛、腹胀，无肝区疼痛等不适。\n输出事件：\n[{'主体词': '疼痛', '发生状态': '否定', '描述词': [], '解剖部位': ['腹']}, {'主体词': '腹胀', '发生状态': '否定', '描述词': [], '解剖部位': ['腹']}, {'主体词': '疼痛', '发生状态': '否定', '描述词': [], '解剖部位': ['肝区']}]\n\n\n输入报告：\n因患者需期末考试，故予以口服“雷贝拉唑钠肠溶片”治疗，现腹痛情况明显好转。\n输出事件：\n",
#     "response_mode": "blocking",
#     "conversation_id": "",
#     "user": "test-gpt4o1",
#     "files": []
# }

## 调试
# try:
#     response = requests.post(url, headers=headers, json=data, timeout=1000)
#     response.raise_for_status()
#     response.encoding = 'utf-8'
#     # 解析JSON数据
#     response_data = response.json()
#     print(response_data)
#
# except requests.exceptions.HTTPError as http_err:
#     print(f'HTTP错误: {http_err}')
# except requests.exceptions.RequestException as err:
#     print(f'请求错误: {err}')
# except ValueError as json_err:
#     print(f'JSON解析错误: {json_err}')
#
for dataset in datasets:
    input_file_path = f'test_data/{dataset}/{dataset}_test.jsonl'
    output_file_path = f'test_result_deepseek-32b/{dataset}_test.jsonl'

    try:
        with open(input_file_path, 'r', encoding='utf-8') as f, \
                open(output_file_path, 'w', encoding='utf-8') as output_file:
            for line in f:
                try:
                    content = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"JSON decode error in line: {e}")
                    continue

                question = content.get('question')
                options = content.get('options')
                other = content.get('other')
                answer= None
                data = {
                    "inputs": '',
                    "query": question,
                    "response_mode": "blocking",
                    "conversation_id": "",
                    "user": "test-gpt4o1",
                    "files": []
                }

                max_retries = 3
                retry_count = 0
                while retry_count <= max_retries:
                    try:
                        response = requests.post(url, headers=headers, json=data, timeout=1000)
                        response.raise_for_status()
                        response.encoding = 'utf-8'
                        answer = json.loads(response.text).get('answer', '')
                        answer=cleaned_text = remove_think_content(answer)
                        print(answer)
                        break  # 如果请求成功，退出循环
                    except requests.exceptions.RequestException as e:
                        print(f"Request error: {e}")
                        print(json.dumps(data, ensure_ascii=False, indent=2))
                        retry_count += 1
                        if retry_count > max_retries:
                            answer = None
                            break

                print("question:", question)
                print("answer:", answer)
                print("other:", other)
                print("options:", options)

                output_content = {
                    "question": question,
                    "answer": answer,
                    "other": other,
                    "options": options
                }
                output_file.write(json.dumps(output_content, ensure_ascii=False) + '\n')

    except FileNotFoundError as e:
        print(f"File not found: {e}")
