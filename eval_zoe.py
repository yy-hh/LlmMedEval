import requests
import json
import os

def get_subdirectories(directory):
    return [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]

# 获取 test_data 目录下的所有子文件夹名称
directory_path = 'test_data'
# datasets = ['CHIP-CDEE', 'CHIP-CDN', 'CHIP-CTC', 'CMB-Clin', 'CMeEE', 'CMeIE', 'DBMHG', 'DDx-advanced', 'DDx-basic', 'DrugCA', 'IMCS-V2-MRG', 'Med-Exam', 'MedDG', 'MedHC', 'MedHG', 'MedMC', 'MedSafety', 'MedSpeQA', 'MedTreat', 'SMDoc']
datasets = ['MedSafety', 'MedSpeQA', 'MedTreat', 'SMDoc']

url = "https://openapi.zuoshouyisheng.com/gpt-pro/v2/chat/completions"
headers = {
  'Accept': 'application/json',
  'Content-Type': 'application/json',
  'X-API-KEY': 'ZOE-c3808dc6-85f7-4d4a-8843-0da4da3be720'
}
# payload = json.dumps({
#     "model": "zoe-gpt-pro",
#     "messages": [
#         {
#             "role": "user",
#             "content": "糖尿病患者饮食方面要注意什么"
#         }
#     ],
#     "max_tokens": 1024,
#     "stream": False,
#     "temperature": None,
#     "top_p": None,
#     "repetition_penalty": None,
#     "user": "user-1234"
# })
# response = requests.request("POST", url, headers=headers, data=payload)
# response.raise_for_status()
# response.encoding = 'utf-8'
# answer = json.loads(response.text).get('choices')[0].get("message").get('content')
# print(answer)

for dataset in datasets:
    input_file_path = f'test_data/{dataset}/{dataset}_test.jsonl'
    output_file_path = f'test_result_zuoyi/{dataset}_test.jsonl'

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
                payload = json.dumps({
                    "model": "zoe-gpt-pro",
                    "messages": [
                        {
                            "role": "user",
                            "content": "注意：禁止输出json格式结果，直接输出文本/n"+question
                        }
                    ],
                    "max_tokens": 1024,
                    "stream": False,
                    "temperature": None,
                    "top_p": None,
                    "repetition_penalty": None,
                    "user": "user-1234"
                })

                try:
                    response = requests.request("POST", url, headers=headers, data=payload)
                    response.raise_for_status()
                    response.encoding = 'utf-8'
                    answer = json.loads(response.text).get('choices')[0].get("message").get('content')
                    print(answer)
                except requests.exceptions.RequestException as e:
                    print(f"Request error: {e}")
                    print(json.dumps(payload, ensure_ascii=False, indent=2))
                    answer = None

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
