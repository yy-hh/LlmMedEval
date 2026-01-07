import requests
import json
import os
import glob

def eval(prompt):
    try:
        import requests
        import json

        api_key = os.environ.get("ANTMED_API_KEY")

        headers = {
            # Already added when you pass json= but not when you pass data=
            # 'Content-Type': 'application/json',
            'Authorization':f'Bearer {api_key}',
        }

        json_data = {
            'model': 'GPT-5.2',
            'messages': [
                {
                    'role': 'user',
                    'content': f"{prompt}",
                },
            ],
            'max_completion_tokens': 5000,
            'top_p': 1,
            'temperature': 0,
            'stream': False,
        }

        response = requests.post('https://freeland.openai.azure.com/openai/v1/chat/completions', headers=headers, json=json_data)
        data = json.loads(response.text)
        content_str = data['choices'][0]['message']['content']
        if "</think>" in content_str:
            parts = content_str.split("</think>", 1)
            content = parts[1].lstrip()
        else:
            content = content_str.strip()
        return content
    except Exception as err:
        print(err)

#旧数据集测试
#datasets = ['CHIP-CDEE', 'CHIP-CDN', 'CHIP-CTC', 'CMB-Clin', 'CMeEE', 'CMeIE', 'DBMHG', 'DDx-advanced', 'DDx-basic', 'DrugCA', 'IMCS-V2-MRG', 'Med-Exam', 'MedDG', 'MedHC', 'MedHG', 'MedMC', 'MedSafety', 'MedSpeQA', 'MedTreat', 'SMDoc']

input_dir = '/data/shenchengwei/LlmMedEval/new_test_data_gpt5.2/test2'
output_dir = '/data/shenchengwei/LlmMedEval/test_result_gpt5'
os.makedirs(output_dir, exist_ok=True)


for input_file_path in glob.glob(os.path.join(input_dir, '*.jsonl')):
    # 输入文件路径：测试集 jsonl
    file_name = os.path.basename(input_file_path)
    output_file_path = os.path.join(output_dir,file_name)
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
                content="注意：禁止输出json格式结果，直接输出文本\n"+question
                try:
                    #response = requests.request("POST", url, headers=headers, data=payload)
                    #response.raise_for_status()
                    #response.encoding = 'utf-8'
                    #answer = json.loads(response.text).get('choices')[0].get("message").get('content')
                    answer=eval(content)
                    print(answer)
                except Exception as e:
                    print(f"Request error: {e}")
                    #print(json.dumps(payload, ensure_ascii=False, indent=2))
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
