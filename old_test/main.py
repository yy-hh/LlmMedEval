import requests
import json
import os

def eval(prompt):
    try:
        import requests
        import json

        headers = {
            # Already added when you pass json= but not when you pass data=
            # 'Content-Type': 'application/json',
            'Authorization':'Bearer sz45c1fcbac46f4161bc1c68fadc040a3a',
        }

        json_data = {
            'model': 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
            'messages': [
                {
                    'role': 'user',
                    'content': f"{prompt}",
                },
            ],
            'max_tokens': 5000,
            'top_k': -1,
            'top_p': 1,
            'temperature': 0,
            'ignore_eos': False,
            'stream': False,
        }

        response = requests.post('http://192.168.201.55:55015/v1/chat/completions', headers=headers, json=json_data)
        rst=json.loads(response.text)['choices'][0]['message']['content'].split("</think>")
        think=rst[0]
        content=rst[1][2:]
        #print(f'思考过程：{think}')
        #print(f'结果：{content}')
        return content
    except Exception as err:
        print(err)
datasets = ['CHIP-CDEE', 'CHIP-CDN', 'CHIP-CTC', 'CMB-Clin', 'CMeEE', 'CMeIE', 'DBMHG', 'DDx-advanced', 'DDx-basic', 'DrugCA', 'IMCS-V2-MRG', 'Med-Exam', 'MedDG', 'MedHC', 'MedHG', 'MedMC', 'MedSafety', 'MedSpeQA', 'MedTreat', 'SMDoc']


for dataset in datasets:
    input_file_path = f'old_test/test_data/{dataset}/{dataset}_test.jsonl'
    output_file_path = f'old_test/test_result_deepseek-70b/{dataset}_test.jsonl'

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
                content="注意：禁止输出json格式结果，直接输出文本/n"+question
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





