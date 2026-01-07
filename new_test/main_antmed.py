import requests  # 用于发送 HTTP 请求
import json      # 用于 JSON 编解码
import os        # 目前未使用，可用于路径操作等
import glob      # 用于文件路径匹配

def eval(prompt):
    try:
        import requests  # 再次导入 requests（冗余，但不影响）
        import json      # 再次导入 json（冗余）

        headers = {
            # 使用 json= 参数时，requests 会自动加 Content-Type: application/json
            # 'Content-Type': 'application/json',
            'Authorization': 'Bearer sz45c1fcbac46f4161bc1c68fadc040a3a',  # 鉴权用的 Bearer token
        }

        # 构造请求体，遵循 chat/completions 风格
        json_data = {
            'model': 'AntAngelMed-FP8',  # 使用的模型名称
            'messages': [
                {
                    'role': 'user',      # 角色为用户
                    'content': f"{prompt}",  # 用户输入的内容
                },
            ],
            'max_tokens': 5000,   # 最多生成的 token 数
            'top_k': -1,          # top-k 采样设置（-1 一般表示不限制）
            'top_p': 1,           # top-p 采样阈值
            'temperature': 0,     # 温度为 0，趋向确定性输出
            'ignore_eos': False,  # 不忽略 EOS，遇到结束符可以停止
            'stream': False,      # 不使用流式输出，一次性返回
        }

        # 向本地/局域网的大模型服务发送 POST 请求
        response = requests.post(
            'http://192.168.201.55:1217/v1/chat/completions',
            headers=headers,
            json=json_data
        )

        # 解析返回结果，取出 content 字段，并按 </think> 切分思考过程和最终答案
        data=json.loads(response.text)
        content_str= data['choices'][0]['message']['content']
        if "</think>" in content_str:
            parts = content_str.split("</think>", 1)
            content = parts[1].lstrip()
        else:
            content = content_str.strip()

        return content  # 返回回答文本
    except Exception as err:
        # 任何异常（网络错误、解析错误等）都打印出来
        print(err)


# 旧的评测的一系列数据集名称
# datasets = [
#     'CHIP-CDEE', 'CHIP-CDN', 'CHIP-CTC', 'CMB-Clin', 'CMeEE', 'CMeIE',
#     'DBMHG', 'DDx-advanced', 'DDx-basic', 'DrugCA', 'IMCS-V2-MRG',
#     'Med-Exam', 'MedDG', 'MedHC', 'MedHG', 'MedMC', 'MedSafety',
#     'MedSpeQA', 'MedTreat', 'SMDoc'
# ]

#新数据集评测
input_dir = '/data/shenchengwei/LlmMedEval/new_test_data'
output_dir = '/data/shenchengwei/LlmMedEval/test_result_antangelmed-fp8'
os.makedirs(output_dir, exist_ok=True)
# 依次处理每一个数据集
for input_file_path in glob.glob(os.path.join(input_dir, '*.jsonl')):
    # 输入文件路径：测试集 jsonl
    file_name = os.path.basename(input_file_path)
    output_file_path = os.path.join(output_dir,file_name)
  
    try:
        # 同时打开输入和输出文件
        with open(input_file_path, 'r', encoding='utf-8') as f, \
                open(output_file_path, 'w', encoding='utf-8') as output_file:
            # 按行读取输入文件，每一行一个样本
            for line in f:
                try:
                    # 将一行解析为 JSON 字典
                    content = json.loads(line)
                except json.JSONDecodeError as e:
                    # 如果该行 JSON 解析失败，打印错误并跳过该行
                    print(f"JSON decode error in line: {e}")
                    continue

                # 从样本中取出问题、选项和其他信息
                question = content.get('question')
                options = content.get('options')
                other = content.get('other')
                answer = None  # 初始化答案为 None

                # 构造发送给大模型的 prompt
                # 注意：这里 '/n' 是普通字符串，不是换行符，如果想真正换行应写成 '\n'
                content = "注意：禁止输出json格式结果，直接输出文本/n" + question

                try:
                    # 旧的请求代码已被注释
                    # response = requests.request("POST", url, headers=headers, data=payload)
                    # response.raise_for_status()
                    # response.encoding = 'utf-8'
                    # answer = json.loads(response.text).get('choices')[0].get("message").get('content')

                    # 调用上面定义的 eval 函数向模型发起请求
                    answer = eval(content)
                    print(answer)  # 打印模型回答，便于调试
                except Exception as e:
                    # 请求过程中的异常（网络、接口等）在此捕获
                    print(f"Request error: {e}")
                    # print(json.dumps(payload, ensure_ascii=False, indent=2))
                    answer = None  # 出错时答案设为 None

                # 打印当前样本的信息
                print("question:", question)
                print("answer:", answer)
                print("other:", other)
                print("options:", options)

                # 构造输出字典
                output_content = {
                    "question": question,
                    "answer": answer,
                    "other": other,
                    "options": options
                }

                # 写入一行 jsonl，保持中文可读
                output_file.write(json.dumps(output_content, ensure_ascii=False) + '\n')

    except FileNotFoundError as e:
        # 如果某个数据集的输入文件不存在，打印错误，不影响其他数据集
        print(f"File not found: {e}")