
import requests
import json
# 读取JSON数据
with open('report/2025-01-17to01-24.json', 'r', encoding='utf-8') as f:
    upload_file_data = json.load(f)['data']

url = "https://dify-srv.weicha88.com/v1/chat-messages"
headers = {
    'Authorization': 'Bearer app-m1KeB5g6LbP0ePFXpQ18flbU',
}
data = {
    "inputs": {},
    "query": "1",
    "response_mode": "blocking",
    "conversation_id": "",
    "user": "test-gpt4o",
    "files": []
}
response = requests.post(url, headers=headers, json=data)
print(response)

