# 基于MedBench工具测试本地医疗大模型
## 评测数据集
![image](https://github.com/user-attachments/assets/8d00f882-77f0-43a9-a367-9d6d48b4a582)


## 评测结果

| 模型               | 是否开源 | 综合得分 | 医学知识问答 | 医学语言生成 | 复杂医学推理 | 医学语言理解 | 医疗安全和伦理 |
| -------------------- | ---------- | ---------- | -------------- | -------------- | -------------- | -------------- | ---------------- |
| GPT4o              | 否       | 55.4     | 39.9         | 77.8         | 42.2         | 60.1         | 83.6           |
| zuoyi              | 否       | 79       | 89.3         | 72.1         | 71.9         | 82.5         | 82             |
| baichun-m1-preview | 是       | 62.8     | 72.2         | 74.4         | 65.7         | 41.8         | 76             |
| Deepseek-r1-32B    | 是       | 67       | 78.5         | 71.9         | 60           | 57.9         | 71.2           |
| 持续更新中   |          |          |              |              |              |              |                |

## 感谢MedBench提供测试数据

@article{MedBench, 
author = {Mianxin Liu and Weiguo Hu and Jinru Ding and Jie Xu and Xiaoyang Li and Lifeng Zhu and Zhian Bai and Xiaoming Shi and Benyou Wang and Haitao Song and Pengfei Liu and Xiaofan Zhang and Shanshan Wang and Kang Li and Haofen Wang and Tong Ruan and Xuanjing Huang and Xin Sun and Shaoting Zhang},
title = {MedBench: A Comprehensive, Standardized, and Reliable Benchmarking System for Evaluating Chinese Medical Large Language Models},
year = {2024},
journal = {Big Data Mining and Analytics},
url = {https://www.sciopen.com/article/10.26599/BDMA.2024.9020044},
doi = {10.26599/BDMA.2024.9020044}
}
