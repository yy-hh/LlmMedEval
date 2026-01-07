# 基于MedBench工具测试不同医疗大模型（含通用）的表现

# 一、老版本 （2025年11月之前）

## 评测数据集
![image](https://github.com/user-attachments/assets/8d00f882-77f0-43a9-a367-9d6d48b4a582)
## 评测结果
| 模型               | 是否开源 | 综合得分 | 医学知识问答 | 医学语言生成 | 复杂医学推理 | 医学语言理解 | 医疗安全和伦理 |
| -------------------- | ---------- | ---------- | -------------- | -------------- | -------------- | -------------- | ---------------- |
| GPT4o              | 否       | 55.4     | 39.9         | 77.8         | 42.2         | 60.1         | 83.6           |
| baichun-m1-preview | 是       | 62.8     | 72.2         | 74.4         | 65.7         | 41.8         | 76             |
| Deepseek-r1-32B    | 是       | 67       | 78.5         | 71.9         | 60           | 57.9         | 71.2           |
|  Deepseek-r1-70B  |   是       |   0       |    64          |      79.9        |       0       |      0        |           39.2     |
|  GPT4-O1mini  | 否 |   64.6     |    75.3      |       81.6       |   66.7         |    56.7        |      52.1       |       
|  某医疗垂直大模型 | 否 |  77.1    |71|    90.2    |    70.4     |  71.9        |    68.5      |     92.2     |   82|  

（Deepseek-r1-70b指令跟随能力很差，基本无法按照要求输出答案，不推荐使用)

# 二、新版本 （2025年11月之后）

## 评测数据集
<img width="100%" alt="{AA452347-5E43-4768-87FA-56D141AED41D}" src="https://github.com/user-attachments/assets/3f526caa-0bef-4a27-a856-d986f7965d96" />

## 评测结果
| 模型               | 是否开源 | 综合得分 | 医学知识问答 | 医学语言生成 | 复杂医学推理 | 医学语言理解 | 医疗安全和伦理 |
| ----------------- | ------- | -------- | ------------ | ----------- | ----------- | ------------ | ------------- |
| AntAngelMed-FP8 | 是      | 	54.9  |      66.8   |   67.7      |	  58.0      |   60.2           |  21.7         | 	

# 三、评测方法

1. 准备要评测的模型
2. 修改eval.py中模型api和key 
3. 修改结果输出目录
4. 直接运行main.py文件
5. 通过MedBench提交 https://medbench.opencompass.org.cn/home


## 四、参数设置
```
system prompt:
You are a helpful assistant.
temperature: 0
```

## 四、感谢@MedBench提供测试框架
```
@misc{ding2025medbenchv4robustscalable,
      title={MedBench v4: A Robust and Scalable Benchmark for Evaluating Chinese Medical Language Models, Multimodal Models, and Intelligent Agents}, 
      author={Jinru Ding and Lu Lu and Chao Ding and Mouxiao Bian and Jiayuan Chen and Wenrao Pang and Ruiyao Chen and Xinwei Peng and Renjie Lu and Sijie Ren and Guanxu Zhu and Xiaoqin Wu and Zhiqiang Liu and Rongzhao Zhang and Luyi Jiang and Bing Han and Yunqiu Wang and Jie Xu},
      year={2025},
      eprint={2511.14439},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2511.14439}
}
```
