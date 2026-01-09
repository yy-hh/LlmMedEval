# 基于MedBench工具的医疗大模型评测与对比

本项目旨在利用MedBench工具，对各类医疗大模型（包括通用大模型）进行系统性评测，并提供透明、客观的性能对比。鉴于部分厂商可能存在“刷榜”行为，我们独立整理并展示了主流模型的评测结果，包括不同模型在每个评测集上的每个问题的实际回答结果。

## 关于MedBench

MedBench是一个由上海人工智能实验室开发的医疗领域大模型评测工具，广泛用于评估医疗大模型的各项能力。

---

## 一、评测结果概览

### 1.1 老版本评测 (2025年11月之前)

#### 评测数据集
![image](https://github.com/user-attachments/assets/8d00f882-77f0-43a9-a367-9d6d48b4a582)

#### 评测结果
| 模型               | 是否开源 | 综合得分 | 医学知识问答 | 医学语言生成 | 复杂医学推理 | 医学语言理解 | 医疗安全和伦理 |
| -------------------- | ---------- | ---------- | -------------- | -------------- | -------------- | -------------- | ---------------- |
| GPT4o              | 否       | 55.4     | 39.9         | 77.8         | 42.2         | 60.1         | 83.6           |
| baichun-m1-preview | 是       | 62.8     | 72.2         | 74.4         | 65.7         | 41.8         | 76             |
| Deepseek-r1-32B    | 是       | 67       | 78.5         | 71.9         | 60           | 57.9         | 71.2           |
| Deepseek-r1-70B    | 是       | 0        | 64           | 79.9         | 0            | 0            | 39.2           |
| GPT4-O1mini        | 否       | 64.6     | 75.3         | 81.6         | 66.7         | 56.7         | 52.1           |
| 某医疗垂直大模型   | 否       | 77.1     | 71           | 90.2         | 70.4         | 71.9         | 68.5           |

**注意：** Deepseek-r1-70b在指令遵循方面表现不佳，难以按照要求输出答案，因此不建议在需要严格指令遵循的场景中使用。

### 1.2 新版本评测 (2025年11月之后)

#### 评测数据集
<img width="100%" alt="{AA452347-5E43-4768-87FA-56D141AED41D}" src="https://github.com/user-attachments/assets/3f526caa-0bef-4a27-a856-d986f7965d96" />

#### 评测结果
| 模型               | 是否开源 | 综合得分 | 医学知识问答 | 医学语言生成 | 复杂医学推理 | 医学语言理解 | 医疗安全和伦理 |
| ----------------- | ------- | -------- | ------------ | ----------- | ----------- | ----------- | ------------- |
| AntAngelMed-FP8 | 是      | 55.8    | 66.8        | 67.7        | 58.0        | 60.2        | 26.7          |
| AntAngelMed-FP8 | 是      | 54.5    | 71.7        | 72.1        | 61.1        | 49.4        | 18.3          |

---

## 二、评测方法

1.  **准备模型：** 配置待评测的医疗大模型。
2.  **修改配置：** 从环境变量中获取模型 API 密钥、模型名称和 API URL。
3.  **设置输出：** 指定评测结果的输出目录。
4.  **测试数据确定：** 在 main文件的run_evaluation函数中选择确定使用旧数据集还是新数据集。
4.  **运行主程序：** 直接执行 `main.py` 文件启动评测。
5.  **提交结果：** 通过 MedBench 官方平台提交评测结果：[https://medbench.opencompass.org.cn/home](https://medbench.opencompass.org.cn/home)

## 三、参数设置

```
system prompt:
You are a helpful assistant.
temperature: 0
```

## 四、致谢

本项目衷心感谢 @MedBench 团队提供的强大评测框架。

```bibtex
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