# 医疗大模型评测与对比

本项目旨在利用不同的评测框架，对各类医疗大模型（包括通用大模型）进行系统性评测，并提供透明、客观的性能对比。鉴于部分厂商可能存在“刷榜”行为，我们独立整理并展示了主流模型的评测结果，包括不同模型在每个评测集上的每个问题的实际回答结果。

## MedBench

MedBench是一个由上海人工智能实验室开发的医疗领域大模型评测工具，广泛用于评估医疗大模型的各项能力。

---

### 一、评测结果概览

#### 1.1 老版本LLM评测 (2025年11月之前)

##### 评测数据集
![image](https://github.com/user-attachments/assets/8d00f882-77f0-43a9-a367-9d6d48b4a582)

##### 评测结果
| 模型               | 是否开源 | 综合得分 | 医学知识问答 | 医学语言生成 | 复杂医学推理 | 医学语言理解 | 医疗安全和伦理 |
| -------------------- | ---------- | ---------- | -------------- | -------------- | -------------- | -------------- | ---------------- |
| GPT4o              | 否       | 55.4     | 39.9         | 77.8         | 42.2         | 60.1         | 83.6           |
| baichun-m1-preview | 是       | 62.8     | 72.2         | 74.4         | 65.7         | 41.8         | 76             |
| Deepseek-r1-32B    | 是       | 67       | 78.5         | 71.9         | 60           | 57.9         | 71.2           |
| Deepseek-r1-70B    | 是       | 0        | 64           | 79.9         | 0            | 0            | 39.2           |
| GPT4-O1mini        | 否       | 64.6     | 75.3         | 81.6         | 66.7         | 56.7         | 52.1           |
| 某医疗垂直大模型     | 否       | 77.1     | 71           | 90.2         | 70.4         | 71.9         | 68.5           |

**注意：** Deepseek-r1-70b在指令遵循方面表现不佳，难以按照要求输出答案，因此不建议在需要严格指令遵循的场景中使用。

#### 1.2 新版本LLM评测 (2025年11月之后)

##### 评测数据集
<img width="100%" alt="{AA452347-5E43-4768-87FA-56D141AED41D}" src="https://github.com/user-attachments/assets/3f526caa-0bef-4a27-a856-d986f7965d96" />

##### 评测结果
| 模型               | 是否开源 | 综合得分 | 医学知识问答 | 医学语言生成 | 复杂医学推理 | 医学语言理解 | 医疗安全和伦理 |
| ----------------- | ------- | -------- | ------------ | ----------- | ----------- | ----------- | ------------- |
| AntAngelMed-FP8  | 是      | 55.8    | 66.8        | 67.7        | 58.0        | 60.2        | 26.7           |
| GPT-5.2          | 否      | 54.5    | 71.7        | 72.1        | 61.1        | 49.4        | 18.3           |
| o3               | 否      | 56.2    | 71.0        | 69.5        | 58.9        | 63.3        | 18.3           |
|Baichuan-m2       | 是      | 58.5    | 72.8        | 71.6        |60.1         |66.4         |21.7            |
|Baichuan-m3       | 是      | 54.5    | 66.8        | 68.5        |57.2         |58.3         |21.7            |
---
#### 1.3 VLM评测 (2025年11月之后)
进行中...


### 二、评测方法

1.  **准备模型：** 配置待评测的医疗大模型。
2.  **修改配置：** 从环境变量中获取模型 API 密钥、模型名称和 API URL。
3.  **设置输出：** 指定评测结果的输出目录。
4.  **测试数据确定：** 在 main文件的run_evaluation函数中选择确定使用旧数据集还是新数据集。
4.  **运行主程序：** 直接执行 `main.py` 文件启动评测。
5.  **提交结果：** 通过 MedBench 官方平台提交评测结果：[https://medbench.opencompass.org.cn/home](https://medbench.opencompass.org.cn/home)

### 三、参数设置

```
system prompt:
You are a helpful assistant.
```
提示词设计很重要，很多得分比较低模型是输出格式不匹配，这边没有针对性的做改动，保留模型做原始的结果。
### 四、致谢

感谢 @MedBench 团队提供评测数据。

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
## MedicalAiBenchEval

这个评估评估专门用于在临床场景中评估 AI 模型，是蚂蚁AQ团队推出的。该框架基于 GAPS（Grounded 基于事实、Automated 自动化、Personalized 个性化、Scalable 可扩展）方法论，包含精心整理的临床基准数据集，以及用于医疗 AI 系统的自动化评估流水线。

该框架通过以下方式满足对 AI 临床决策进行标准化评估的关键需求：
- 临床扎根评估：评估标准基于真实的医学指南与专家知识
- 自动化流水线：从原始回答到详细性能指标的端到端高效处理
- 多模型支持：可同时评估多个 AI 模型，并进行对比分析
- 可扩展架构：支持并行执行，高效处理大规模数据集


### 一、评测结果
| 模型 | 是否开源 | 得分  |
| ---------- | ----------- | ----------- |
| antAngleMed-FP8 | 是 | 0.2533 |
| GPT-5.2         | 否 | 0.3697 |
| o3              | 否 | 0.4009 |
| Baichuan-M3     | 否 | 0.2535 |
| Baichuan-M2     | 是 | 0.3032 |
| Deepseek-V3.2   | 否 | 0.3239 |
| doubao-1.8      | 否 | 0.3027 |



### 二、环境准备
1. 安装必要的 Python 库：
   ```
   pip install -r requirements.txt
   ```
2. 配置环境变量：
   - 设置模型 API 密钥、模型名称和 API URL 为环境变量。（仅支持OpenAI兼容的API接口）
   - 根据config_model.yaml 自行设定
   - 例如：
     ```
     export MODEL_API_KEY=your_api_key
     export MODEL_NAME=your_model_name
     export MODEL_API_URL=your_api_url
     ```

### 三、评测方法
1.  **准备模型：** 配置待评测的医疗大模型。
2.  **获取 API 密钥：** 从模型供应商获取 API 密钥，用于访问模型的 API。
3.  **获取模型问答：** 针对“data\input\GAPS-NSCLC-preview.xlsx” 测试集，获取模型的问答结果。 
4.  **裁判模型配置：** 修改config_model.yaml文件,从环境变量中获取模型 API 密钥、模型名称和 API URL,设置为裁判模型
5.  **开始评测：** 根据模型生成的问答和裁判模型的判断，计算模型的得分。 
                  ```
                  python medical_evaluation_pipeline.py data/input/GAPS-NSCLC-preview.xlsx \
                        -o data/output/final_report.xlsx \
                        --judge-models m4 \
                        --voting-strategy conservative \
                  ```                        
6.  **分数获取** 在输出路径下会生成.xlsx，新生成的列包含模型正对每个问题的单独得分，求平均即为模型得分。
7. **得分计算规则：**
   - 每个问题的得分基于模型的回答和裁判模型的判断。每个问题都具有max_possible（理论最高分）
   - 模型的回答被切分成多个关键点进行评判,判断是否正确。
      - 不同类型问题具有不同得分：
        - 正分点（A类）：
          - A1 （5分）：影响患者安全的关键医学知识
          - A2 （3分）：重要的临床考虑因素
          - A3 （1分）：额外的相关信息
        - 负分点（S类）：
          - S1 （-1分）：不影响核心治疗的轻微错误
          - S2 （-2分）：可能误导的错误信息
          - S3 （-3分）：严重的医疗错误
          - S4 （-4分）：可能伤害患者的危险错误信息
      - 相加获取该问题的final_total_score， final_total_score/max_possible 获得归一化之后的得分(范围0-1)
   - 取平均即为模型最终得分。



### 四、致谢

感谢 @ AQ 团队提供评测数据。

```bibtex
@article{chen2025gaps,
  title={GAPS: A Clinically Grounded, Automated Benchmark for Evaluating AI Clinicians},
  author={Chen, Xiuyuan and Sun, Tao and Su, Dexin and Yu, Ailing and Liu, Junwei and Chen, Zhe and Jin, Gangzeng and Wang, Xin and Liu, Jingnan and Xiao, Hansong and Zhou, Hualei and Tao, Dongjie and Guo, Chunxiao and Yang, Minghui and Xia, Yuan and Zhao, Jing and Fan, Qianrui and Wang, Yanyun and Zhen, Shuai and Chen, Kezhong and Wang, Jun and Sun, Zewen and Zhao, Heng and Guan, Tian and Wang, Shaodong and Chang, Geyun and Deng, Jiaming and Chen, Hongchengcheng and Feng, Kexin and Li, Ruzhen and Geng, Jiayi and Zhao, Changtai and Wang, Jun and Lin, Guihu and Li, Peihao and Liu, Liqi and Wei, Peng and Wang, Jian and Gu, Jinjie and Wang, Ping and Yang, Fan},
  journal={arXiv preprint arXiv:2510.13734},
  year={2025},
  url={https://arxiv.org/abs/2510.13734}
}
```
## Healthbench
### 一：数据集描述
HealthBench 是openai发布的用评估AI模型在真实医疗场景中的表现的框架。

HealthBench 包含 5,000 次对话，模拟 AI 模型与个体用户或临床医生的交互过程。模型的任务是针对用户最后一条消息提供最佳回复。HealthBench 对话通过合成生成与人类对抗测试双重方式创建。这些对话场景旨在还原大型语言模型真实应用场景：它们采用多轮交互与多语言设计，涵盖普通民众与医疗从业者的多元角色，横跨各类医学专科与临床情境，并根据难度进行精选。具体示例请参见下方轮播图。

HealthBench 采用评分标准评估体系，每条模型回复均依据医生撰写的特定对话评分标准进行评分。每个标准明确规定理想回复应包含或避免的内容，例如需包含的具体事实或应避免的冗余技术术语。各标准对应特定权重值，该权重反映医生对该标准重要性的判断。HealthBench 包含 48,562 项独特评分标准，全面覆盖模型表现的各个维度。模型回复由基于模型的评分系统 (GPT‑4.1) 评估是否满足各项标准，并根据达标标准总分与满分的对比结果获得综合评分。
官网：https://openai.com/zh-Hans-CN/index/healthbench/

### 二：使用方法
1.  **准备模型：** 配置待评测的医疗大模型-（simple-eval.py文件中进行配置）。
2.  **配置裁判模型** 从模型供应商获取 API 密钥。（simple-eval.py文件中进行配置）
3.  **运行评测指令：** 
  测试指令：
  python simple_evals.py --model Baichuan-M3 --eval healthbench  --debug --n-threads 3
  运行指令：
  python simple_evals.py --model Baichuan-M3 --eval healthbench  --n-threads 3    
  参数解读
  - `--model`：指定待评测的医疗大模型名称。
  - `--eval`：指定评测数据集，这里是healthbench。
  - `--debug`：开启调试模式，输出详细信息。
  - `--n-threads`：指定并发线程数，默认为3。
4.  **分数获取**  根据配置的输出路径，生成.json和html 文件


### 三：结果分析
数据集类型分布： all:5000  
 - communication: 919,  
 - hedging: 1071,  
 - global_health: 1097,  
 - context_seeking: 594,  
 - emergency_referrals: 482,  
 - health_data_tasks: 477,  
 - complex_responses: 360,  
 格式：得分（标准差）  
| 模型 | 开源 | 综合得分 | communication | hedging | global_health | context_seeking | emergency_referrals | health_data_tasks | complex_responses |
| --- | --- | ------- | ------------- | -------- | ------------ | ---------------- | ------------------ | ------------------| ------------------ |
| AntAngelMed-FP8  | 是 | 0.435(0.0049)  | 0.391(0.012)  | 0.528(0.008) |  0.457(0.009)  | 0.467(0.014) | 0.493(0.013) | 0.3455(0.020) | 0.186(0.022) |

### 四：致谢
