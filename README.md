# Simple RAG

一个从零实现的轻量级PDF RAG问答系统，用于学习 RAG 基本原理（Just for practice.XD）
本项目不直接使用LangChain等RAG框架封装，而是将PDF解析、文本切块、Embedding、语义检索、Prompt构造和大语言模型调用等步骤逐步实现，并针对Retriever性能和Chunk参数进行了初步实验。


## 项目流程

当前实现的基本流程如下：

```text
PDF文档
   ↓
文本提取
   ↓
Chunk文本切块
   ↓
Embedding向量化
   ↓
用户问题向量化
   ↓
计算语义相似度
   ↓
Top-K检索
   ↓
相似度阈值过滤
   ↓
构造Context + Prompt
   ↓
大语言模型
   ↓
最终回答+来源页码
```

## 当前功能

目前已经实现：

- PDF文本读取
- PDF页码保留与来源溯源
- 固定长度Chunk切分
- Chunk Overlap
- 中文Embedding
- Top-K语义检索
- 相似度阈值过滤
- Prompt构造
- DeepSeek API调用
- 最终回答来源页码展示
- 连续问答
- Embedding模型本地优先加载
- 文档Embedding本地缓存
- Retriever自动评测
- Chunk Size参数对比实验
- RAG各阶段性能统计
- 基础模块化工程结构

当前使用的 Embedding 模型：

```text
BAAI/bge-small-zh-v1.5
```
当前主要参数：

```text
Chunk Size=200
Overlap=50
Top-K=3
Threshold=0.54
```

## 项目结构

```text
simple_rag/
│
├── data/
│   └── knowledge.pdf
│
├── evaluation/
│   └── test_questions.json
│
├── loader.py
├── chunker.py
├── retriever.py
├── prompt.py
├── llm.py
├── config.py
│
├── main.py
├── evaluate.py
│
├── experiments.md
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

各模块职责：

```text
loader.py
→读取PDF并保留页码

chunker.py
→文本切块

retriever.py
→Embedding、Top-K检索、Threshold过滤和向量缓存

prompt.py
→构造RAG Prompt

llm.py
→调用大语言模型

config.py
→统一管理项目参数

main.py
→RAG主流程

evaluate.py
→Retriever自动评测
```

## 安装与运行

### 1.创建虚拟环境

```bash
python -m venv .venv
```

Windows PowerShell：

```bash
.\.venv\Scripts\Activate.ps1
```

### 2.安装依赖

```bash
pip install -r requirements.txt
```

### 3.配置API Key

复制：

```text
.env.example
```

并创建：

```text
.env
```

填写：

```text
DEEPSEEK_API_KEY=your_api_key_here
```

### 4.运行RAG

```bash
python main.py
```

程序初始化完成后可以连续提问：

```text
请输入你的问题：Cache利用了什么原理？

===== RAG最终回答 =====
Cache利用的是程序访问的局部性原理，包括时间局部性和空间局部性。

参考来源：
第2页
```

输入：

```text
exit
```

即可退出。

## Embedding模型加载策略

项目首先尝试从本地加载Embedding模型：

```text
本地存在模型
↓
直接加载
↓
避免Hugging Face网络访问
```

如果新环境中没有对应模型：

```text
本地未找到模型
↓
自动连接Hugging Face Hub
↓
下载模型
↓
保存到本地缓存
```

这样既可以避免已有模型时不必要的网络访问，又保证项目在新环境中能够运行。

## 性能实验

初始版本中，Retriever初始化曾需要约11～13s，并且受到Hugging Face网络访问影响，极端情况下甚至达到数十秒。

通过分析发现，主要波动来自Embedding模型加载时的网络访问。

改为本地优先加载后，连续3次测试结果为：

| 测试 | Embedding模型加载 | Retriever初始化 | 系统启动 |
| ---: | ---: | ---: | ---: |
| 1 | 0.1312s | 0.1321s | 0.1602s |
| 2 | 0.1330s | 0.1341s | 0.1596s |
| 3 | 0.1324s | 0.1333s | 0.1591s |

说明在模型和文档向量均已缓存的情况下，系统启动时间可以稳定在约0.16s。

在连续问答实验中，语义检索通常只需要约0.006～0.009s，单次问答主要耗时来自LLM调用。

## Retriever自动评测

目前建立了包含15个问题的小型人工标注测试集，覆盖：

- Cache
- RISC-V
- 虚拟内存与TLB
- 指令流水线
- 哈夫曼编码

在当前`Chunk Size=200`、`Overlap=50`配置下：

```text
Top-1命中率：15/15（100%）
Top-3命中率：15/15（100%）
```

当前测试集规模仍然较小，因此该结果主要用于验证Retriever基本能力，并不能代表系统在大型知识库中的实际性能。

## Chunk Size实验

固定`Overlap=50`，分别测试不同Chunk Size：

| Chunk Size | Chunk数量 | Top-1 | Top-3 | 平均检索时间 |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 24 | 100.0% | 100.0% | 0.005903s |
| 200 | 11 | 100.0% | 100.0% | 0.005633s |
| 300 | 6 | 93.3% | 100.0% | 0.005957s |

当`Chunk Size=300`时出现一次Top-1未命中，但正确结果仍位于Top-3中。

初步说明：

- Chunk过大可能混入更多语义，影响具体问题的Top-1排序。
- Chunk过小会产生更多向量，增加索引规模。
- Top-K检索能够一定程度上缓解Top-1排序不稳定的问题。

因此当前暂时采用：

```text
Chunk Size=200
Overlap=50
Top-K=3
```

详细实验过程见`experiments.md`。

## 当前存在的问题

目前仍存在以下局限：

- 测试知识库规模较小
- 自动评测集目前只有15个问题
- Threshold仍然是人工设定的固定值
- 相邻Chunk可能因为Overlap产生重复内容
- 当前仅支持固定PDF文件
- 尚未加入图形化交互界面
- 尚未测试大规模向量检索性能

## 后续计划

- [ ] 增加Streamlit Web界面
- [ ] 支持网页上传PDF
- [ ] 完善回答来源展示
- [ ] 扩大Retriever测试集
- [ ] 在更大的PDF知识库上进行测试
- [ ] 进一步研究Threshold和Top-K参数
- [ ] 研究重复Chunk去除策略
- [ ] 完善README截图和项目架构图

## 项目目的

本项目的主要目标不是构建生产级RAG平台，而是通过从基础组件开始实现一个完整RAG流程，理解以下问题：

- Embedding如何实现语义检索？
- Chunk Size为什么会影响检索结果？
- Top-K有什么作用？
- 固定Threshold存在哪些局限？
- PDF解析质量如何影响RAG？
- Retriever错误如何影响最终回答？
- RAG系统真正的性能瓶颈在哪里？
- 如何通过缓存减少重复计算和网络访问？

后续将在当前基础上继续完善工程结构、自动评测和交互界面。