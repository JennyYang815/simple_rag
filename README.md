# Simple RAG

一个用于学习 RAG 基本原理的小型项目（Just for practice.XD）
本项目不直接使用 LangChain 等 RAG 框架封装，而是将 PDF 解析、文本切块、Embedding、语义检索、Prompt 构造和大语言模型调用等步骤逐步实现，以理解一个基础 RAG 系统的完整工作流程。


## 项目流程

当前实现的基本流程如下：

```text
PDF 文档
   ↓
文本提取
   ↓
Chunk 文本切块
   ↓
Embedding 向量化
   ↓
用户问题向量化
   ↓
计算语义相似度
   ↓
Top-K 检索
   ↓
相似度阈值过滤
   ↓
构造 Context + Prompt
   ↓
大语言模型
   ↓
生成最终回答
```

## 当前功能

目前已经实现：

* PDF 文本读取
* PDF 页面信息保留
* 固定长度文本切块（Chunk）
* Chunk Overlap
* 中文 Embedding
* 语义相似度计算
* Top-K 相关文本检索
* 相似度阈值过滤
* 检索结果页码溯源
* RAG Context 构造
* Prompt 构造
* 调用大语言模型生成回答
* 基础检索实验记录

当前使用的 Embedding 模型：

```text
BAAI/bge-small-zh-v1.5
```

## 示例

测试问题：

```text
流水线提高的是延迟还是吞吐率？
```

系统首先从 PDF 中检索相关内容：

```text
PDF 页码：第 5 页

流水线的主要作用是提高指令吞吐率，
也就是单位时间内完成更多指令。
它通常不会直接缩短单条指令从开始到结束的总延迟。
```

随后将检索结果作为 Context 提供给大语言模型，得到最终回答：

```text
根据参考资料，流水线的主要作用是提高指令吞吐率，
也就是单位时间内完成更多指令；
它通常不会直接缩短单条指令的总延迟。

因此答案是：吞吐率。
```

## 项目结构

当前项目结构：

```text
simple_rag/
│
├── data/
│   └── knowledge.pdf       # 测试知识库
│
├── main.py                 # 当前 RAG 主程序
├── experiments.md          # 实验记录
├── README.md
├── .gitignore
└── .env                    # API Key，本文件不会上传 GitHub
```

后续计划进一步拆分不同模块，提高代码可维护性。

## 检索方法

首先将 PDF 文本划分为多个 Chunk：

```text
PDF
 ↓
Chunk 1
Chunk 2
Chunk 3
...
```

每个 Chunk 使用 Embedding 模型转换为向量。

用户问题同样转换为向量，然后计算问题向量与各 Chunk 向量之间的相似度，并返回相似度最高的 Top-K 个 Chunk。

当前使用：

```python
top_k = 3
```

同时使用相似度阈值过滤明显不相关的结果。


## 当前存在的问题

目前版本仍然比较基础，存在以下问题：

* 相似度阈值仍然依赖人工设置
* 相邻 Chunk 存在 Overlap，Top-K 中可能出现内容重复
* Chunk Size 和 Overlap 尚未经过系统调参
* 每次启动程序都需要加载 Embedding 模型，存在一定启动延迟
* 当前主要针对单个 PDF 进行测试
* PDF 中的封面、目录等非正文内容可能干扰检索，需要进行数据清洗
* 当前代码主要集中在 `main.py`，尚未进行完整模块化

## 后续计划

接下来计划逐步完成：

* [ ] 将 PDF Loader、Chunker、Retriever 和 LLM 调用拆分为独立模块
* [ ] 完善来源页码引用
* [ ] 增加运行时间统计
* [ ] 对 Chunk Size、Overlap 和 Top-K 进行更系统的实验
* [ ] 改进固定相似度阈值策略
* [ ] 减少 Top-K 中重复 Chunk
* [ ] 增加多个测试问题构成简单评测集
* [ ] 完善实验结果与 README
* [ ] 增加简单的 Web 交互界面

## 项目目的

该项目目前主要用于学习 RAG 系统的基本工作原理，而不是构建完整的生产级知识库系统。

希望通过从基础组件开始实现的方式，理解以下问题：

* Embedding 如何支持语义检索？
* Chunk Size 为什么会影响检索效果？
* Top-K 应该如何选择？
* 固定相似度阈值有什么局限？
* PDF 文档质量如何影响 RAG？
* Retriever 的结果如何影响最终 LLM 回答？

后续将在当前基础上继续进行工程化整理和检索策略实验。

