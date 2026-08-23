import time

from loader import load_pdf
from chunker import split_pages
from retriever import Retriever
from llm import generate_answer
from prompt import build_prompt


startup_start = time.perf_counter()

# 读取pdf文件
start = time.perf_counter()

pages = load_pdf(
    "data/knowledge.pdf",
    skip_pages=[1]
)

pdf_time = time.perf_counter() - start

# 切分chunk   
start = time.perf_counter()

chunks = split_pages(
    pages,
    chunk_size=200,
    overlap=50
)

chunk_time = time.perf_counter() - start

print(f"PDF 共读取 {len(pages)} 页")
print(f"共生成 {len(chunks)} 个 Chunk")

# 初始化Retriever
start = time.perf_counter()

retriever = Retriever(
    chunks,
    top_k=3,
    threshold=0.54
)
retriever_init_time = time.perf_counter() - start

startup_time = time.perf_counter() - startup_start

# 输入问题

question = input("请输入你的问题：")

query_start = time.perf_counter()

# 语义检索
start = time.perf_counter()

results = retriever.retrieve(question)

retrieval_time = time.perf_counter() - start

# 输出
print("\n你的问题是：", question)

if len(results) == 0:
    print("\n没有找到足够相关的资料。")

else:
    print(f"\n找到 {len(results)} 条相关资料：")

    for rank, result in enumerate(results, start=1):
        print(f"\n===== 第 {rank} 名 =====")
        print(f"相似度：{result['score']:.4f}")
        print(f"PDF 页码：第 {result['page']} 页")
        print("内容：")
        print(result["text"])
    
    # 组织Prompt
    start = time.perf_counter()
     
    prompt = build_prompt(question, results)
    
    prompt_time = time.perf_counter() - start

    print("\n===== 准备发送给大模型的 Prompt =====")
    print(prompt)
    
    # 调用大模型
    start = time.perf_counter()
    
    answer = generate_answer(prompt)
    
    llm_time = time.perf_counter() - start

    print("\n===== RAG 最终回答 =====")
    print(answer)
    
    query_time = time.perf_counter() - query_start

    # 性能统计
    print("\n===== 运行时间统计 =====")
    print(f"PDF 读取时间：{pdf_time:.4f} 秒")
    print(f"文本切块时间：{chunk_time:.4f} 秒")
    print(f"Retriever 初始化时间：{retriever_init_time:.4f} 秒")
    print(f"语义检索时间：{retrieval_time:.4f} 秒")
    print(f"Prompt 构造时间：{prompt_time:.4f} 秒")
    print(f"LLM 回答时间：{llm_time:.4f} 秒")
    print(f"系统启动时间：{startup_time:.4f} 秒")
    print(f"单次问答时间：{query_time:.4f} 秒")

