import time

from loader import load_pdf
from chunker import split_pages
from retriever import Retriever
from llm import generate_answer
from prompt import build_prompt

from config import (
    PDF_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    TOP_K,
    SIMILARITY_THRESHOLD
)

startup_start = time.perf_counter()

# 读取pdf文件
start = time.perf_counter()

pages = load_pdf(
    PDF_PATH,
    skip_pages=[1]
)

pdf_time = time.perf_counter() - start

# 切分chunk   
start = time.perf_counter()

chunks = split_pages(
    pages,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
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

print("\n===== 系统初始化完成 =====")
print(f"PDF 读取时间：{pdf_time:.4f} 秒")
print(f"文本切块时间：{chunk_time:.4f} 秒")
print(f"Retriever 初始化时间：{retriever_init_time:.4f} 秒")
print(f"系统启动时间：{startup_time:.4f} 秒")

# 输入问题
while True:

    question = input("请输入你的问题（输入exit退出）：").strip()
    
    if question.lower() in ["exit", "quit", "q"] or question == "退出":
        print("\n程序已退出。")
        break

    query_start = time.perf_counter()

    # 语义检索
    start = time.perf_counter()

    results = retriever.retrieve(question)

    retrieval_time = time.perf_counter() - start

    # 输出
    print("\n你的问题是：", question)

    if len(results) == 0:
        print("\n没有找到足够相关的资料。")
        
        query_time = time.perf_counter() - query_start

        print("\n===== 本次问答性能 =====")
        print(f"语义检索时间：{retrieval_time:.4f} 秒")
        print(f"本次处理时间：{query_time:.4f} 秒")

        continue

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
        
        # 调用大模型
        start = time.perf_counter()
        
        answer = generate_answer(prompt)
        
        llm_time = time.perf_counter() - start

        print("\n===== RAG 最终回答 =====")
        print(answer)
        
        source_pages = sorted(set(
            result["page"] for result in results
        ))

        print("\n参考来源：")
        print("、".join(
            f"第 {page} 页" for page in source_pages
        ))
        
        query_time = time.perf_counter() - query_start

        # 性能统计
        print("\n===== 本次问答性能 =====")
        print(f"语义检索时间：{retrieval_time:.4f} 秒")
        print(f"Prompt 构造时间：{prompt_time:.4f} 秒")
        print(f"LLM 回答时间：{llm_time:.4f} 秒")
        print(f"本次问答总时间：{query_time:.4f} 秒")

