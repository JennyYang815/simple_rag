import json
import time

from loader import load_pdf
from chunker import split_pages
from retriever import Retriever
from config import (
    PDF_PATH,
    EMBEDDING_MODEL
)


# =========================
# 读取测试集
# =========================

with open(
    "evaluation/test_questions.json",
    "r",
    encoding="utf-8"
) as file:
    test_questions = json.load(file)


# =========================
# 读取 PDF
# =========================

pages = load_pdf(
    PDF_PATH,
    skip_pages=[1]
)


# =========================
# 实验参数
# =========================

chunk_sizes = [100, 200, 300]
overlap = 50

experiment_results = []


# =========================
# 分别测试不同 Chunk Size
# =========================

for chunk_size in chunk_sizes:

    print("\n")
    print("=" * 50)
    print(f"开始测试 Chunk Size = {chunk_size}")
    print("=" * 50)

    # 切块
    chunks = split_pages(
        pages,
        chunk_size=chunk_size,
        overlap=overlap
    )

    print(f"Chunk 数量：{len(chunks)}")

    # 创建 Retriever
    # threshold=0，评测纯粹的检索排序能力
    retriever = Retriever(
        chunks,
        model_name=EMBEDDING_MODEL,
        top_k=3,
        threshold=0.0,
        cache_dir=f"cache/evaluation_{chunk_size}"
    )

    top1_correct = 0
    top3_correct = 0

    retrieval_times = []

    # 测试所有问题
    for item in test_questions:

        question = item["question"]
        expected_page = item["expected_page"]

        start = time.perf_counter()

        results = retriever.retrieve(question)

        retrieval_time = time.perf_counter() - start

        retrieval_times.append(retrieval_time)

        top3_pages = [
            result["page"]
            for result in results
        ]

        top1_page = (
            top3_pages[0]
            if len(top3_pages) > 0
            else None
        )

        if top1_page == expected_page:
            top1_correct += 1

        if top1_page != expected_page:
            print("\n[Top-1 未命中]")
            print(f"问题：{question}")
            print(f"正确页码：第 {expected_page} 页")
            print(f"Top-1 页码：第 {top1_page} 页")
            print(f"Top-3 页码：{top3_pages}")
        
        if expected_page in top3_pages:
            top3_correct += 1


    # =========================
    # 计算结果
    # =========================

    total = len(test_questions)

    top1_accuracy = top1_correct / total
    top3_accuracy = top3_correct / total

    average_retrieval_time = (
        sum(retrieval_times) / len(retrieval_times)
    )

    experiment_results.append({
        "chunk_size": chunk_size,
        "chunk_count": len(chunks),
        "top1_accuracy": top1_accuracy,
        "top3_accuracy": top3_accuracy,
        "avg_retrieval_time": average_retrieval_time
    })

    print("\n本组结果：")
    print(
        f"Top-1 命中率："
        f"{top1_correct}/{total} "
        f"({top1_accuracy * 100:.1f}%)"
    )

    print(
        f"Top-3 命中率："
        f"{top3_correct}/{total} "
        f"({top3_accuracy * 100:.1f}%)"
    )

    print(
        f"平均检索时间："
        f"{average_retrieval_time:.6f} 秒"
    )


# =========================
# 最终对比
# =========================

print("\n")
print("=" * 72)
print("Chunk Size 参数对比结果")
print("=" * 72)

print(
    f"{'Chunk Size':<12}"
    f"{'Chunk 数量':<12}"
    f"{'Top-1':<12}"
    f"{'Top-3':<12}"
    f"{'平均检索时间':<16}"
)

print("-" * 72)

for result in experiment_results:

    top1 = f"{result['top1_accuracy'] * 100:.1f}%"
    top3 = f"{result['top3_accuracy'] * 100:.1f}%"
    avg_time = f"{result['avg_retrieval_time']:.6f}"

    print(
        f"{result['chunk_size']:<12}"
        f"{result['chunk_count']:<12}"
        f"{top1:<12}"
        f"{top3:<12}"
        f"{avg_time:<16}"
    )