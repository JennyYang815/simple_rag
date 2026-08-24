import json

from loader import load_pdf
from chunker import split_pages
from retriever import Retriever
from config import (
    PDF_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL
)


# 读取测试集
with open(
    "evaluation/test_questions.json",
    "r",
    encoding="utf-8"
) as file:
    test_questions = json.load(file)

# 建立知识库
pages = load_pdf(
    PDF_PATH,
    skip_pages=[1]
)

chunks = split_pages(
    pages,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
)

# 评测时暂时关闭threshold
# 只观察Retriever本身的排序能力
retriever = Retriever(
    chunks,
    model_name=EMBEDDING_MODEL,
    top_k=3,
    threshold=0.0
)

top1_correct = 0
top3_correct = 0

for i, item in enumerate(test_questions, start=1):

    question = item["question"]
    expected_page = item["expected_page"]

    results = retriever.retrieve(question)

    top3_pages = [
        result["page"]
        for result in results
    ]

    top1_page = (
        top3_pages[0]
        if len(top3_pages) > 0
        else None
    )

    top1_hit = top1_page == expected_page
    top3_hit = expected_page in top3_pages

    if top1_hit:
        top1_correct += 1

    if top3_hit:
        top3_correct += 1


    print(f"\n===== 测试 {i} =====")
    print(f"问题：{question}")
    print(f"正确页码：第 {expected_page} 页")
    print(f"Top-1 页码：第 {top1_page} 页")
    print(f"Top-3 页码：{top3_pages}")
    print(f"Top-1：{'命中' if top1_hit else '未命中'}")
    print(f"Top-3：{'命中' if top3_hit else '未命中'}")


total = len(test_questions)

top1_accuracy = top1_correct / total
top3_accuracy = top3_correct / total

print("\n========================")
print("      评测结果")
print("========================")

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