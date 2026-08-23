from loader import load_pdf
from chunker import split_pages
from retriever import Retriever
from llm import generate_answer
from prompt import build_prompt


# 读取pdf文件
pages = load_pdf(
    "data/knowledge.pdf",
    skip_pages=[1]
)

# 切分chunk   
chunks = split_pages(
    pages,
    chunk_size=200,
    overlap=50
)

print(f"PDF 共读取 {len(pages)} 页")
print(f"共生成 {len(chunks)} 个 Chunk")

# 输入问题
question = input("请输入你的问题：")

# 语义检索并筛选
retriever = Retriever(
    chunks,
    top_k=3,
    threshold=0.54
)

results = retriever.retrieve(question)

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
    prompt = build_prompt(question, results)

    print("\n===== 准备发送给大模型的 Prompt =====")
    print(prompt)
    
    # 调用大模型
    answer = generate_answer(prompt)

    print("\n===== RAG 最终回答 =====")
    print(answer)

