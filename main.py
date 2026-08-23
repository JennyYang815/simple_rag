from sentence_transformers import SentenceTransformer
from loader import load_pdf
from chunker import split_pages

import os

from dotenv import load_dotenv
from openai import OpenAI

# 读取.env文件
load_dotenv()

# 创建大模型客户端
def generate_answer(prompt):
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com"
    )

    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content

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

# 提取chunk中的文字
documents = [chunk["text"] for chunk in chunks]

# 输入问题
question = input("请输入你的问题：")

# 加载模型
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

# Embedding编码
document_vectors = model.encode(
    documents,
    normalize_embeddings=True
)

question_vector = model.encode(
    question,
    normalize_embeddings=True
)

scores = document_vectors @ question_vector     # 计算相似度

top_k = 3   # Top-K   

top_indices = scores.argsort()[::-1][:top_k]

threshold = 0.54    # 相似度阈值

# 输出
print("\n你的问题是：", question)

valid_indices = []

for index in top_indices:
    if scores[index] >= threshold:
        valid_indices.append(index)

if len(valid_indices) == 0:
    print("\n没有找到足够相关的资料。")

else:
    print(f"\n找到 {len(valid_indices)} 条相关资料：")

    for rank, index in enumerate(valid_indices, start=1):
        print(f"\n第 {rank} 名")
        print(f"相似度：{scores[index]:.4f}")
        print(f"PDF 页码：第 {chunks[index]['page']} 页")
        print("内容：")
        print(chunks[index]["text"])
        
        context_parts = []

    for index in valid_indices:
        context_parts.append(
            f"[第 {chunks[index]['page']} 页]\n"
            f"{chunks[index]['text']}"
        )

    # 组织prompt
    context = "\n\n".join(context_parts)

    prompt = f"""
你是一个文档问答助手。请仅根据下面提供的参考资料回答问题。
如果参考资料中没有足够的信息，请回答“根据当前资料无法回答”，不要自己编造内容。
    
用户问题：
{question}

参考资料：
{context}

请回答：
"""

    print("\n===== 准备发送给大模型的 Prompt =====")
    print(prompt)
    
    answer = generate_answer(prompt)

    print("\n===== RAG 最终回答 =====")
    print(answer)

