from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# 读取文件
def load_pdf(file_path):
    reader = PdfReader(file_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        
        # 第一页为无关内容，跳过
        if page_number == 1:
            continue
        
        text = page.extract_text()

        if text:
            pages.append({
                "page": page_number,
                "text": text
            })

    return pages

pages = load_pdf("data/knowledge.pdf")

# 切分chunk   
def split_pages(pages, chunk_size=200, overlap=50):
    chunks = []

    for page in pages:
        text = page["text"]

        start = 0

        while start < len(text):
            end = start + chunk_size

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "page": page["page"],
                    "text": chunk_text
                })

            if end >= len(text):
                break

            start = end - overlap

    return chunks 

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

threshold = 0.60    # 相似度阈值

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

