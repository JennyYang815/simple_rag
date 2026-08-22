from sentence_transformers import SentenceTransformer

# 读取文件
with open("data/knowledge.txt", "r", encoding="utf-8") as file:
    text = file.read()

# 切分chunk    
def split_text(text, chunk_size=300, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks

documents = split_text(text)

print(f"共切分成 {len(documents)} 个 Chunk：")

for i, chunk in enumerate(documents, start=1):
    print(f"\n--- Chunk {i} ---")
    print(chunk)

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

scores = document_vectors @ question_vector     # 比较相似度

top_k = 3

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
        print(f"资料：{documents[index]}")