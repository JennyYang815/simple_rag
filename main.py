from sentence_transformers import SentenceTransformer

with open("data/knowledge.txt", "r", encoding="utf-8") as file:
    documents = [
        line.strip()
        for line in file
        if line.strip()
    ]
    
print(f"已加载 {len(documents)} 条知识")

question = input("请输入你的问题：")

model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

document_vectors = model.encode(
    documents,
    normalize_embeddings=True
)

question_vector = model.encode(
    question,
    normalize_embeddings=True
)

scores = document_vectors @ question_vector

top_k = 3

top_indices = scores.argsort()[::-1][:top_k]

threshold = 0.60

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