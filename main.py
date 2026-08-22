from sentence_transformers import SentenceTransformer

documents = [
    "Cache 是位于 CPU 和主存之间的高速存储器。",
    "RISC-V 是一种开源的精简指令集架构。",
    "哈夫曼编码是一种变长编码方法。",
    "虚拟内存可以让程序使用比物理内存更大的地址空间。",
    "流水线可以提高处理器执行指令的吞吐率。"
]

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