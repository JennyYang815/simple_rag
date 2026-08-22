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

print("\n你的问题是：", question)

print(f"\n最相关的前 {top_k} 条资料：")

for rank, index in enumerate(top_indices, start=1):
    print(f"\n第 {rank} 名")
    print(f"相似度：{scores[index]:.4f}")
    print(f"资料：{documents[index]}")