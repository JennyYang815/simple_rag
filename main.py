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

best_index = scores.argmax()

print("\n你的问题是：", question)

print("\n所有资料的相似度：")

for i in range(len(documents)):
    print(f"{scores[i]:.4f}  {documents[i]}")

print("\n最相关的资料：")
print(documents[best_index])