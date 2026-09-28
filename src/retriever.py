import os
import json
import hashlib
import numpy as np
import time

from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(
        self,
        chunks,
        model_name="BAAI/bge-small-zh-v1.5",
        top_k=3,
        threshold=0.54,
        cache_dir="cache"
    ):
        self.chunks = chunks
        self.top_k = top_k
        self.threshold = threshold
        self.model_name = model_name

        self.documents = [
            chunk["text"]
            for chunk in chunks
        ]
        
        # 加载Embedding模型 
        start = time.perf_counter()
        
        try:
            # 优先从本地加载，避免不必要的网络访问
            self.model = SentenceTransformer(
            model_name,
            local_files_only=True
            )

            print("检测到本地 Embedding 模型，直接加载。")

        except Exception:
            # 本地没有时，首次联网下载
            print("本地未找到 Embedding 模型，正在从 Hugging Face 下载...")

            self.model = SentenceTransformer(model_name)

            print("模型下载并加载完成。")
        
        print(
            f"Embedding 模型加载时间："
            f"{time.perf_counter() - start:.4f} 秒"
        )
        
         # 缓存路径
        os.makedirs(cache_dir, exist_ok=True)

        self.vector_path = os.path.join(
            cache_dir,
            "document_vectors.npy"
        )

        self.metadata_path = os.path.join(
            cache_dir,
            "metadata.json"
        )

        # 判断能否直接使用旧缓存
        if self._cache_is_valid():
            print("检测到已有文档向量，直接加载缓存。")

            self.document_vectors = np.load(
                self.vector_path
            )

        else:
            print("未找到有效缓存，正在计算文档 Embedding...")

            self.document_vectors = self.model.encode(
                self.documents,
                normalize_embeddings=True
            )

            self._save_cache()

            print("文档向量已保存到本地缓存。")


    def _get_document_hash(self):
        content = ""

        for chunk in self.chunks:
            content += (
                str(chunk["page"])
                + chunk["text"]
            )

        return hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()


    def _cache_is_valid(self):
        if not os.path.exists(self.vector_path):
            return False

        if not os.path.exists(self.metadata_path):
            return False

        try:
            with open(
                self.metadata_path,
                "r",
                encoding="utf-8"
            ) as file:
                metadata = json.load(file)

        except Exception:
            return False

        if metadata.get("model_name") != self.model_name:
            return False

        if metadata.get("document_hash") != self._get_document_hash():
            return False

        return True


    def _save_cache(self):
        np.save(
            self.vector_path,
            self.document_vectors
        )

        metadata = {
            "model_name": self.model_name,
            "document_hash": self._get_document_hash(),
            "chunk_count": len(self.chunks)
        }

        with open(
            self.metadata_path,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                metadata,
                file,
                ensure_ascii=False,
                indent=2
            )

    def retrieve(self, question):
        question_vector = self.model.encode(
            question,
            normalize_embeddings=True
        )

        scores = self.document_vectors @ question_vector

        top_indices = scores.argsort()[::-1][:self.top_k]

        results = []

        for index in top_indices:
            if scores[index] >= self.threshold:
                results.append({
                    "page": self.chunks[index]["page"],
                    "text": self.chunks[index]["text"],
                    "score": float(scores[index])
                })

        return results