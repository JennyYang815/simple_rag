from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(
        self,
        chunks,
        model_name="BAAI/bge-small-zh-v1.5",
        top_k=3,
        threshold=0.54
    ):
        self.chunks = chunks
        self.top_k = top_k
        self.threshold = threshold

        self.model = SentenceTransformer(model_name)

        self.documents = [
            chunk["text"]
            for chunk in chunks
        ]

        self.document_vectors = self.model.encode(
            self.documents,
            normalize_embeddings=True
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