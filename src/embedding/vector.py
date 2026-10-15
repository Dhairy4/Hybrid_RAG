from typing import List
from fastembed import TextEmbedding

class DenseEmbedding:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model = TextEmbedding(model_name= model_name)

    def embed_documents(self, texts: List[str], batch_size : int = 32) -> List[List[float]]:
        embeddings = self.model.embed(texts, batch_size = batch_size)
        return [emb.tolist() for emb in embeddings]
    
    def embed_query(self, query: str) -> List[float]:
        return list(self.model.embed([query]))[0].tolist()

