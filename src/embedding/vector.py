from typing import List, Any
from fastembed import TextEmbedding
import json

class DenseEmbedding:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model = TextEmbedding(model_name= model_name)

    def embed_documents(self, documents: List[Any], batch_size: int = 32, save_path: str = "dense_embed.json") -> List[List[float]]:
        # Extract text whether documents are strings, dicts, or Document objects
        texts = [doc.page_content if hasattr(doc, "page_content") else (doc.get("text", "") if isinstance(doc, dict) else str(doc)) for doc in documents]
        
        dense_vectors = [emb.tolist() for emb in self.model.embed(texts, batch_size=batch_size)]
        
        # Save payload to JSON
        data = [{"id": f"chunk_{i}", "text": texts[i], "dense_vector": dense_vectors[i]} for i, doc in enumerate(documents)]
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            
        return dense_vectors
    
    def embed_query(self, query: str) -> List[float]:
        return list(self.model.embed([query]))[0].tolist()

