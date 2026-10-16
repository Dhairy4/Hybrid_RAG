from typing import List, Dict, Any
from fastembed import SparseTextEmbedding
import json

class SparseEmbedding:
    def __init__(self, model_name: str = "prithivida/Splade_PP_en_v1"):
        self.model = SparseTextEmbedding(model_name=model_name)
        
    def embed_documents(self, documents: List[Any], batch_size: int = 32, save_path: str = "sparse_embed.json") -> List[Dict[str, List[Any]]]:
        # Extract text whether documents are strings, dicts, or Document objects
        texts = [doc.page_content if hasattr(doc, "page_content") else (doc.get("text", "") if isinstance(doc, dict) else str(doc)) for doc in documents]
        
        sparse_vectors = [{"indices": emb.indices.tolist(), "values": emb.values.tolist()} for emb in self.model.embed(texts, batch_size=batch_size)]
        
        # Save payload to JSON
        data = [{"id": f"chunk_{i}", "text": texts[i], "sparse_vector": sparse_vectors[i]} for i, doc in enumerate(documents)]
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            
        return sparse_vectors
    def embed_query(self, query: str) -> Dict[str, List[Any]]:
        embed = self.model.embed_query([query])[0]
        return {
            "indices": embed.indices.tolist(),
            "values": embed.values.tolist()
        }