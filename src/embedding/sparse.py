from typing import List, Dict, Any
from fastembed import SparseTextEmbedding

class SparseEmbedder:
    def __init__(self, model_name: str = "prithivida/Splade_PP_en_v1"):
        self.model = SparseTextEmbedding(model_name=model_name)
        
    def embed_documents(self, texts: List[str], batch_size : int = 32)	-> List[Dict[str, List[Any]]]:
        result = []
        for embed in self.model.embed(texts, batch_size=batch_size):
            result.append({
                "indices": embed.indices.tolist(),
                "values": embed.values.tolist()
            }) 
        return result
    def embed_query(self, query: str) -> Dict[str, List[Any]]:
        embed = self.model.embed_query([query])[0]
        return {
            "indices": embed.indices.tolist(),
            "values": embed.values.tolist()
        }