from ingestion.chuker import Split_Documents
from ingestion.loaders import load_pdf

from embedding.sparse import SparseEmbedding
from embedding.vector import DenseEmbedding

# load the document
pdf = load_pdf("src/Artifical Intelligence.pdf")

chunked = Split_Documents(pdf)

if chunked and hasattr(chunked[0], "page_content"):
    # If Split_Documents returns LangChain Document objects:
    texts = [doc.page_content for doc in chunked]

dense_embedder = DenseEmbedding()
sparse_embedder = SparseEmbedding()

dense_embeddings = dense_embedder.embed_documents(texts)
sparse_embeddings = sparse_embedder.embed_documents(texts)

from vector_store.qdrant_store import (
    create_collection,
    store_documents,
)
from core.retrieval.hybrid_retriever import retrieve

# You already have:
# chunks
# dense_embedder
# sparse_embedder

create_collection()

dense_vectors = dense_embedder.embed_documents(texts)
sparse_vectors = sparse_embedder.embed_documents(texts)

store_documents(
    texts,
    dense_vectors,
    sparse_vectors,
)

query = input("Ask a question: ")

results = retrieve(
    query,
    dense_embedder,
    sparse_embedder,
)

for result in results:
    print("Score:", result.score)
    print("Text:", result.payload["text"])
    print("Metadata:", {
        key: value
        for key, value in result.payload.items()
        if key != "text"
    })
    print("-" * 50)