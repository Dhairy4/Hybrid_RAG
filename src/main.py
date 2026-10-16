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

