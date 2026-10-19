from ingestion.chuker import Split_Documents
from ingestion.loaders import load_pdf

from embedding.sparse import SparseEmbedding
from embedding.vector import DenseEmbedding

from vector_store.qdrant_store import (
    create_collection,
    store_documents
)

from core.retrieval.hybrid_retriever import retrieve

from generation.llm import create_rag_chain

def main():

    # 1 load and split the document
    pdf = load_pdf("src/Artifical Intelligence.pdf")
    chunked = Split_Documents(pdf)

    if chunked and hasattr(chunked[0], "page_content"):
        # If Split_Documents returns LangChain Document objects:
        texts = [doc.page_content for doc in chunked]

    # 2 initalize embedding models
    dense_embedder = DenseEmbedding()
    sparse_embedder = SparseEmbedding()

    #3 generate embedding once
    dense_embeddings = dense_embedder.embed_documents(texts)
    sparse_embeddings = sparse_embedder.embed_documents(texts)

    # 4 store the chunks and vector
    create_collection()

    store_documents(
        chunked,
        dense_embeddings,
        sparse_embeddings
    )

    print(f"stored {len(chunked)} in qdrant")

    #5 retrieve the relevant documents
    while True :
        query = input("Enter your query: ")

        if query.lower() == "exit":
            break

        if not query:
            continue

        results = retrieve(
            query,
            dense_embedder,
            sparse_embedder
        )

        if not results:
            print("No relevant context was retrieved.")
            return

        context_parts = []

        for result in results:
            text = result.payload.get("text", "")

            context_parts.append(
                f"[Chunk {result.id}]\n{text}"
            )

        context = "\n\n".join(context_parts)

        rag_chain = create_rag_chain()

        answer = rag_chain.invoke({
            "question": query,
            "context": context,
        })

        print("\nAnswer:")
        print(answer)

        print("\nRetrieved sources:")
        for result in results:
            print(f"Chunk ID: {result.id}")
            print(result.payload.get("text", "")[:200])
            print("-" * 40)
            

if __name__ == "__main__":
    main()