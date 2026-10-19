from uuid import NAMESPACE_URL, uuid5
from qdrant_client import QdrantClient, models

COLLECTION_NAME = "hybrid_rag"

client = QdrantClient(url="http://localhost:6333")


def create_collection():

    if client.collection_exists(COLLECTION_NAME):
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config= {
            "dense": models.VectorParams(
                size=384,
                distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams()
        },
    )

def store_documents(documents, dense_vectors, sparse_vectors):
    points = []

    for i, doc in enumerate(documents):
        points.append(
            models.PointStruct(
                id=i,
                vector={
                    "dense": dense_vectors[i],
                    "sparse": models.SparseVector(
                        indices=sparse_vectors[i]["indices"],
                        values=sparse_vectors[i]["values"],
                    ),
                },
                payload={
                    "text": doc.page_content,
                    **(doc.metadata or {}),
                },
            )
        )



    client.upload_points(
            collection_name=COLLECTION_NAME,
            points=points,
            batch_size=64,
        )