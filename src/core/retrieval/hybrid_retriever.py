from qdrant_client import models 
from vector_store.qdrant_store import client, COLLECTION_NAME

def retrieve( query, dense_embedder, sparse_embedder, top_k= 5):

    dense_query = dense_embedder.embed_query(query)
    sparse_query = sparse_embedder.embed_query(query)

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                query= dense_query,
                using="dense",
                limit= 20
            ),

            models.Prefetch(
                query= models.SparseVector(
                    indices=sparse_query["indices"], 
                    values=sparse_query["values"]),
                using="sparse",
                limit= 20,
            ),
        ],
        query = models.FusionQuery(
            fusion  =models.Fusion.RRF
        ),
        limit = top_k,
        with_payload= True,
    )

    return response.points
