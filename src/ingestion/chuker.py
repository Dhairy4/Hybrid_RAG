from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import json

def Split_Documents(documents: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    split_documents = text_splitter.split_documents(documents)

    import json


    serialized_chunks = [doc.model_dump() for doc in split_documents]  # Pydantic v2

    with open("chunks.json", "w") as f:
        json.dump(serialized_chunks, f, indent=4)
    return split_documents


