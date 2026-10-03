from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def Split_Documents(documents: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_documents(documents)