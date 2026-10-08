from pypdf import PdfReader
from langchain_core.documents import Document

# read the document on the given path
def load_pdf(file_path):
    loader = PdfReader(file_path)

    documents = []

    for page in loader.pages:
        text = page.extract_text()
        documents.append(Document(page_content=text))

    return documents
                    


## testing in this file

# load_pdf = PdfReader("src/Artifical Intelligence.pdf")

# print(load_pdf.pages[5].extract_text())