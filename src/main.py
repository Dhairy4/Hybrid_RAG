from ingestion.chuker import Split_Documents
from ingestion.loaders import load_pdf

# load the document
pdf = load_pdf("src/Artifical Intelligence.pdf")

chunked = Split_Documents(pdf)

# print(chunked)
print(len(chunked))






# testing the code in details

# from ingestion.chuker import Split_Documents
# from ingestion.loaders import load_pdf

# pdf = load_pdf("src/Artifical Intelligence.pdf")

# print("PDF type:", type(pdf))
# print("Number of documents:", len(pdf))
# print("First document:", pdf[0])
# print("First document type:", type(pdf[0]))

# chunked = Split_Documents(pdf)

# print("Number of chunks:", len(chunked))
# print("First chunk:", chunked[0])




