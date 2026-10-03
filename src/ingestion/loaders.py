from pypdf import PdfReader



loader = PdfReader("src/Artifical Intelligence.pdf")


for page in loader.pages:
    text = page.extract_text()
    print(text)