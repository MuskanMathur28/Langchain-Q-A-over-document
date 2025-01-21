from langchain_community.document_loaders import PyPDFLoader
loader=PyPDFLoader("C:/Users/Asus/OneDrive/Documents/MLBOOK.pdf")
pages=loader.load()
print(len(pages))
print(pages[0].page_content[0:500])
print(pages[0].metadata)
