from langchain_community.document_loaders import NotionDirectoryLoader
loader=NotionDirectoryLoader("C:/Users/Asus/Downloads/0f8b428b-efb0-4df9-bf0f-1d3cd1e2308c_Export-1cf0d4c6-471a-4573-8b08-e315aa0c5ad5")
docs=loader.load()
print(docs[0].page_content[0:200])
print(docs[0].metadata)
