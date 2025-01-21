from langchain_community.document_loaders import NotionDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load Notion data
loader = NotionDirectoryLoader("C:/Users/Asus/Downloads/0f8b428b-efb0-4df9-bf0f-1d3cd1e2308c_Export-1cf0d4c6-471a-4573-8b08-e315aa0c5ad5")
docs = loader.load()

# Display basic information about the first document
print(f"First 200 characters of the first document:\n{docs[0].page_content[:200]}")
print(f"Metadata of the first document:\n{docs[0].metadata}")

# Combine all documents into a single text
document_text = " ".join([doc.page_content for doc in docs])

# Apply RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Maximum size of each chunk in characters
    chunk_overlap=200,  # Overlap between chunks to maintain context
    separators=["\n\n", "\n", " "]  # Prioritize splitting by paragraphs, then lines, then spaces
)

# Split the combined document text into chunks
chunks = text_splitter.split_text(document_text)

# Display information about the chunks
print(f"\nTotal chunks: {len(chunks)}")
for i, chunk in enumerate(chunks[:3], 1):  # Display the first 3 chunks
    print(f"\nChunk {i}:\n{chunk}\n")
