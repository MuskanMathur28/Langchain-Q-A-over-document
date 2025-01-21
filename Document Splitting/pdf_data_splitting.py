from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load the PDF
loader = PyPDFLoader("C:/Users/Asus/OneDrive/Documents/MLBOOK.pdf")
pages = loader.load()

# Display basic information about the PDF
print(f"Total pages: {len(pages)}")
print(f"First 500 characters of page 1:\n{pages[0].page_content[:500]}")
print(f"Metadata of page 1:\n{pages[0].metadata}")

# Combine all pages into a single document
document_text = " ".join([page.page_content for page in pages])

# Apply RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Maximum size of each chunk in characters
    chunk_overlap=200,  # Overlap between chunks to maintain context
    separators=["\n\n", "\n", " "]  # Prioritize splitting by paragraphs, then lines, then spaces
)

# Split the document into chunks
chunks = text_splitter.split_text(document_text)

# Display information about the chunks
print(f"\nTotal chunks: {len(chunks)}")
for i, chunk in enumerate(chunks[:3], 1):  # Display the first 3 chunks
    print(f"\nChunk {i}:\n{chunk}\n")
