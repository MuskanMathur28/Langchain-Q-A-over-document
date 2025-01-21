import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Define the headers with the User-Agent string
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# Load webpage content
url = "https://en.wikipedia.org/wiki/OpenAI"
response = requests.get(url, headers=headers)

# Parse the content using BeautifulSoup
soup = BeautifulSoup(response.content, "html.parser")

# Extract the main content
content = soup.get_text()

# Save the raw content to a file (optional)
with open('output.txt', 'w', encoding='utf-8') as file:
    file.write(content)

# Apply text splitting using RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Max size of each chunk in characters
    chunk_overlap=200,  # Overlap between chunks to maintain context
    separators=["\n\n", "\n", " "]  # Prioritize paragraphs, then lines, then spaces
)

# Split the content into chunks
chunks = text_splitter.split_text(content)

# Display information about the chunks
print(f"Total chunks: {len(chunks)}")
for i, chunk in enumerate(chunks[:3], 1):  # Display the first 3 chunks
    print(f"\nChunk {i}:\n{chunk}\n")

# Optionally save chunks to a file for later use
with open('output_chunks.txt', 'w', encoding='utf-8') as file:
    for i, chunk in enumerate(chunks, 1):
        file.write(f"Chunk {i}:\n{chunk}\n\n")

print("Content chunks have been saved to 'output_chunks.txt'")
