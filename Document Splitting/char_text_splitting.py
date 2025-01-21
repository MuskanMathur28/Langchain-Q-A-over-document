from langchain.text_splitter import CharacterTextSplitter

# Sample unstructured data
unstructured_text = """
Here is some unstructured data. This text doesn't have any clear boundaries like paragraphs or headings. It's just a long block of text that can be processed to make it more manageable.
We want to split this text into smaller chunks to make it easier to work with. The goal is to break it down into pieces without worrying too much about structure.
Character splitting, tokenization, and other techniques are great ways to deal with unstructured data in natural language processing tasks. But the most important thing is to make sure the chunks are small enough to be processed by the model.
Let's see how this chunk looks when split.
"""

# Apply CharacterTextSplitter
character_splitter = CharacterTextSplitter(
    chunk_size=1000,  # Each chunk will be 1000 characters long
    chunk_overlap=200  # 200-character overlap between chunks
)
character_chunks = character_splitter.split_text(unstructured_text)

# Output results
print(f"Total chunks: {len(character_chunks)}")
print(f"Sample chunk:\n{character_chunks[0]}")
