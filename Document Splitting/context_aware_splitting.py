# Import necessary libraries
from langchain.text_splitter import MarkdownHeaderTextSplitter

# Define the Markdown document
markdown_document = """# Title

## Chapter 1
Hi this is Jim
Hi this is Joe

### Section
Hi this is Lance

## Chapter 2
Hi this is Molly"""

# Define the headers to split on (for Title, Chapters, and Sections)
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

# Initialize the MarkdownHeaderTextSplitter with the defined headers
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)

# Use the split_text method to split the document based on headers
md_header_splits = markdown_splitter.split_text(markdown_document)

# Print the splits and their metadata
for i, split in enumerate(md_header_splits, 1):
    print(f"Split {i}:")
    print(f"Content: {split.page_content}")
    print(f"Metadata: {split.metadata}\n")
