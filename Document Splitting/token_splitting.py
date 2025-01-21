from langchain_community.document_loaders import NotionDirectoryLoader
import tiktoken

# Load Notion data
loader = NotionDirectoryLoader("C:/Users/Asus/Downloads/0f8b428b-efb0-4df9-bf0f-1d3cd1e2308c_Export-1cf0d4c6-471a-4573-8b08-e315aa0c5ad5")
docs = loader.load()

# Check if we have valid content
if len(docs) == 0 or len(docs[0].page_content.strip()) == 0:
    print("Error: No valid content in the Notion document.")
else:
    # Combine all documents into a single text
    document_text = " ".join([doc.page_content for doc in docs])

    # Print the length of the document to check if it's too small
    print(f"Document length: {len(document_text)}")
    
    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("cl100k_base")  # Tokenizer for OpenAI models
    tokens = tokenizer.encode(document_text)

    # Print the number of tokens to check if tokenization is working
    print(f"Number of tokens: {len(tokens)}")

    # If tokens are too small, return early to avoid empty chunk errors
    if len(tokens) == 0:
        print("Error: No tokens generated from the document text.")
    else:
        # Split tokens into chunks
        chunk_size = 100  # Number of tokens per chunk
        token_chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]

        # Decode token chunks back to text
        decoded_chunks = [tokenizer.decode(chunk) for chunk in token_chunks]

        # Output results
        print(f"Total token chunks: {len(decoded_chunks)}")
        if len(decoded_chunks) > 0:
            print(f"Sample chunk:\n{decoded_chunks[0]}")
        else:
            print("No chunks to display.")
