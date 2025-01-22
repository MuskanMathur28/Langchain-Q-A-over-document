from langchain_community.document_loaders import PyPDFLoader
from transformers import pipeline

# Initialize summarizer
print("Initializing summarizer...")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)

# Load the PDF document
loader = PyPDFLoader("C:/Users/Asus/OneDrive/Documents/MLBOOK.pdf")
pages = loader.load()

# Check if pages are loaded correctly
if len(pages) == 0:
    raise ValueError("No pages were loaded from the PDF. Check the file path or content.")
print(f"Number of pages loaded: {len(pages)}")

# Extract text from all pages
document_text = ""
for page in pages:
    document_text += page.page_content
print(f"Extracted text length: {len(document_text)}")

# Function to apply contextual compression (summarization)
def contextual_compression(input_text, default_max_length=300, default_min_length=100, batch_size=10, output_file="summary.txt"):
    # Split text into smaller chunks
    chunk_size = 1000  # Define chunk size for large input
    chunks = [input_text[i:i + chunk_size] for i in range(0, len(input_text), chunk_size)]
    total_chunks = len(chunks)
    print(f"Total chunks to process: {total_chunks}")
    
    summary = ""
    for i, chunk in enumerate(chunks):
        chunk_word_count = len(chunk.split())  # Word count of the chunk
        max_length = min(default_max_length, chunk_word_count // 2)  # Adjust max_length dynamically
        
        # Ensure min_length is less than max_length
        min_length = min(default_min_length, max_length - 10) if max_length > 10 else 1
        
        # Summarize each chunk
        print(f"Processing chunk {i + 1} of {total_chunks}")
        chunk_summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
        summary += chunk_summary[0]['summary_text'] + "\n"
        
        # Save intermediate summary after processing each batch
        if (i + 1) % batch_size == 0 or i == total_chunks - 1:
            with open(output_file, "a", encoding="utf-8") as file:
                file.write(f"Summary after chunk {i + 1}:\n")
                file.write(summary + "\n")
            print(f"Intermediate summary saved to {output_file} after processing {i + 1} chunks.")
            summary = ""  # Clear summary for the next batch
            
            # Stop processing if batch is complete
            if (i + 1) % batch_size == 0:
                break
    
    return summary

# Apply summarization and save to file
batch_size = 10  # Number of chunks to process at a time
output_file = "summary.txt"  # File to save the compressed summary
compressed_text = contextual_compression(document_text, batch_size=batch_size, output_file=output_file)

# Print confirmation
print(f"Compressed summary saved to {output_file}.")
