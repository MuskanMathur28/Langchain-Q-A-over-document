from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
import re  # Regular expressions for cleaning

# Step 1: Load the PDF
loader = PyPDFLoader("C:/Users/Asus/OneDrive/Documents/MLBOOK.pdf")
documents = loader.load()
documents = documents[:5]  # Limit to the first 5 pages for testing

# Step 2: Split the document into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# Step 3: Extract and clean text
clean_texts = []
for doc in docs:
    text = doc.page_content
    cleaned_text = re.sub(r'[^A-Za-z0-9\s.,?!]', '', text)  # Keep only valid characters
    clean_texts.append(cleaned_text)

# Debugging: Print some cleaned text
print("Cleaned Texts (First 200 characters of each chunk):")
for text in clean_texts[:3]:
    print(text[:200])

# Step 4: Generate embeddings using HuggingFaceEmbeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embeddings = embedding_model.embed_documents(clean_texts)  # Generate embeddings

# Step 5: Create FAISS vectorstore
vectorstore = FAISS.from_texts(clean_texts, embedding_model)  # Corrected method

# Step 6: Set up the retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})

# Step 7: Query the vectorstore using 'invoke'
query = "What is the main topic discussed in the document?"
results = retriever.invoke(query)  # Updated to use 'invoke'

# Display results
print("\nRelevant Documents:")
for i, result in enumerate(results):
    print(f"Document {i+1}:\n{result.page_content[:300]}...\n")  # Corrected extraction
