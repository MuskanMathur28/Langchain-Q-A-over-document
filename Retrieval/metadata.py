from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 1. Prepare a list of documents with metadata (e.g., title, author, date)
documents = [
    {"content": "Python is a programming language.", "title": "Python Programming", "author": "John Doe", "date": "2023-05-10"},
    {"content": "Machine learning is a field of AI.", "title": "Machine Learning", "author": "Jane Smith", "date": "2023-06-15"},
    {"content": "Deep learning models are used for AI tasks.", "title": "Deep Learning", "author": "Alice Johnson", "date": "2023-07-20"},
]

# 2. Initialize the SentenceTransformer model to convert text into embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# 3. Function to compute cosine similarity between two vectors
def cosine_similarity_score(vec1, vec2):
    return cosine_similarity([vec1], [vec2])[0][0]

# 4. Function to retrieve top documents based on query and metadata
def retrieve_documents(query, documents, top_k=3):
    # 4.1 Generate embeddings for the query and all documents
    query_embedding = model.encode([query])[0]
    document_embeddings = [model.encode([doc["content"]])[0] for doc in documents]
    
    # 4.2 Compute relevance score (cosine similarity) for each document
    similarities = []
    for i, doc_embedding in enumerate(document_embeddings):
        relevance_score = cosine_similarity_score(query_embedding, doc_embedding)
        similarities.append((relevance_score, documents[i]))
    
    # 4.3 Sort documents by relevance score in descending order
    similarities.sort(key=lambda x: x[0], reverse=True)
    
    # 4.4 Retrieve top-k documents along with their metadata
    top_documents = similarities[:top_k]
    
    return top_documents

# 5. Example: Query to search for documents
query = "What is machine learning?"

# 6. Retrieve top 2 documents based on metadata and relevance
top_k_documents = retrieve_documents(query, documents, top_k=2)

# 7. Display the results
print(f"Top {len(top_k_documents)} documents for query '{query}':\n")
for idx, (score, doc) in enumerate(top_k_documents, 1):
    print(f"Rank {idx}:")
    print(f"  Title: {doc['title']}")
    print(f"  Author: {doc['author']}")
    print(f"  Date: {doc['date']}")
    print(f"  Relevance Score: {score:.4f}")
    print(f"  Content: {doc['content']}\n")


"""
Metadata Support: Each document includes additional metadata fields, like title, author, and date, which provide extra information about the document alongside the content.

Relevance-Based Retrieval: The system uses the cosine similarity between the query and document embeddings (using SentenceTransformers) to measure relevance. The higher the cosine similarity, the more relevant the document is to the query.

Content Embeddings: Each document's content is transformed into a vector (embedding) using a pre-trained SentenceTransformer model. This vector representation helps capture the semantic meaning of the document for better similarity matching.

Ranked Results: The retrieved documents are ranked by their relevance score, and the top k most relevant documents are returned. These documents are displayed alongside their metadata, which helps users better understand the context of the results.
"""

"""
This Python script demonstrates a document retrieval system based on the relevance of documents to a query using the Sentence-Transformer model and cosine similarity.

1. **Document Preparation**: 
   - A list of documents is prepared where each document contains `content`, `title`, `author`, and `date` metadata. This metadata helps provide additional context about each document.

2. **Sentence Transformer Model**:
   - The `SentenceTransformer` model (`all-MiniLM-L6-v2`) is used to generate embeddings (vector representations) for both the query and the documents. The model is pre-trained to generate semantically meaningful embeddings for text.

3. **Cosine Similarity Calculation**:
   - The `cosine_similarity_score()` function calculates the similarity between the query and each document. The cosine similarity score ranges from -1 to 1, where 1 means the documents are identical and 0 means no similarity.

4. **Document Retrieval**:
   - The `retrieve_documents()` function takes a query and a list of documents, calculates the similarity score for each document, sorts the documents by their relevance score in descending order, and retrieves the top `k` most relevant documents.
   - The function returns the top `k` documents along with their metadata.

5. **Example**:
   - The script includes an example query, "What is machine learning?", and retrieves the top 2 most relevant documents. The results are displayed with the document title, author, date, relevance score, and content.

This code is useful for implementing a basic document retrieval system that considers both content relevance and document metadata for more informative search results.
"""
