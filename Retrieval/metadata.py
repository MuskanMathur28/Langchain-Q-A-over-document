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
