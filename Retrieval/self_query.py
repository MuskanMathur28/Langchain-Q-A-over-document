from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Define a simple knowledge base (list of documents)
documents = [
    {"content": "Python is a high-level programming language.", "title": "Python Programming", "author": "John Doe", "date": "2023-05-10"},
    {"content": "Machine learning is a method of data analysis that automates analytical model building.", "title": "Machine Learning", "author": "Jane Smith", "date": "2023-06-15"},
    {"content": "Deep learning is a subset of machine learning that uses neural networks with many layers.", "title": "Deep Learning", "author": "Alice Johnson", "date": "2023-07-20"},
]

# Step 2: Initialize the model for encoding text
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to compute cosine similarity
def cosine_similarity_score(vec1, vec2):
    return cosine_similarity([vec1], [vec2])[0][0]

# Step 3: Function for self-query retrieval based on query and knowledge base
def self_query(query, documents, top_k=3):
    # Step 4: Generate embeddings for the query and all documents
    query_embedding = model.encode([query])[0]
    document_embeddings = [model.encode([doc["content"]])[0] for doc in documents]

    # Step 5: Calculate cosine similarity between the query and each document
    similarities = []
    for i, doc_embedding in enumerate(document_embeddings):
        relevance_score = cosine_similarity_score(query_embedding, doc_embedding)
        similarities.append((relevance_score, documents[i]))

    # Step 6: Sort documents by relevance score (descending)
    similarities.sort(key=lambda x: x[0], reverse=True)

    # Step 7: Return the top-k relevant documents
    top_documents = similarities[:top_k]

    return top_documents

# Step 8: Example: Use self-query to search for relevant information
query = "What is machine learning?"

top_k_documents = self_query(query, documents, top_k=2)

# Step 9: Display the top-k results
print(f"Top {len(top_k_documents)} documents for query '{query}':\n")
for idx, (score, doc) in enumerate(top_k_documents, 1):
    print(f"Rank {idx}:")
    print(f"  Title: {doc['title']}")
    print(f"  Author: {doc['author']}")
    print(f"  Date: {doc['date']}")
    print(f"  Relevance Score: {score:.4f}")
    print(f"  Content: {doc['content']}\n")


"""
Self-query focuses on retrieving documents based purely on the content of the query and the documents themselves, often in a self-contained manner without relying on external context.
"""
