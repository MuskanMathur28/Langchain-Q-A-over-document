import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

# Step 1: Load Pre-trained SentenceTransformer Model for Embedding
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Step 2: Function to Compute Cosine Similarity
def cosine_similarity(vec1, vec2):
    # Flatten the vectors to 1D
    vec1 = vec1.flatten()
    vec2 = vec2.flatten()
    return 1 - cosine(vec1, vec2)

# Step 3: Compute relevance of documents to the query (using embeddings)
def compute_relevance(query, document):
    query_embedding = model.encode([query])[0]  # Get the 1D embedding vector
    doc_embedding = model.encode([document])[0]  # Get the 1D embedding vector
    return cosine_similarity(query_embedding, doc_embedding)

# Step 4: Compute redundancy between documents
def compute_redundancy(doc1, doc2):
    doc1_embedding = model.encode([doc1])[0]  # Get the 1D embedding vector
    doc2_embedding = model.encode([doc2])[0]  # Get the 1D embedding vector
    return cosine_similarity(doc1_embedding, doc2_embedding)

# Step 5: MMR Implementation
def max_marginal_relevance(query, documents, lambda_param=0.5, top_k=5):
    selected_docs = []
    remaining_docs = documents.copy()

    while len(selected_docs) < top_k and remaining_docs:
        scores = []
        for doc in remaining_docs:
            # Calculate relevance and redundancy
            relevance = compute_relevance(query, doc)
            redundancy = 0
            for selected_doc in selected_docs:
                redundancy = max(redundancy, compute_redundancy(doc, selected_doc))
            
            # Apply MMR formula
            mmr_score = lambda_param * relevance - (1 - lambda_param) * redundancy
            scores.append((doc, mmr_score))
        
        # Select the document with the highest MMR score
        best_doc = max(scores, key=lambda x: x[1])[0]
        selected_docs.append(best_doc)
        remaining_docs.remove(best_doc)
    
    return selected_docs

# Step 6: Example Query and Documents
query = "What are the benefits of Maximum Marginal Relevance?"
documents = [
    "MMR is useful for balancing relevance and diversity in retrieval.",
    "Maximum Marginal Relevance ensures that retrieved documents are diverse.",
    "MMR is often used in information retrieval to avoid redundancy.",
    "Relevance scoring is essential in document retrieval tasks.",
    "Maximum Marginal Relevance combines relevance and diversity using a scoring mechanism."
]

# Step 7: Run MMR
top_documents = max_marginal_relevance(query, documents, lambda_param=0.5, top_k=3)

# Output the selected documents
print("Selected Documents:")
for doc in top_documents:
    print(f"- {doc}")





"""
MMR is commonly used in document retrieval, search engines, and question-answering systems to retrieve a set of documents that are not only highly relevant to the query but also diverse in their content. This implementation uses the Sentence-Transformer model to compute embeddings and the cosine similarity measure to calculate relevance and redundancy.
Relevance: How well a document answers or matches the query.
Redundancy: How similar a document is to the documents already selected.
MMR Score: A combination of relevance and redundancy, calculated using the formula:-
MMR(D)=λ⋅Relevance(D)−(1−λ)⋅Redundancy(D) ,
Where λ (lambda) is a hyperparameter that controls the balance between relevance and redundancy.

How It Works:
1. Embed the query and documents: Use the Sentence-Transformer model to convert the text into embeddings (numerical vectors).
2. Compute relevance: Calculate how relevant each document is to the query using cosine similarity.
3. Compute redundancy: Calculate how similar each document is to the documents that have already been selected.
4. Score documents using MMR: Combine relevance and redundancy to select the document with the highest MMR score.
5. Select top K documents: Repeat the process until the desired number of documents is retrieved.
"""
