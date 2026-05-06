from sentence_transformers import SentenceTransformer
import numpy as np

# Step 1 — Load the embedding model
# This converts text → vectors of numbers
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 2 — Some sample sentences (your "documents")
sentences = [
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks with many layers",
    "Python is a popular programming language for data science",
    "PyTorch is a deep learning framework developed by Meta",
    "cricket is the most popular sport in India",
    "Virat Kohli is a famous Indian cricketer",
]

# Step 3 — Convert sentences to embeddings
# Each sentence becomes a vector of 384 numbers
embeddings = model.encode(sentences)

print(f"Number of sentences: {len(sentences)}")
print(f"Each embedding shape: {embeddings[0].shape}")
print(f"Each sentence is now {embeddings[0].shape[0]} numbers!\n")

# Step 4 — Define cosine similarity from scratch
# This measures how similar two vectors are (0 = different, 1 = identical)
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    magnitude = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot_product / magnitude

# Step 5 — Ask a question and find most similar sentence
query = "Who plays cricket in india?"
query_embedding = model.encode([query])[0]

print(f"Query: '{query}'")
print("-" * 50)

# Compare query to every sentence
scores = []
for i, sentence in enumerate(sentences):
    score = cosine_similarity(query_embedding, embeddings[i])
    scores.append((score, sentence))

# Sort by similarity score
scores.sort(reverse=True)

print("Results (most similar to least):")
for rank, (score, sentence) in enumerate(scores):
    print(f"{rank+1}. Score: {score:.4f} → {sentence}")