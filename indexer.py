import pdfplumber
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ── Step 1: Load PDF ─────────────────────────────────
def load_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# ── Step 2: Chunk text ───────────────────────────────
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

# ── Step 3: Embed chunks ─────────────────────────────
def embed_chunks(chunks):
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Embedding all chunks... (this may take a moment)")
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings, model

# ── Step 4: Store in FAISS ───────────────────────────
def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]  # 384
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index

# ── Step 5: Search function ──────────────────────────
def search(query, model, index, chunks, k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "rank": i + 1,
            "chunk": chunks[idx],
            "distance": distances[0][i]
        })
    return results

# ── Run everything ───────────────────────────────────
pdf_path = r"C:\Users\Admin\Downloads\module 2.pptx.pdf"  # your path

print("=" * 50)
print("Loading PDF...")
text = load_pdf(pdf_path)
print(f"✅ Characters extracted: {len(text)}")

print("\nChunking...")
chunks = chunk_text(text)
print(f"✅ Chunks created: {len(chunks)}")

print("\nEmbedding chunks...")
embeddings, model = embed_chunks(chunks)
print(f"✅ Embeddings shape: {embeddings.shape}")

print("\nBuilding FAISS index...")
index = build_faiss_index(embeddings)
print(f"✅ FAISS index built with {index.ntotal} vectors")

# ── Test a search ────────────────────────────────────
print("\n" + "=" * 50)
query = "what is the difference between a class and an object?"
print(f"Query: '{query}'")
print("=" * 50)

results = search(query, model, index, chunks, k=3)
for r in results:
    print(f"\n📄 Rank {r['rank']} (distance: {r['distance']:.4f}):")
    print(r['chunk'])
    print("-" * 40)