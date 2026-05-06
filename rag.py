import pdfplumber
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

# ── Config ───────────────────────────────────────────
PDF_PATH = r"C:\Users\Admin\Downloads\module 2.pptx.pdf"
import os
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = "llama-3.1-8b-instant"

# ── Step 1: Load PDF ─────────────────────────────────
def load_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# ── Step 2: Chunk ────────────────────────────────────
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start+chunk_size])
        start += chunk_size - overlap
    return chunks

# ── Step 3: Embed + Index ────────────────────────────
def build_index(chunks, embed_model):
    print("Embedding chunks...")
    embeddings = embed_model.encode(chunks, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings

# ── Step 4: Retrieve ─────────────────────────────────
def retrieve(query, embed_model, index, chunks, k=3):
    query_vec = embed_model.encode([query])
    _, indices = index.search(np.array(query_vec), k)
    return [chunks[i] for i in indices[0]]

# ── Step 5: Generate answer ──────────────────────────
def generate_answer(query, context_chunks, groq_client):
    context = "\n\n---\n\n".join(context_chunks)
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {query}

Answer:"""

    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# ── Main: Put it all together ────────────────────────
def main():
    print("🔧 Loading models...")
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    groq_client = Groq(api_key=GROQ_API_KEY)

    print("📄 Loading PDF...")
    text = load_pdf(PDF_PATH)
    chunks = chunk_text(text)
    print(f"✅ {len(chunks)} chunks created")

    print("🗄️ Building FAISS index...")
    index, _ = build_index(chunks, embed_model)
    print("✅ Index ready!\n")

    # ── Chat loop ────────────────────────────────────
    print("💬 RAG Chatbot ready! Type 'quit' to exit.\n")
    print("=" * 50)

    while True:
        query = input("\nYou: ").strip()
        if query.lower() == "quit":
            print("Bye! 👋")
            break
        if not query:
            continue

        # Retrieve relevant chunks
        relevant_chunks = retrieve(query, embed_model, index, chunks)

        # Generate answer
        answer = generate_answer(query, relevant_chunks, groq_client)
        print(f"\n🤖 Bot: {answer}")
        print("-" * 50)

if __name__ == "__main__":
    main()