import numpy as np
import faiss
import json
import os
from sentence_transformers import SentenceTransformer
from loader import load_all_pdfs
from finance_chunker import chunk_pages

def build_finance_index(documents_folder="documents", data_folder="data"):
    """Process all PDFs and build a FAISS index with metadata"""
    
    # Create data folder if it doesn't exist
    os.makedirs(data_folder, exist_ok=True)
    
    # Step 1: Load all PDFs
    print("📄 Loading PDFs...")
    pages = load_all_pdfs(documents_folder)
    print(f"✅ {len(pages)} pages loaded")
    
    # Step 2: Chunk with metadata
    print("\n✂️ Chunking...")
    chunks = chunk_pages(pages)
    print(f"✅ {len(chunks)} chunks created")
    
    # Step 3: Embed all chunks
    print("\n🔢 Embedding chunks (this will take a few minutes)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=64
    )
    print(f"✅ Embeddings shape: {embeddings.shape}")
    
    # Step 4: Build FAISS index
    print("\n🗄️ Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype=np.float32))
    print(f"✅ FAISS index built with {index.ntotal} vectors")
    
    # Step 5: Save everything to disk
    print("\n💾 Saving to disk...")
    
    # Save FAISS index
    faiss.write_index(index, os.path.join(data_folder, "faiss.index"))
    
    # Save chunks metadata (without embeddings — FAISS handles those)
    with open(os.path.join(data_folder, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Saved faiss.index and chunks.json to {data_folder}/")
    print(f"\n🎉 Ingestion complete! {len(chunks)} chunks indexed.")
    
    return index, chunks, model

if __name__ == "__main__":
    build_finance_index()