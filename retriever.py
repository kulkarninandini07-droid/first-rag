import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

def load_index_and_chunks(data_folder="data"):
    index = faiss.read_index(f"{data_folder}/faiss.index")
    with open(f"{data_folder}/chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks

def retrieve(query, model, index, chunks, k=5, company_filter=None):
    query_vec = model.encode([query])
    search_k = max(k * 6, 30)
    distances, indices = index.search(np.array(query_vec, dtype=np.float32), search_k)

    import re
    query_lower = query.lower()
    query_words = [re.sub(r"[^a-z0-9]", "", w.lower()) for w in query.split()]
    query_words = [w for w in query_words if len(w) > 3]

    # NEW: detect important 2-3 word phrases from the query for exact phrase matching
    # e.g. "net interest margin" should match as a PHRASE, not just separate words
    stopwords = {"what", "is", "the", "of", "for", "how", "does", "did"}
    meaningful_words = [w for w in query_words if w not in stopwords]
    key_phrase = " ".join(meaningful_words)  # e.g. "hdfc banks net interest margin"

    seen_ids = set()
    candidates = []

    def score_chunk(chunk):
        text_lower = chunk["text"].lower()
        word_hits = sum(1 for w in query_words if w in text_lower)
        # Big bonus if 3+ consecutive meaningful words appear together as a phrase
        phrase_bonus = 0
        for i in range(len(meaningful_words) - 1):
            two_word = f"{meaningful_words[i]} {meaningful_words[i+1]}"
            if two_word in text_lower:
                phrase_bonus += 5
        for i in range(len(meaningful_words) - 2):
            three_word = f"{meaningful_words[i]} {meaningful_words[i+1]} {meaningful_words[i+2]}"
            if three_word in text_lower:
                phrase_bonus += 10
        return word_hits + phrase_bonus

    for dist, idx in zip(distances[0], indices[0]):
        chunk = chunks[idx]
        if company_filter and chunk["company"].lower() != company_filter.lower():
            continue
        score = score_chunk(chunk)
        candidates.append({**chunk, "distance": float(dist), "keyword_hits": score})
        seen_ids.add(chunk["chunk_id"])

    if meaningful_words:
        for chunk in chunks:
            if chunk["chunk_id"] in seen_ids:
                continue
            if company_filter and chunk["company"].lower() != company_filter.lower():
                continue
            score = score_chunk(chunk)
            if score >= 5:  # only add if phrase match found
                candidates.append({**chunk, "distance": 1.0, "keyword_hits": score})
                seen_ids.add(chunk["chunk_id"])

    candidates.sort(key=lambda x: (-x["keyword_hits"], x["distance"]))
    return candidates[:k]

if __name__ == "__main__":
    print("Loading model and index...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index, chunks = load_index_and_chunks()
    print(f"✅ Loaded {len(chunks)} chunks\n")

    query = "What is HDFC Bank's net interest margin?"
    results = retrieve(query, model, index, chunks, k=8)
    for r in results:
        print(f"{r['company']} page {r['page']} - dist {r['distance']:.4f} - keyword hits: {r['keyword_hits']}")
        print(r['text'][:150])
        print()