def chunk_pages(pages_data, chunk_size=500, overlap=50):
    """
    Chunk pages into smaller pieces, carrying metadata through each chunk
    """
    all_chunks = []

    for page in pages_data:
        text = page["text"]
        start = 0

        while start < len(text):
            chunk_text = text[start:start + chunk_size]

            # Skip empty or very short chunks (less than 50 chars — not useful)
            if len(chunk_text.strip()) > 50:
                all_chunks.append({
                    "text": chunk_text,
                    "company": page["company"],
                    "source": page["source"],
                    "page": page["page"],
                    "chunk_id": len(all_chunks)
                })

            start += chunk_size - overlap

    return all_chunks

# ── Test it ───────────────────────────────────────────
if __name__ == "__main__":
    from loader import load_all_pdfs

    print("Loading PDFs...")
    pages = load_all_pdfs("documents")

    print("Chunking...")
    chunks = chunk_pages(pages)

    print(f"\n✅ Total chunks: {len(chunks)}")

    # Show breakdown by company
    from collections import Counter
    company_counts = Counter(c["company"] for c in chunks)
    print(f"\nChunks per company:")
    for company, count in company_counts.items():
        print(f"  {company}: {count} chunks")

    print(f"\n--- Sample chunk ---")
    print(f"Company: {chunks[0]['company']}")
    print(f"Source: {chunks[0]['source']}")
    print(f"Page: {chunks[0]['page']}")
    print(f"Chunk ID: {chunks[0]['chunk_id']}")
    print(f"Text preview: {chunks[0]['text'][:200]}")