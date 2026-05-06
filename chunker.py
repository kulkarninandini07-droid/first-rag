import pdfplumber

def load_pdf(pdf_path):
    """Extract all text from a PDF file"""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:  # some pages might be empty
                text += page_text + "\n"
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # overlap with next chunk
    
    return chunks

# ── Test it ──────────────────────────────────────────
pdf_path = r"C:\Users\Admin\Downloads\module 2.pptx.pdf"  # 👈 change this to your PDF filename

print("Loading PDF...")
text = load_pdf(pdf_path)
print(f"✅ Total characters extracted: {len(text)}")

print("\nChunking text...")
chunks = chunk_text(text, chunk_size=500, overlap=50)
print(f"✅ Total chunks created: {len(chunks)}")

print("\n--- Preview of first 3 chunks ---")
for i, chunk in enumerate(chunks[:3]):
    print(f"\nChunk {i+1} ({len(chunk)} chars):")
    print(chunk)
    print("-" * 40)