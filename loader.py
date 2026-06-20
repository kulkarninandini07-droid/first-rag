import pdfplumber
import os

COMPANY_MAP = {
    "annual_report_summary_db.pdf": "Deutche Bank"
}

def format_table(table):
    """Convert a pdfplumber table (list of rows) into readable text"""
    if not table:
        return ""
    lines = []
    for row in table:
        # Clean None values and join cells with " | "
        clean_row = [str(cell).strip() if cell else "" for cell in row]
        lines.append(" | ".join(clean_row))
    return "\n".join(lines)

def load_pdf_with_metadata(pdf_path):
    """Extract text AND tables from a PDF, page by page, with metadata"""
    filename = os.path.basename(pdf_path)
    company = COMPANY_MAP.get(filename, filename.replace(".pdf", "").title())
    
    pages_data = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            
            # Extract tables and convert to readable text
            tables = page.extract_tables()
            table_text = ""
            if tables:
                for table in tables:
                    table_text += "\n[TABLE]\n" + format_table(table) + "\n[/TABLE]\n"
            
            combined_text = text + "\n" + table_text
            
            if combined_text.strip():
                pages_data.append({
                    "text": combined_text,
                    "page": page_num,
                    "source": filename,
                    "company": company,
                    "has_table": bool(tables)
                })
    return pages_data

def load_all_pdfs(documents_folder="documents"):
    all_pages = []
    for filename in os.listdir(documents_folder):
        if filename.endswith(".pdf"):
            path = os.path.join(documents_folder, filename)
            print(f"Loading {filename}...")
            pages = load_pdf_with_metadata(path)
            all_pages.extend(pages)
            tables_found = sum(1 for p in pages if p["has_table"])
            print(f"  -> {len(pages)} pages extracted ({tables_found} with tables)")
    return all_pages

if __name__ == "__main__":
    all_pages = load_all_pdfs("documents")
    print(f"\n✅ Total pages loaded: {len(all_pages)}")
    
    # Show a page that has a table
    table_pages = [p for p in all_pages if p["has_table"]]
    if table_pages:
        print(f"\n--- Sample table page ---")
        print(f"Company: {table_pages[0]['company']}, Page: {table_pages[0]['page']}")
        print(table_pages[0]['text'][:500])