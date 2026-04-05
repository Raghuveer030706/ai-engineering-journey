import os
import re
import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

from pypdf import PdfReader
from rich import print

def clean_text(text: str) -> str:
    """Remove noise common in PDF extraction."""
    # Remove excessive whitespace and newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    # Remove page numbers (standalone digits on a line)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    return text.strip()

def load_pdf(filepath: str) -> str:
    """Extract text from all pages of a PDF."""
    reader = PdfReader(filepath)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and len(text.strip()) > 50:  # skip blank/near-blank pages
            pages.append(text)
    full_text = "\n\n".join(pages)
    return clean_text(full_text)

def load_txt(filepath: str) -> str:
    """Load plain text file."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return clean_text(f.read())

def load_documents(folder: str) -> list[dict]:
    """
    Load all PDF and TXT files from a folder.
    Returns list of {filename, text, type} dicts.
    """
    docs = []
    supported = {".pdf": load_pdf, ".txt": load_txt}

    for filename in os.listdir(folder):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in supported:
            continue

        filepath = os.path.join(folder, filename)
        print(f"[cyan]Loading:[/cyan] {filename}")

        try:
            text = supported[ext](filepath)
            word_count = len(text.split())
            print(f"  [green]✓[/green] {word_count:,} words extracted")
            docs.append({
                "filename": filename,
                "text": text,
                "type": ext.strip(".")
            })
        except Exception as e:
            print(f"  [red]✗ Failed: {e}[/red]")

    return docs


if __name__ == "__main__":
    docs = load_documents("./documents")
    print(f"\n[bold]Loaded {len(docs)} document(s)[/bold]")
    for doc in docs:
        print(f"\n--- {doc['filename']} ---")
        print(f"First 300 chars:\n{doc['text'][:300]}")