import os
import sys
import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

import chromadb
from sentence_transformers import SentenceTransformer
from rich import print
from rich.progress import track
import re
import nltk
from pypdf import PdfReader
import numpy as np

nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)

# ── Loaders ───────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'http\S+', '', text)
    return text.strip()

def load_pdf(path: str) -> str:
    reader = PdfReader(path)
    pages  = [p.extract_text() for p in reader.pages if p.extract_text()]
    return clean_text("\n\n".join(pages))

def load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return clean_text(f.read())

def load_documents(folder: str) -> list[dict]:
    docs = []
    for fname in os.listdir(folder):
        ext  = os.path.splitext(fname)[1].lower()
        path = os.path.join(folder, fname)
        if ext == ".pdf":
            text = load_pdf(path)
        elif ext == ".txt":
            text = load_txt(path)
        else:
            continue
        print(f"[cyan]Loaded:[/cyan] {fname} ({len(text):,} chars)")
        docs.append({"filename": fname, "text": text, "type": ext.strip(".")})
    return docs

# ── Semantic chunker ──────────────────────────────────────────────────────────
def chunk_semantic(text: str, model: SentenceTransformer,
                   threshold: float = 0.45, max_chars: int = 800) -> list[str]:
    sentences  = nltk.sent_tokenize(text)
    if len(sentences) <= 1:
        return sentences
    embeddings = model.encode(sentences, show_progress_bar=False)
    chunks, current = [], [sentences[0]]

    for i in range(1, len(sentences)):
        a, b = embeddings[i-1], embeddings[i]
        sim  = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        current_text = " ".join(current)
        if sim < threshold or len(current_text) > max_chars:
            chunks.append(current_text)
            current = [sentences[i]]
        else:
            current.append(sentences[i])
    if current:
        chunks.append(" ".join(current))

    merged, buf = [], ""
    for c in chunks:
        buf = (buf + " " + c).strip()
        if len(buf) >= 100:
            merged.append(buf)
            buf = ""
    if buf:
        merged.append(buf)
    return merged

# ── Filter ────────────────────────────────────────────────────────────────────
def is_valid(chunk: str) -> bool:
    t = chunk.strip()
    if len(t) < 60:                                          return False
    if re.match(r'^\[\d+\]', t):                            return False
    if re.match(r'^[A-Z]{2,6}[,.]', t):                     return False
    if re.match(r'^In (Proc\.|Advances|Proceedings)', t):    return False
    if re.match(r'^<[A-Za-z]+>', t):                         return False
    if re.search(r'\d+\.\s*$', t):                          return False
    if len(re.findall(r'\[\d+\]', t)) >= 2:                 return False
    if re.search(r'\bIn [A-Z]{2,6},|\barXiv preprint', t):  return False
    if re.search(r'\w+-\n\w+', t):                          return False
    if sum(c.isalpha() for c in t) / len(t) < 0.5:         return False
    return True

# ── Main ingest ───────────────────────────────────────────────────────────────
def ingest(docs_folder: str = "./documents",
           chroma_path: str = "./chroma_db",
           collection_name: str = "day4"):

    print("[bold cyan]Day 4 ingest pipeline starting...[/bold cyan]\n")
    docs = load_documents(docs_folder)
    if not docs:
        print("[red]No documents found.[/red]")
        return

    model  = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=chroma_path)

    try:
        client.delete_collection(collection_name)
        print(f"[yellow]Cleared existing collection: {collection_name}[/yellow]")
    except Exception:
        pass

    col = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    all_chunks, all_ids, all_metas = [], [], []
    chunk_id = 0

    for doc in docs:
        raw    = chunk_semantic(doc["text"], model)
        chunks = [c for c in raw if is_valid(c)]
        print(f"[bold]{doc['filename']}:[/bold] {len(raw)} raw → {len(chunks)} clean chunks")

        for chunk in chunks:
            all_chunks.append(chunk)
            all_ids.append(f"chunk_{chunk_id}")
            all_metas.append({"source": doc["filename"], "type": doc["type"], "chunk_id": chunk_id})
            chunk_id += 1

    print(f"\n[bold]Embedding {len(all_chunks)} chunks...[/bold]")
    embeddings = []
    for i in track(range(0, len(all_chunks), 64), description="Embedding..."):
        batch = all_chunks[i:i+64]
        embeddings.extend(model.encode(batch, show_progress_bar=False).tolist())

    col.add(ids=all_ids, embeddings=embeddings, documents=all_chunks, metadatas=all_metas)

    print(f"\n[bold green]✓ Ingest complete — {len(all_chunks)} chunks stored[/bold green]")

if __name__ == "__main__":
    ingest()