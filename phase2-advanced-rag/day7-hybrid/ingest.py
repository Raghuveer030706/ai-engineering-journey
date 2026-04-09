import os, re, logging, numpy as np
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
import nltk, chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from rich import print
from rich.progress import track

nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)

def clean_text(text):
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    return text.strip()

def load_pdf(path):
    reader = PdfReader(path)
    return clean_text("\n\n".join(
        p.extract_text() for p in reader.pages if p.extract_text()
    ))

def load_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return clean_text(f.read())

def chunk_semantic(text, model, threshold=0.45, max_chars=800):
    sentences  = nltk.sent_tokenize(text)
    if len(sentences) <= 1: return sentences
    embeddings = model.encode(sentences, show_progress_bar=False)
    chunks, current = [], [sentences[0]]
    for i in range(1, len(sentences)):
        a, b = embeddings[i-1], embeddings[i]
        sim  = float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
        if sim < threshold or len(" ".join(current)) > max_chars:
            chunks.append(" ".join(current))
            current = [sentences[i]]
        else:
            current.append(sentences[i])
    if current: chunks.append(" ".join(current))
    merged, buf = [], ""
    for c in chunks:
        buf = (buf + " " + c).strip()
        if len(buf) >= 100:
            merged.append(buf)
            buf = ""
    if buf: merged.append(buf)
    return merged

def is_valid(chunk):
    t = chunk.strip()
    if len(t) < 60:                                          return False
    if re.match(r'^\[\d+\]', t):                            return False
    if re.match(r'^[A-Z]{2,6}[,.]', t):                    return False
    if re.match(r'^In (Proc\.|Advances|Proceedings)', t):   return False
    if re.match(r'^<[A-Za-z]+>', t):                        return False
    if re.search(r'\d+\.\s*$', t):                          return False
    if len(re.findall(r'\[\d+\]', t)) >= 2:                return False
    if re.search(r'\bIn [A-Z]{2,6},|\barXiv preprint', t): return False
    if re.search(r'\w+-\n\w+', t):                          return False
    if sum(c.isalpha() for c in t)/len(t) < 0.5:           return False
    return True

def ingest(docs_folder="./documents", chroma_path="./chroma_db", col_name="day7"):
    print("[bold cyan]Day 7 ingest...[/bold cyan]\n")
    model  = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=chroma_path)
    try:    client.delete_collection(col_name)
    except: pass
    col = client.create_collection(name=col_name, metadata={"hnsw:space":"cosine"})
    all_chunks, all_ids, all_metas = [], [], []
    chunk_id = 0
    for fname in os.listdir(docs_folder):
        ext  = os.path.splitext(fname)[1].lower()
        path = os.path.join(docs_folder, fname)
        if ext == ".pdf":   text = load_pdf(path)
        elif ext == ".txt": text = load_txt(path)
        else: continue
        print(f"[cyan]Loaded:[/cyan] {fname}")
        raw    = chunk_semantic(text, model)
        chunks = [c for c in raw if is_valid(c)]
        print(f"  {len(raw)} raw → {len(chunks)} clean")
        for chunk in chunks:
            all_chunks.append(chunk)
            all_ids.append(f"chunk_{chunk_id}")
            all_metas.append({"source": fname, "chunk_id": chunk_id})
            chunk_id += 1
    print(f"\n[bold]Embedding {len(all_chunks)} chunks...[/bold]")
    embeddings = []
    for i in track(range(0, len(all_chunks), 64), description="Embedding..."):
        batch = all_chunks[i:i+64]
        embeddings.extend(model.encode(batch, show_progress_bar=False).tolist())
    col.add(ids=all_ids, embeddings=embeddings,
            documents=all_chunks, metadatas=all_metas)
    print(f"\n[bold green]✓ {len(all_chunks)} chunks stored in '{col_name}'[/bold green]")

if __name__ == "__main__":
    ingest()