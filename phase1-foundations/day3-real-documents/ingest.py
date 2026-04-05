import os
import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

import chromadb
from sentence_transformers import SentenceTransformer
from loader import load_documents
from chunker import chunk_semantic
from rich import print
from rich.progress import track

def is_valid_chunk(chunk: str) -> bool:
    """
    Filter out chunks that are noise, not content.
    Returns False for chunks we don't want in ChromaDB.
    """
    import re
    text = chunk.strip()

    # Too short
    if len(text) < 60:
        return False

    # Starts with citation marker [15]
    if re.match(r'^\[\d+\]', text):
        return False

    # Starts with journal abbreviation ACL, NIPS, NAACL etc
    if re.match(r'^[A-Z]{2,6}[,.]', text):
        return False

    # Starts with "In Proc." or "In Advances" -- reference entries
    if re.match(r'^In (Proc\.|Advances|Proceedings)', text):
        return False

    # Starts with XML/model tokens like <EOS> <pad>
    if re.match(r'^<[A-Za-z]+>', text):
        return False

    # Incomplete list items -- header + lone number with no content
    if re.search(r'\d+\.\s*$', text):
        return False

    # Contains 2+ citation markers
    if len(re.findall(r'\[\d+\]', text)) >= 2:
        return False

    # Contains reference patterns mid-text
    if re.search(r'\bIn [A-Z]{2,6},|\barXiv preprint|\bProceedings of\b', text):
        return False

    # Hyphenated line breaks from PDF extraction
    if re.search(r'\w+-\n\w+', text):
        return False

    # Mostly non-alphabetic
    alpha_ratio = sum(c.isalpha() for c in text) / len(text)
    if alpha_ratio < 0.5:
        return False

    return True

def ingest(documents_folder: str, chroma_path: str, collection_name: str):
    """
    Full ingest pipeline:
    Load documents → semantic chunk → embed → store in ChromaDB
    """
    print("[bold cyan]Starting ingest pipeline...[/bold cyan]\n")

    # Load
    docs = load_documents(documents_folder)
    if not docs:
        print("[red]No documents found. Add PDF or TXT files to ./documents[/red]")
        return

    # Embed model
    print("\n[cyan]Loading embedding model...[/cyan]")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # ChromaDB
    client     = chromadb.PersistentClient(path=chroma_path)

    # Delete existing collection to start fresh
    try:
        client.delete_collection(collection_name)
        print(f"[yellow]Cleared existing collection: {collection_name}[/yellow]")
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    # Chunk + embed + store
    all_chunks = []
    all_ids    = []
    all_metas  = []

    chunk_id = 0
    for doc in docs:
        print(f"\n[bold]Chunking:[/bold] {doc['filename']}")
        chunks = chunk_semantic(doc["text"], model)
        print(f"  {len(chunks)} semantic chunks created")

        for chunk in chunks:
            if not is_valid_chunk(chunk):
                continue
            all_chunks.append(chunk)
            all_ids.append(f"chunk_{chunk_id}")
            all_metas.append({
                "source":   doc["filename"],
                "type":     doc["type"],
                "chunk_id": chunk_id,
            })
            chunk_id += 1

    print(f"\n[bold]Total chunks to embed:[/bold] {len(all_chunks)}")

    # Embed in batches of 64
    batch_size = 64
    all_embeddings = []
    for i in track(range(0, len(all_chunks), batch_size), description="Embedding..."):
        batch = all_chunks[i:i+batch_size]
        embeddings = model.encode(batch, show_progress_bar=False).tolist()
        all_embeddings.extend(embeddings)

    # Store
    collection.add(
        ids=all_ids,
        embeddings=all_embeddings,
        documents=all_chunks,
        metadatas=all_metas,
    )

    print(f"\n[bold green]✓ Ingest complete.[/bold green]")
    print(f"  Documents loaded : {len(docs)}")
    print(f"  Total chunks     : {len(all_chunks)}")
    print(f"  Collection       : {collection_name}")
    print(f"  ChromaDB path    : {chroma_path}")

    return collection


if __name__ == "__main__":
    ingest(
        documents_folder="./documents",
        chroma_path="./chroma_db",
        collection_name="day3",
    )