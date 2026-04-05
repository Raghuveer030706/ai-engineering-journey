import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

import nltk
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from rich import print
from rich.table import Table
import numpy as np

# --- Strategy 1: Naive fixed-size chunking ---
def chunk_fixed(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text every N characters with overlap.
    Simple. Fast. Breaks sentences and thoughts arbitrarily.
    This is what naive RAG uses.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(text)

# --- Strategy 2: Semantic chunking ---
def chunk_semantic(text: str, model: SentenceTransformer,
                   similarity_threshold: float = 0.45) -> list[str]:
    """
    Split text by detecting meaning shifts between sentences.
    When consecutive sentences become semantically dissimilar,
    start a new chunk. Preserves context that fixed chunking destroys.
    """
    # Split into sentences first
    sentences = nltk.sent_tokenize(text)
    if len(sentences) <= 1:
        return sentences

    # Embed all sentences at once
    embeddings = model.encode(sentences, show_progress_bar=False)

    # Find split points where meaning shifts
    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        # Cosine similarity between consecutive sentences
        a, b = embeddings[i-1], embeddings[i]
        sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

        if sim < similarity_threshold:
            # Meaning shifted -- start a new chunk
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
        else:
            # Same topic continuing -- keep building current chunk
            current_chunk.append(sentences[i])

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    # Merge chunks that are too small (under 100 chars)
    merged = []
    buffer = ""
    for chunk in chunks:
        buffer = (buffer + " " + chunk).strip()
        if len(buffer) >= 100:
            merged.append(buffer)
            buffer = ""
    if buffer:
        merged.append(buffer)

    return merged

def compare_strategies(text: str, model: SentenceTransformer):
    """Show fixed vs semantic chunking side by side on the same text."""
    fixed_chunks    = chunk_fixed(text)
    semantic_chunks = chunk_semantic(text, model)

    print("\n[bold]--- Chunking Strategy Comparison ---[/bold]\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Strategy",       width=16)
    table.add_column("Chunk count",    width=13)
    table.add_column("Avg length",     width=12)
    table.add_column("Shortest chunk", width=16)
    table.add_column("Longest chunk",  width=15)

    for name, chunks in [("Fixed (naive)", fixed_chunks), ("Semantic", semantic_chunks)]:
        lengths = [len(c) for c in chunks]
        table.add_row(
            name,
            str(len(chunks)),
            f"{int(sum(lengths)/len(lengths))} chars",
            f"{min(lengths)} chars",
            f"{max(lengths)} chars",
        )

    print(table)

    # Show a concrete example of the difference
    print("\n[bold yellow]Fixed chunk example (chunk 3):[/bold yellow]")
    if len(fixed_chunks) >= 3:
        print(f"  {fixed_chunks[2][:300]}")

    print("\n[bold green]Semantic chunk example (chunk 3):[/bold green]")
    if len(semantic_chunks) >= 3:
        print(f"  {semantic_chunks[2][:300]}")

    print("\n[bold]Key observation:[/bold]")
    print("Fixed chunks cut mid-sentence. Semantic chunks preserve complete thoughts.")

    return fixed_chunks, semantic_chunks


if __name__ == "__main__":
    from loader import load_documents
    model = SentenceTransformer("all-MiniLM-L6-v2")

    docs = load_documents("./documents")
    if not docs:
        print("[red]No documents found in ./documents folder[/red]")
        exit()

    # Use first document for comparison
    doc  = docs[0]
    text = doc["text"]
    print(f"\n[bold]Comparing chunking strategies on:[/bold] {doc['filename']}")
    print(f"Document length: {len(text):,} chars\n")

    compare_strategies(text, model)