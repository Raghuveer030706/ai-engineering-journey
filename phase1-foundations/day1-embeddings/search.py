import chromadb
from sentence_transformers import SentenceTransformer
from rich import print
from rich.table import Table

# 1. Your text corpus -- 12 chunks covering the topics you will learn
DOCUMENTS = [
    {"id": "1",  "text": "Python is a high-level programming language known for readability."},
    {"id": "2",  "text": "Machine learning enables systems to learn patterns from data."},
    {"id": "3",  "text": "Transformers are a neural network architecture from the paper Attention Is All You Need."},
    {"id": "4",  "text": "Vector databases store embeddings and allow fast similarity search."},
    {"id": "5",  "text": "RAG stands for Retrieval-Augmented Generation -- search combined with LLMs."},
    {"id": "6",  "text": "ChromaDB is an open-source embedding database built for AI applications."},
    {"id": "7",  "text": "Cosine similarity measures the angle between two vectors in high-dimensional space."},
    {"id": "8",  "text": "Embeddings are dense numerical representations of text that capture meaning."},
    {"id": "9",  "text": "The attention mechanism lets models focus on relevant parts of the input."},
    {"id": "10", "text": "Fine-tuning adapts a pre-trained model to a specific task with less data."},
    {"id": "11", "text": "Agents use LLMs to decide which tools to call and in what sequence."},
    {"id": "12", "text": "MCP is the Model Context Protocol -- an open standard for connecting AI to tools."},
]

# 2. Load embedding model -- downloads ~90MB on first run, cached after
print("[bold cyan]Loading embedding model...[/bold cyan]")
model = SentenceTransformer("all-MiniLM-L6-v2")  # each sentence → 384 floats

# 3. ChromaDB -- stores vectors on disk in ./chroma_db folder
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="day1",
    metadata={"hnsw:space": "cosine"}
)

# 4. Ingest only if collection is empty -- safe to run multiple times
if collection.count() == 0:
    print(f"[yellow]Embedding and storing {len(DOCUMENTS)} documents...[/yellow]")
    texts      = [d["text"] for d in DOCUMENTS]
    ids        = [d["id"]   for d in DOCUMENTS]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()
    collection.add(ids=ids, embeddings=embeddings, documents=texts)
    print(f"[green]Stored {len(DOCUMENTS)} vectors in ChromaDB.[/green]")
else:
    print(f"[green]Collection already has {collection.count()} vectors. Skipping ingest.[/green]")

# 5. Three test queries -- watch which chunks come back for each
QUERIES = [
    "how do neural networks learn from data?",
    "what is the best way to store and search vectors?",
    "how do AI agents decide what actions to take?",
]

print("\n[bold]--- Semantic Search Results ---[/bold]\n")

for query in QUERIES:
    print(f"[bold magenta]Query:[/bold magenta] {query}")

    query_vec = model.encode([query]).tolist()
    results   = collection.query(query_embeddings=query_vec, n_results=3)

    table = Table(show_header=True, header_style="bold")
    table.add_column("Rank",     width=6)
    table.add_column("Distance", width=10)
    table.add_column("Retrieved text")

    for rank, (doc, dist) in enumerate(
        zip(results["documents"][0], results["distances"][0]), 1
    ):
        table.add_row(str(rank), f"{dist:.4f}", doc)

    print(table)
    print()