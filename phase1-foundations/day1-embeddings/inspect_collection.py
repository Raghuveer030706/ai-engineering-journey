import chromadb
from rich import print
from rich.table import Table

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="day1",
    metadata={"hnsw:space": "cosine"}
)

# Fetch everything stored in the collection
results = collection.get(include=["documents", "embeddings"])

print(f"\n[bold]Total vectors stored:[/bold] {len(results['ids'])}\n")

# Show documents and first 5 floats of each embedding vector
table = Table(show_header=True, header_style="bold")
table.add_column("ID",    width=5)
table.add_column("First 5 floats of vector",   width=45)
table.add_column("Document text")

for id_, emb, doc in zip(
    results["ids"],
    results["embeddings"],
    results["documents"]
):
    floats = [f"{x:.4f}" for x in emb[:5]]
    table.add_row(id_, str(floats), doc[:60])

print(table)