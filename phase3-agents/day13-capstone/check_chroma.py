# phase3-agents/day13-capstone/check_chroma.py
import chromadb
from pathlib import Path

# Try common locations — update path after Step 1
paths_to_check = [
    Path("C:/Raghu/ai-engineering-journey/phase2-advanced-rag/day9-capstone/chroma_db"),
    Path("C:/Raghu/ai-engineering-journey/phase2-advanced-rag/chroma_db"),
    Path("C:/Raghu/ai-engineering-journey/phase1-foundations/chroma_db"),
]

for p in paths_to_check:
    if p.exists():
        print(f"\nFound DB at: {p}")
        client = chromadb.PersistentClient(path=str(p))
        collections = client.list_collections()
        print(f"Collections: {[c.name for c in collections]}")
    else:
        print(f"Not found: {p}")