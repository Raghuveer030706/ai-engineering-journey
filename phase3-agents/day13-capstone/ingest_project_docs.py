# phase3-agents/day13-capstone/ingest_project_docs.py
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

# Resolve path relative to this file — works on any machine
CHROMA_PATH = Path(__file__).parent.parent.parent / "phase2-advanced-rag" / "day9-capstone" / "chroma_db"
COLLECTION_NAME = "day9"

PROJECT_DOCS = [
    {
        "id": "proj_day3_chunks",
        "text": (
            "Day 3 real documents chunking results: started with 180 raw chunks, "
            "reduced to 124 after pass 1, 104 final clean chunks after pass 2 and 3. "
            "42.2% of chunks were removed from the original 180."
        ),
    },
    {
        "id": "proj_ragas_scores",
        "text": (
            "RAGAS evaluation scores by day: Day 6 baseline overall 0.638, "
            "Day 7 hybrid retrieval overall 0.807, Day 8 query expansion overall 0.702, "
            "Day 9 capstone overall 0.827. Phase 1 baseline is 0.638. Phase 2 final is 0.827. "
            "Total absolute improvement from Phase 1 to Phase 2 is 0.189."
        ),
    },
    {
        "id": "proj_ragas_metrics_day9",
        "text": (
            "Day 9 final RAGAS metrics breakdown: Faithfulness 0.924, "
            "Answer relevancy 0.926, Context precision 0.709, "
            "Context recall 0.750, Overall average 0.827."
        ),
    },
    {
        "id": "proj_ragas_metrics_day6",
        "text": (
            "Day 6 RAGAS baseline metrics breakdown: Faithfulness 0.896, "
            "Answer relevancy 0.741, Context precision 0.667, "
            "Context recall 0.250, Overall average 0.638."
        ),
    },
    {
        "id": "proj_phase2_gain",
        "text": (
            "Phase 2 RAGAS improvement over Phase 1 baseline: "
            "Phase 1 baseline score 0.638, Phase 2 final score 0.827, "
            "absolute gain 0.189, percentage gain 29.6 percent. "
            "Biggest single gain was hybrid retrieval on Day 7 which improved "
            "context recall from 0.25 to 0.75."
        ),
    },
    {
        "id": "proj_cross_encoder",
        "text": (
            "Cross-encoder reranking built on Day 5: retrieve top-10 candidates "
            "by vector distance using bi-encoder, then rerank using "
            "ms-marco-MiniLM-L-6-v2 cross-encoder which reads query and chunk "
            "together as one input to score true relevance. Return top-3 after reranking. "
            "Threshold -2.0 used for academic papers with mathematical notation. "
            "Bi-encoder embeds query and chunk independently. "
            "Cross-encoder reads the pair together — slower but more precise."
        ),
    },
    {
        "id": "proj_hyde",
        "text": (
            "HyDE (Hypothetical Document Embeddings) built on Day 4: generate a "
            "hypothetical answer to the query using an LLM, then embed that answer "
            "instead of the raw question. Wins on vague queries where question "
            "vocabulary differs from document language. "
            "Naive RAG wins on precise terminology questions. "
            "HyDE failure mode: LLM hypothesis drifts off topic producing worse distance."
        ),
    },
    {
        "id": "proj_hybrid_retrieval",
        "text": (
            "Hybrid retrieval built on Day 7: combines Naive RAG top-10 and HyDE top-10 "
            "into one candidate pool, deduplicates, then reranks combined set. "
            "Targets context recall gap from Day 6 which was 0.25. "
            "Result: recall improved from 0.25 to 0.75, overall score from 0.638 to 0.807. "
            "Largest single gain in Phase 2."
        ),
    },
    {
        "id": "proj_semantic_chunking",
        "text": (
            "Semantic chunking built on Day 3: detects cosine similarity drop between "
            "consecutive sentences and splits there rather than splitting every N characters. "
            "Similarity threshold 0.45 controls when a new chunk starts. "
            "Preserves complete thoughts unlike fixed chunking which cuts mid-sentence. "
            "Result on Attention Is All You Need paper: 180 raw chunks reduced to 104 clean chunks."
        ),
    },
    {
        "id": "proj_day10_agent",
        "text": (
            "Day 10 ReAct agent built from scratch with no frameworks. "
            "Three tools: calculator, dictionary, rag_search. "
            "Verified trace: 3 tools used, 4 steps, 5 LLM calls. "
            "Agent correctly inferred 8 heads times 64 dims equals 512 equals dmodel "
            "from RAG results without being told to make that connection. "
            "Model used: claude-haiku-4-5-20251001."
        ),
    },
    {
        "id": "proj_day11_memory",
        "text": (
            "Day 11 memory system: two-tier architecture. "
            "Short-term memory uses Python dict, lives in RAM, cleared on exit. "
            "Long-term memory uses SQLite, survives across sessions. "
            "Every entry carries a confidence float 0.0 to 1.0. "
            "Confidence degrades when RAG corpus contradicts a stored fact. "
            "Entries below 0.6 threshold flagged with warning in system prompt. "
            "retrieve() checks short-term first then falls back to long-term."
        ),
    },
    {
        "id": "proj_selective_expansion",
        "text": (
            "Day 9 selective query expansion: triggers expansion only when naive "
            "retrieval distance is greater than 0.35. "
            "Day 8 showed expansion hurts precision when naive retrieval is already "
            "confident. Selective triggering preserves Day 7 precision while recovering "
            "Day 8 recall gains. Final Day 9 score 0.827 versus Day 8 score 0.702."
        ),
    },
]


def ingest():
    if not CHROMA_PATH.exists():
        print(f"ERROR: ChromaDB not found at {CHROMA_PATH}")
        print("Make sure you have run the Day 9 ingest pipeline first.")
        return

    print(f"Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Connecting to ChromaDB at {CHROMA_PATH}")
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    try:
        collection = client.get_collection(COLLECTION_NAME)
        print(f"Found collection '{COLLECTION_NAME}' with {collection.count()} existing chunks")
    except Exception as e:
        print(f"ERROR: Could not get collection '{COLLECTION_NAME}': {e}")
        print(f"Available collections: {[c.name for c in client.list_collections()]}")
        return

    print(f"Embedding {len(PROJECT_DOCS)} project fact chunks...")
    texts = [d["text"] for d in PROJECT_DOCS]
    ids = [d["id"] for d in PROJECT_DOCS]
    embeddings = model.encode(texts).tolist()

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"source": "project_notes"} for _ in PROJECT_DOCS],
    )

    print(f"Ingested {len(PROJECT_DOCS)} project fact chunks into '{COLLECTION_NAME}'")
    print(f"Collection now has {collection.count()} total chunks")

    # Quick verification
    print(f"\nVerification — querying 'RAGAS scores Phase 1 Phase 2':")
    results = collection.query(
        query_texts=["RAGAS scores Phase 1 Phase 2"],
        n_results=3
    )
    for i, doc in enumerate(results["documents"][0]):
        print(f"  [{i+1}] {doc[:100]}...")


if __name__ == "__main__":
    ingest()