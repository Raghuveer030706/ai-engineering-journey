# Day 1 — Embeddings & Semantic Search

## What this builds
Embeds 12 text chunks using sentence-transformers, stores vectors
in ChromaDB on disk, and retrieves the most semantically relevant
chunks for a query. No LLM. Pure retrieval. This is the R in RAG.

## Stack
- sentence-transformers (all-MiniLM-L6-v2, 384 dimensions)
- ChromaDB (local persistent vector store)
- Python 3.11 via conda environment (ai-journey)

## How to run
conda activate ai-journey
python search.py
python inspect_embeddings.py

## Key results
Query: "how do neural networks learn from data?"
Top hit: "Machine learning enables systems to learn patterns from data."

Cosine similarity between paraphrases : ~0.85
Cosine similarity between unrelated   : ~0.02

# Day 2 — Naive RAG Pipeline

## What this builds
Full naive RAG: embed query → retrieve chunks → build prompt
→ call Claude → return grounded answer.
Includes a retrieval eval harness with hit rate scoring.

## Stack
- sentence-transformers (retrieval)
- ChromaDB (vector store)
- Anthropic claude-haiku (generation)
- python-dotenv (API key management)

## How to run
conda activate ai-journey
python rag.py        # full pipeline with 4 test questions
python eval.py       # retrieval eval harness -- saves your baseline score

## Baseline retrieval score
Hit rate: X/8 (fill in after running eval.py)

## Key insight
The LLM answer is only as good as the retrieved chunks.
Bad retrieval = bad answer even with a great LLM.
This is why Phase 2 focuses entirely on improving retrieval.

## Baseline retrieval score
Hit rate: 8/8 = 100% on 12-chunk corpus (naive RAG, Day 2)