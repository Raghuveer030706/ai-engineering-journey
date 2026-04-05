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

# Day 3 — Real Documents + Semantic Chunking

## What this builds
Loads real PDF and TXT files, chunks them semantically
(by meaning shifts, not character count), embeds and stores
in ChromaDB, runs RAG pipeline on real content.

## Stack
- pypdf (PDF text extraction)
- nltk (sentence tokenization)
- langchain-text-splitters (fixed chunking for comparison)
- sentence-transformers (semantic chunking + embeddings)
- ChromaDB (vector store)
- Anthropic claude-haiku (generation)

## Run order
1. Add PDF and TXT files to ./documents/
2. python loader.py        -- verify files load correctly
3. python chunker.py       -- compare fixed vs semantic chunking
4. python ingest.py        -- embed and store all chunks
5. python rag_day3.py      -- RAG on real documents

## Key observation
Fixed chunking: cuts mid-sentence, destroys context
Semantic chunking: detects meaning shifts, preserves thoughts

## Chunk counts
Document              | Raw chunks | After filter
Attention Is All You Need (PDF) | 180 | 104
Avg chunk length: 247 chars
Shortest chunk: 110 chars

## Key lesson
Spent more time cleaning data than writing RAG code.
That's production. Garbage in, garbage out.

## Eval results
Retrieval hit rate: 6/6 = 100% on real documents

## Production insight observed
Two questions retrieved from my_notes.txt instead of the paper
despite the paper being the authoritative source.
This is source competition -- multiple docs covering the same topic.
Phase 2 will address this with metadata filtering and reranking.