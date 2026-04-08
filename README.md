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

# Day 4 — HyDE (Hypothetical Document Embeddings)

## What this builds
Implements HyDE retrieval — generate a hypothetical answer,
embed that instead of the raw question, retrieve with a
richer, denser vector. Compares head-to-head against naive RAG.

## Why HyDE works
Questions and document chunks live in different parts of vector
space. A hypothetical answer uses the same vocabulary and
structure as real chunks — it lands geometrically closer to
the right content.

## Stack
Same as Phase 1 + no new dependencies

## Run order
1. python ingest.py      — rebuild ChromaDB for Day 4
2. python hyde.py        — HyDE retrieval with full output
3. python compare.py     — naive vs HyDE side by side

## Results
| Method    | Hit rate | Notes                                        |
|-----------|----------|----------------------------------------------|
| Naive RAG | 6/6      | Wins on precise terminology questions        |
| HyDE      | 6/6      | Wins on vague/short questions by distance    |

## When HyDE wins
- Vague questions where the raw question vector is weak
- Questions where vocabulary differs from document language

## When Naive wins
- Questions containing exact terminology from the document
- Short corpus where question-chunk vocabulary overlap is high

## HyDE failure mode
- LLM hypothesis drifts off topic → worse distance than naive
- Fix: tighter system prompt constraining hypothesis to exact topic

## Key insight
HyDE doesn't change your embedding model or vector DB.
It changes what you embed. That's the entire improvement.

## Key production lesson — Day 4
Q5 (layer normalisation) failed with both Naive and HyDE — dist 0.43+ for both.
Root cause: corpus coverage gap. Paper mentions normalisation only in passing.
Fix: added Ba et al. (2016) Layer Normalisation paper to document corpus.
Result: Q5 distance dropped significantly after re-ingest.

Rule: when both retrieval methods show high distance (>0.40),
the problem is corpus coverage — not retrieval technique.
Tune the corpus before tuning the retrieval.

# Day 5 — Cross-encoder reranking

## What this builds
Adds a cross-encoder reranker to the retrieval pipeline.
Retrieve top-10 by vector distance → rerank by relevance score → return top-3.
Three-way comparison: Naive vs HyDE vs Reranked.

## Why reranking works
Bi-encoder: embed question independently, embed chunk independently, compare.
Fast but approximate — doesn't see how question and chunk relate.

Cross-encoder: read question + chunk together as one input.
Scores true relevance. Slower but dramatically more precise.

## Stack
cross-encoder/ms-marco-MiniLM-L-6-v2 (new)
all others same as Phase 1

## Run order
1. python ingest.py    — rebuild collection for day5
2. python reranker.py  — full reranked RAG with output
3. python compare.py   — 3-way comparison table

## Results
Table 1 — Retrieval hit rate (does top-3 contain the right chunk?)
YES = keyword found in top-3 results · NO = missed · rank shown when all hit

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                       ┃              ┃ Naive      ┃ HyDE      ┃ Reranked     ┃                       ┃
┃ Question                              ┃ Keyword      ┃ top-3 hit  ┃ top-3 hit ┃ top-3 hit    ┃ Verdict               ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━┩
│ What is scaled dot-product attention? │ dot-product  │ YES (rank  │ YES (rank │ YES (rank 1) │ All retrieve          │
│                                       │              │ 1)         │ 1)        │              │ correctly             │
├───────────────────────────────────────┼──────────────┼────────────┼───────────┼──────────────┼───────────────────────┤
│ How does multi-head attention work?   │ head         │ YES (rank  │ YES (rank │ YES (rank 1) │ All retrieve          │
│                                       │              │ 1)         │ 1)        │              │ correctly             │
├───────────────────────────────────────┼──────────────┼────────────┼───────────┼──────────────┼───────────────────────┤
│ What is positional encoding?          │ position     │ YES (rank  │ YES (rank │ YES (rank 3) │ All retrieve          │
│                                       │              │ 1)         │ 1)        │              │ correctly             │
├───────────────────────────────────────┼──────────────┼────────────┼───────────┼──────────────┼───────────────────────┤
│ What optimizer and parameters were    │ Adam         │ YES (rank  │ YES (rank │ YES (rank 3) │ All retrieve          │
│ use                                   │              │ 3)         │ 2)        │              │ correctly             │
├───────────────────────────────────────┼──────────────┼────────────┼───────────┼──────────────┼───────────────────────┤
│ What is the role of layer             │ normalizati… │ YES (rank  │ NO        │ YES (rank 1) │ Mixed                 │
│ normalisatio                          │              │ 1)         │           │              │                       │
├───────────────────────────────────────┼──────────────┼────────────┼───────────┼──────────────┼───────────────────────┤
│ How does the encoder-decoder          │ encoder      │ YES (rank  │ YES (rank │ YES (rank 7) │ All retrieve          │
│ architect                             │              │ 1)         │ 1)        │              │ correctly             │
├───────────────────────────────────────┼──────────────┼────────────┼───────────┼──────────────┼───────────────────────┤
│ What were the BLEU scores on          │ BLEU         │ YES (rank  │ YES (rank │ YES (rank 1) │ All retrieve          │
│ translati                             │              │ 1)         │ 1)        │              │ correctly             │
├───────────────────────────────────────┼──────────────┼────────────┼───────────┼──────────────┼───────────────────────┤
│ Why does the model use residual       │ residual     │ YES (rank  │ YES (rank │ YES (rank 1) │ All retrieve          │
│ connec                                │              │ 1)         │ 1)        │              │ correctly             │
└───────────────────────────────────────┴──────────────┴────────────┴───────────┴──────────────┴───────────────────────┘

Hit rate summary:
  Naive RAG  : 8/8 questions retrieved correctly in top-3
  HyDE       : 7/8 questions retrieved correctly in top-3
  Reranked   : 8/8 questions retrieved correctly in top-3

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

Table 2 — Rank movement after reranking
Shows where the correct chunk ranked BEFORE reranking vs AFTER.
A jump from rank 5 → rank 1 is the reranker doing its job.

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃                                        ┃              ┃ Rank before    ┃ Rank after   ┃ Rerank     ┃                 ┃
┃ Question                               ┃ Keyword      ┃ reranking      ┃ reranking    ┃ score      ┃ Movement        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ What is scaled dot-product attention?  │ dot-product  │ #1             │ #1           │ +8.06      │ No change       │
├────────────────────────────────────────┼──────────────┼────────────────┼──────────────┼────────────┼─────────────────┤
│ How does multi-head attention work?    │ head         │ #1             │ #1           │ +7.98      │ No change       │
├────────────────────────────────────────┼──────────────┼────────────────┼──────────────┼────────────┼─────────────────┤
│ What is positional encoding?           │ position     │ #1             │ #3           │ +6.43      │ ↓ -2 positions  │
├────────────────────────────────────────┼──────────────┼────────────────┼──────────────┼────────────┼─────────────────┤
│ What optimizer and parameters were use │ Adam         │ #3             │ #3           │ +0.40      │ No change       │
├────────────────────────────────────────┼──────────────┼────────────────┼──────────────┼────────────┼─────────────────┤
│ What is the role of layer normalisatio │ normalizati… │ #1             │ #1           │ +2.72      │ No change       │
├────────────────────────────────────────┼──────────────┼────────────────┼──────────────┼────────────┼─────────────────┤
│ How does the encoder-decoder architect │ encoder      │ #1             │ #7           │ +6.17      │ ↓ -6 positions  │
├────────────────────────────────────────┼──────────────┼────────────────┼──────────────┼────────────┼─────────────────┤
│ What were the BLEU scores on translati │ BLEU         │ #1             │ #1           │ -1.47      │ No change       │
├────────────────────────────────────────┼──────────────┼────────────────┼──────────────┼────────────┼─────────────────┤
│ Why does the model use residual connec │ residual     │ #1             │ #1           │ +6.71      │ No change       │
└────────────────────────────────────────┴──────────────┴────────────────┴──────────────┴────────────┴─────────────────┘

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

Phase progression:
  Phase 1 baseline (naive, easy 6Q)  : 6/6   = 100%
  Day 5 naive      (harder 8Q)        : 8/8
  Day 5 HyDE       (harder 8Q)        : 7/8
  Day 5 reranked   (harder 8Q)        : 8/8
