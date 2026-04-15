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

# Day 6 — RAGAS Evaluation

## What this measures
Four metrics your keyword harness cannot see:

| Metric             | What it catches                          |
|--------------------|------------------------------------------|
| Faithfulness       | Hallucination — claims not in context    |
| Answer relevancy   | Off-topic or incomplete answers          |
| Context precision  | Noisy retrieval — irrelevant chunks      |
| Context recall     | Missing chunks — incomplete context      |

## Stack
ragas, datasets, langchain-anthropic (new)
everything else same as Phase 2

## Run order
1. python ingest.py     — rebuild collection for day6
2. python ragas_eval.py — full RAGAS evaluation

## Day 6 RAGAS baseline scores
| Metric             | Score | Status        |
|--------------------|-------|---------------|
| Faithfulness       | 0.896 | Good          |
| Answer relevancy   | 0.741 | Needs work    |
| Context precision  | 0.667 | Needs work    |
| Context recall     | 0.250 | Critical gap  |
| Overall average    | 0.638 |               |

## Key finding
Context recall at 0.25 is the biggest gap.
Retrieval is incomplete — correct chunks exist but aren't all surfacing.
Fix: hybrid retrieval (Day 7) + wider n_wide.

## What each score means
Faithfulness 0.90 → Claude stays grounded in context 90% of the time.
Answer relevancy 0.74 → Q4 optimizer question pulling average down.
Context precision 0.67 → 1 in 3 retrieved chunks is noise.
Context recall 0.25 → missing 75% of what complete answers need.

# Day 7 — Hybrid Retrieval

## What this builds
Combines Naive + HyDE retrieval into one candidate pool,
deduplicates, then reranks the combined set.
Targets the context recall gap identified in Day 6 (0.25).

## Why hybrid works
Naive misses vague queries. HyDE occasionally drifts.
Combined pool covers both failure modes.
Chunks found by BOTH methods are the strongest signal.

## Architecture
Naive (top 10) ──┐
                  ├── merge + dedup ── rerank ── top 5
HyDE  (top 10) ──┘

## Run order
1. python ingest.py      — rebuild collection for day7
2. python hybrid.py      — test hybrid pipeline with output
3. python ragas_eval.py  — RAGAS comparison vs Day 6

## Results vs Day 6
| Metric             | Day 6  | Day 7  | Delta   |
|--------------------|--------|--------|---------|
| Faithfulness       | 0.896  | 0.814  | -0.082  |
| Answer relevancy   | 0.741  | 0.767  | +0.026  |
| Context precision  | 0.667  | 0.636  | -0.031  |
| Context recall     | 0.250  | 0.583  | +0.333  |
| Overall            | 0.638  | 0.700  | +0.062  |

## Key finding
Context recall improved +0.333 — the largest single gain in Phase 2.
Tradeoff: faithfulness dropped -0.082 as expected with larger context.
Precision-recall tradeoff is real and measurable.

## Production insight
Hybrid retrieval improves recall at a small faithfulness and
precision cost. In production, choose based on your priority:
- Patient safety / legal → maximise faithfulness, accept lower recall
- Research assistant → maximise recall, accept lower faithfulness

## Final Day 7 results after fixes
| Metric             | Day 6  | Day 7  | Delta   |
|--------------------|--------|--------|---------|
| Faithfulness       | 0.896  | 0.883  | -0.013  |
| Answer relevancy   | 0.741  | 0.873  | +0.132  |
| Context precision  | 0.667  | 0.723  | +0.056  |
| Context recall     | 0.250  | 0.750  | +0.500  |
| Overall            | 0.638  | 0.807  | +0.169  |

## What produced the +0.169 improvement
1. Hybrid retrieval (Naive + HyDE merged pool) → recall +0.500
2. Rerank threshold -2.0 → optimizer chunk now passes through
3. Duplicate chunk removal → cleaner context, less noise
4. Ground truth aligned to corpus language (Greek symbols)

## Key production lessons
- Never tune chunking to fix one question — it breaks others
- Rerank thresholds must be calibrated to corpus type
  - Natural language: 0.0
  - Academic papers with formulas: -1.5 to -2.0
- Duplicate documents silently double your chunk count
  and pollute retrieval with redundant candidates
- Ground truth language must match corpus language exactly


# Day 8 — Query Expansion

## What this builds
Generates multiple reformulations of each question, retrieves
candidates for each, merges all pools, deduplicates, reranks
with the original question. Targets remaining recall gaps
from Day 7 — specifically Q3 (positional encoding).

## Why query expansion works
Different vocabulary → different vector space regions →
different chunks retrieved. Chunks found by 2+ query
reformulations are the strongest retrieval signal.

## Architecture
Original query   ──┐
Expansion 1      ──┼── retrieve (5 each) ── merge+dedup ── rerank ── top 5
Expansion 2      ──┤
Expansion 3      ──┘

## Important distinction
Expansions are for RETRIEVAL DIVERSITY only.
Reranking always uses the ORIGINAL question.
Never rerank with expanded queries — it introduces drift.

## Run order
1. python ingest.py      — rebuild collection for day8
2. python expansion.py   — test with 4 questions, see expansions
3. python ragas_eval.py  — full comparison vs Day 6 and Day 7

## Results
| Metric             | Day 6  | Day 7  | Day 8  | Total gain |
|--------------------|--------|--------|--------|------------|
| Faithfulness       | 0.896  | 0.883  | fill   |            |
| Answer relevancy   | 0.741  | 0.873  | fill   |            |
| Context precision  | 0.667  | 0.723  | fill   |            |
| Context recall     | 0.250  | 0.750  | fill   |            |
| Overall            | 0.638  | 0.807  | fill   |            |

# Day 9 — Phase 2 Capstone

## What this builds
The complete Phase 2 production pipeline combining the best
elements from Days 4-8 into one clean, evaluated system.

## Pipeline decisions
- Hybrid retrieval (Day 7) as the primary strategy — best overall
- Selective query expansion (Day 8) only when naive distance > 0.35
- Cross-encoder reranking (Day 5) on combined candidate pool
- Threshold -2.0 to handle academic paper mathematical notation

## Why selective expansion
Day 8 showed expansion hurts precision when naive retrieval
is already confident (distance < 0.35). Expansion adds value
only when retrieval signal is weak. Selective triggering
preserves Day 7 precision while recovering Day 8 recall gains.

# Day 9 — Phase 2 Capstone — COMPLETE

## Final Phase 2 results
| Metric             | Day 6  | Day 7  | Day 8  | Day 9  | Total gain |
|--------------------|--------|--------|--------|--------|------------|
| Faithfulness       | 0.896  | 0.883  | 0.755  | 0.924  | +0.028     |
| Answer relevancy   | 0.741  | 0.873  | 0.772  | 0.926  | +0.185     |
| Context precision  | 0.667  | 0.723  | 0.531  | 0.709  | +0.042     |
| Context recall     | 0.250  | 0.750  | 0.750  | 0.750  | +0.500     |
| Overall average    | 0.638  | 0.807  | 0.702  | 0.827  | +0.189     |

## Phase 2 complete — 0.638 → 0.827

## What selective expansion proved
Triggering expansion only when naive distance > 0.35:
- Preserved Day 7 precision (0.709 vs 0.723)
- Maintained Day 7 recall (0.750)
- Improved faithfulness to 0.924 — best across all days
- Improved answer relevancy to 0.926 — best across all days

## Known limitation
Q3 (positional encoding) context recall = 0.00 across all days.
Naive retrieval confident (distance < 0.35) but ground truth
requires sinusoidal formula details not cleanly in any chunk.
Lower threshold (0.30) or corpus expansion would fix this.

# Day 10 — ReAct Agent From Scratch

## What this builds
A complete ReAct (Reasoning + Acting) loop in pure Python.
No LangChain. No LlamaIndex. No agent framework.
The LLM reasons, decides which tool to call, your code
executes it, the observation feeds back — repeat until done.

## Tools available
- calculator  — evaluates math expressions locally
- dictionary  — looks up AI/ML term definitions
- rag_search  — searches Phase 2 ChromaDB knowledge base

## The ReAct loop
Thought → Action → Observation → Thought → ... → Final Answer

## Why build without frameworks
Frameworks hide the loop. When something breaks in production
you need to know exactly which step failed — was it the
tool call, the parser, the prompt format, or the LLM output?
Building it yourself means you always know.

## Run order
1. python agent.py         — run 3 test questions, see full loop
2. python inspect_agent.py — step-by-step trace table

## Key numbers
LLM calls per question: 1 per step + 1 for forced final if max reached
Max steps default: 8
Tools: 3 (calculator, dictionary, rag_search)

## Verified trace results
| Step | Type       | Tool       | Result              |
|------|------------|------------|---------------------|
| 1    | TOOL       | dictionary | attention definition|
| 2    | TOOL       | rag_search | dk=dv=dmodel/h=64   |
| 3    | TOOL       | calculator | 512                 |
| 4    | FINAL      | —          | complete answer     |

Tools used : 3
Steps used : 4
LLM calls  : 5

Key finding: agent correctly inferred 8 heads × 64 dims = 512 = dmodel
from RAG results without being told to make that connection.

# Day 11 — Memory Systems for ReAct Agents

## What was built
Two-tier memory system integrated into the Day 10 ReAct agent.
Short-term memory lives in RAM for the session.
Long-term memory persists to SQLite across sessions.
Both tiers exposed as agent tools so the LLM can read and write memory mid-reasoning.

## Memory architecture
Memory (facade)
├── ShortTermMemory   — dict, RAM only, cleared on exit
└── LongTermMemory    — SQLite (memory.db), survives sessions

Unified retrieve() checks short-term first, falls back to long-term.
store() writes to both tiers by default (persist=True).

## Confidence scores
Every memory entry carries a confidence float (0.0–1.0).
Confidence degrades when RAG corpus contradicts a stored fact.
Entries below threshold (default 0.6) are flagged ⚠ in the agent system prompt.
low_confidence() returns all entries below threshold for re-verification.

## Tools added to agent
| Tool | Input format | Purpose |
|---|---|---|
| memory_store | key\|value\|confidence | Save a fact for future sessions |
| memory_retrieve | key | Recall a previously stored fact |

Total tools available to agent: calculator, dictionary, rag_search, memory_store, memory_retrieve

## Test coverage
- Short-term store and retrieve
- Long-term store and retrieve
- Confidence degradation
- Low-confidence query
- Unified facade (short-term wins on fresh store)
- Context block generation

## Key concepts

### Why short-term before long-term in retrieve()
Current session facts are always fresher than persisted facts.
If the agent just stored something this session, that version wins.
Prevents stale long-term memory poisoning a live reasoning chain.

### Why persist=False exists
Some reasoning is ephemeral — intermediate calc results, scratchpad notes.
persist=False writes to short-term only, keeps SQLite clean.

### Confidence degradation
Protects against stale facts compounding across sessions.
Example: memory stores "attention uses dot-product scoring" at conf=0.9.
RAG returns chunk describing additive attention instead.
Agent calls degrade("attention_mechanism", 0.2) → conf drops to 0.7.
Next session the ⚠ flag prompts re-verification before use.

### Memory injection at scale
Injecting full memory context into every system prompt works for small stores.
At hundreds of entries this bloats the context window and increases cost.
Fix: vector-embed memory keys, retrieve only top-k relevant entries per query.
This is memory RAG — Phase 4 territory.

# Day 12 — Multi-Agent Orchestration

## What this builds
Supervisor agent that routes questions to specialist sub-agents.
Each specialist has a constrained tool set matching its domain.
Supervisor synthesizes the specialist result into a clean final answer.
Fixes the single-agent tool confusion problem from Day 11.

## Architecture
User question
↓
Supervisor — routes to correct specialist (1 LLM call)
├── RAG agent    → rag_search + dictionary
├── Math agent   → calculator only
└── Memory agent → memory_store + memory_retrieve
↓
Supervisor synthesizes final answer (1 LLM call)

## Stack
Same as Day 11 — no new dependencies

## Run order
1. python test_routing.py   — verify routing only (cheap, 1 LLM call per case)
2. python run.py            — full multi-agent run with 4 test questions

## Routing logic
| Question type | Agent routed |
|---|---|
| AI/ML concepts, research, technical knowledge | rag |
| Arithmetic, calculations, numeric problems | math |
| Store a fact, recall a stored fact | memory |

## LLM calls per question
| Stage | Calls |
|---|---|
| Router | 1 |
| Specialist agent (avg 3 steps) | ~3 |
| Synthesizer | 1 |
| Total | ~5 |

## Key findings

### Why specialist agents have constrained tool sets
Giving all tools to all agents causes tool confusion.
A math agent with rag_search available may search the knowledge base
instead of calculating — because the question mentions a familiar concept.
Constrained tools force the right behaviour. Fewer tools = less ambiguity.

### Why the synthesizer is a separate LLM call
Specialist agents return answers in different formats and tones.
Synthesizer normalises everything into one coherent user-facing answer.
System prompt instructs: trust the agent result completely, do not question it.
Without this instruction the synthesizer second-guesses valid agent answers.

### Why router max_tokens=10
Enforces the one-word constraint structurally, not just in the prompt.
Without it the LLM ignores "one word only" and returns an explanation.

## Bugs fixed
Agent was writing Action + Final Answer in the same response without
waiting for a real Observation. Parser accepted Final Answer, guard
rejected it (no tools used), nudge fired, loop repeated until max steps.

Fix 1: nudge message now explicitly says "do not write Final Answer yet"
and shows only the tool call format.
Fix 2: _base_rules() method added to BaseAgent, injected into all
specialist system prompts. Rule: never write Action and Final Answer
in the same response.

Synthesizer was ignoring correct specialist results and generating
its own response from scratch. Root cause: no system prompt grounding.
Fix: explicit system prompt — "you are a result presenter, trust the
agent result completely."

## Limitation discovered
Single-path routing cannot decompose multi-part questions.
A question needing both RAG and Math gets sent to one agent only.
The second part is never answered.
Fix: planner layer — built in Day 13.

## Scores and counters
- Phase 1 naive RAG keyword hit rate : 6/6 = 100%
- Phase 2 RAGAS baseline Day 6       : 0.638
- Phase 2 Day 7 hybrid               : 0.807
- Phase 2 Day 9 capstone             : 0.827
- Phase 3 Day 10 agent               : 3 tools, 4 steps, 5 LLM calls
- Phase 3 Day 11 memory              : 2 memory tools, 5 total tools, SQLite
- Phase 3 Day 12 multi-agent         : 3 specialists, ~5 LLM calls per question


# Day 14 — MCP Foundations

## What this builds
First MCP (Model Context Protocol) implementation from scratch.
A local MCP server exposes two tools over the protocol.
A public fetch MCP server reads URLs safely with no auth required.
A unified MCP client routes tool calls to the correct server automatically.
The Day 10 ReAct agent is rewired to call tools via MCP instead of
hardcoded Python functions.

## What MCP changes architecturally
Day 10: tools were Python functions hardcoded inside the agent process.
Day 14: tools are served by separate processes over a defined protocol.
The agent never imports tool code directly — it discovers tools at runtime
and calls them through the MCP client. Adding a new tool means adding a
new server, not editing agent code.

## Architecture
LLM
↓ tool call
MCP Client (mcp_client.py)
↓ routes automatically by tool name
├── Local MCP server  (local_server.py — calculator + project_facts)
└── Fetch MCP server  (uvx mcp-server-fetch — reads public URLs)
↓ result
MCP Client
↓ observation
LLM

### Why MCP over hardcoded functions
Hardcoded tools couple tool logic to agent code.
MCP decouples them — tools live in separate processes, on separate machines,
or on public servers. The agent discovers tools at runtime via list_tools().
New tools require no agent code changes.

### Tool discovery at runtime
MCPClient.initialize() calls list_tools() on both servers before any
question is asked. The agent builds its system prompt from discovered
tool schemas — not from hardcoded strings. If a server adds a new tool,
the agent picks it up automatically on next initialization.

### Schema-aware argument inference
When Action Input is not valid JSON, the parser falls back to raw string.
The client reads the tool's actual input schema, takes the first required
property name, and builds the correct argument dict automatically.
fetch gets {"url": raw} — not {"expression": raw}.

### Why a new connection per tool call
Each call opens and closes a stdio connection to the server process.
Tradeoff: slight overhead per call but no state management complexity.
At Day 14 scale this is the right choice. Persistent connections add
reconnect logic, heartbeats, and error recovery — not worth it yet.

### Fetch server (public — uvx mcp-server-fetch)
| Tool | Input | Purpose |
|---|---|---|
| fetch | {"url": "..."} | Read content from any public URL |

Fetch is safe because it reads public URLs only — no filesystem access,
no credentials, no auth tokens required. Results truncated to 2000 chars
to prevent context window bloat.

## Scores and counters
- Phase 1 naive RAG keyword hit rate : 6/6 = 100%
- Phase 2 RAGAS baseline Day 6       : 0.638
- Phase 2 Day 7 hybrid               : 0.807
- Phase 2 Day 9 capstone             : 0.827
- Phase 3 Day 10 agent               : 3 tools, 4 steps, 5 LLM calls
- Phase 3 Day 11 memory              : 2 memory tools, SQLite persistence
- Phase 3 Day 12 multi-agent         : 3 specialists, ~5 LLM calls
- Phase 3 Day 13 capstone            : planner, ~11 LLM calls
- Phase 4 Day 14 MCP                 : 2 servers, 3 tools, runtime discovery