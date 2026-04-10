import os
import sys
import math
import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv(dotenv_path="../../.env")

# ── Tool 1: Calculator ────────────────────────────────────────────────────────
def calculator(expression: str) -> str:
    """
    Evaluates a mathematical expression safely.
    Supports: +, -, *, /, **, sqrt(), sin(), cos(), log(), abs()
    Example: calculator("2 ** 10") → "1024"
    """
    try:
        # Safe eval — only allow math operations
        allowed = {
            "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
            "log": math.log, "abs": abs, "pi": math.pi, "e": math.e,
            "pow": pow, "round": round,
        }
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as ex:
        return f"Error: {ex}"

# ── Tool 2: Dictionary lookup ─────────────────────────────────────────────────
DEFINITIONS = {
    "attention":       "A mechanism in neural networks that allows the model to focus on relevant parts of the input when producing each output token.",
    "transformer":     "A neural network architecture based entirely on attention mechanisms, introduced in 'Attention Is All You Need' (2017).",
    "embedding":       "A dense numerical representation of text that captures semantic meaning as a vector of floating-point numbers.",
    "tokenization":    "The process of splitting text into smaller units (tokens) that a language model can process.",
    "fine-tuning":     "Adapting a pre-trained model to a specific task by training on a smaller, task-specific dataset.",
    "rag":             "Retrieval-Augmented Generation — combining document retrieval with language model generation for grounded answers.",
    "hallucination":   "When a language model generates plausible-sounding but factually incorrect or unsupported content.",
    "temperature":     "A parameter controlling the randomness of LLM outputs — higher values produce more varied responses.",
    "context window":  "The maximum amount of text (measured in tokens) a language model can process in a single inference call.",
    "vector database": "A database optimised for storing and querying high-dimensional embedding vectors using approximate nearest-neighbor search.",
}

def dictionary(word: str) -> str:
    """
    Looks up the definition of an AI/ML term.
    Example: dictionary("attention") → definition string
    """
    key = word.lower().strip()
    if key in DEFINITIONS:
        return DEFINITIONS[key]
    # Partial match
    for term, defn in DEFINITIONS.items():
        if key in term or term in key:
            return f"({term}): {defn}"
    return f"No definition found for '{word}'. Available terms: {', '.join(DEFINITIONS.keys())}"

# ── Tool 3: RAG search ────────────────────────────────────────────────────────
_embed_model  = None
_chroma_col   = None

def _init_rag():
    global _embed_model, _chroma_col
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    if _chroma_col is None:
        try:
            client = chromadb.PersistentClient(
                path="../../phase2-advanced-rag/day9-capstone/chroma_db"
            )
            _chroma_col = client.get_collection("day9")
        except Exception:
            _chroma_col = None

def rag_search(query: str) -> str:
    """
    Searches the Phase 2 document collection for relevant chunks.
    Uses the Attention Is All You Need paper + layer normalisation paper.
    Example: rag_search("what is multi-head attention") → relevant text
    """
    _init_rag()
    if _chroma_col is None:
        return "RAG search unavailable — Phase 2 ChromaDB collection not found."
    try:
        vec     = _embed_model.encode([query]).tolist()
        results = _chroma_col.query(
            query_embeddings=vec, n_results=2,
            include=["documents", "distances"]
        )
        chunks = results["documents"][0]
        dists  = results["distances"][0]
        if not chunks:
            return "No relevant content found."
        output = []
        for chunk, dist in zip(chunks, dists):
            output.append(f"[relevance={1-dist:.2f}] {chunk[:300]}")
        return "\n\n".join(output)
    except Exception as ex:
        return f"RAG search error: {ex}"

# ── Tool registry ─────────────────────────────────────────────────────────────
TOOLS = {
    "calculator": {
        "fn":          calculator,
        "description": "Evaluates mathematical expressions. Input: a valid math expression as a string. Example: '2 ** 10' or 'sqrt(144)'",
    },
    "dictionary": {
        "fn":          dictionary,
        "description": "Looks up definitions of AI/ML terms. Input: a single term or phrase. Example: 'attention' or 'transformer'",
    },
    "rag_search": {
        "fn":          rag_search,
        "description": "Searches the document knowledge base for relevant information. Input: a natural language query. Example: 'how does multi-head attention work'",
    },
}

def describe_tools() -> str:
    """Returns a formatted description of all available tools for the system prompt."""
    lines = []
    for name, info in TOOLS.items():
        lines.append(f"- {name}: {info['description']}")
    return "\n".join(lines)

def run_tool(name: str, input_str: str) -> str:
    """Executes a tool by name with the given input string."""
    if name not in TOOLS:
        return f"Unknown tool '{name}'. Available tools: {', '.join(TOOLS.keys())}"
    return TOOLS[name]["fn"](input_str)