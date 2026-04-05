import os
import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

import chromadb
import anthropic
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from rich import print
from rich.panel import Panel

# Load API key from .env file at repo root
load_dotenv(dotenv_path="../../.env")
client_llm = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Same corpus as Day 1 -- in Phase 2 you will load real documents
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

# --- Setup: embedding model + vector store ---
print("[bold cyan]Loading embedding model...[/bold cyan]")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="day2",
    metadata={"hnsw:space": "cosine"}
)

if collection.count() == 0:
    print("[yellow]Ingesting documents...[/yellow]")
    texts = [d["text"] for d in DOCUMENTS]
    ids   = [d["id"]   for d in DOCUMENTS]
    embeddings = embed_model.encode(texts, show_progress_bar=False).tolist()
    collection.add(ids=ids, embeddings=embeddings, documents=texts)
    print(f"[green]Stored {len(DOCUMENTS)} vectors.[/green]\n")

# --- Core RAG function ---
def rag(question: str, n_results: int = 3) -> dict:
    """
    Full naive RAG pipeline:
    1. Embed the question
    2. Retrieve top-n chunks from ChromaDB
    3. Build a prompt with retrieved context
    4. Call Claude and return grounded answer
    """

    # Step 1: Retrieve
    query_vec = embed_model.encode([question]).tolist()
    results   = collection.query(query_embeddings=query_vec, n_results=n_results)
    chunks    = results["documents"][0]
    distances = results["distances"][0]

    # Step 2: Build prompt
    context = "\n\n".join([f"[{i+1}] {chunk}" for i, chunk in enumerate(chunks)])

    system_prompt = """You are a helpful assistant. Answer the user's question
using ONLY the context provided below. If the context does not contain
enough information to answer, say "I don't have enough context to answer that."
Do not use any outside knowledge."""

    user_message = f"""Context:
{context}

Question: {question}

Answer based only on the context above:"""

    # Step 3: Call Claude
    response = client_llm.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": user_message}],
        system=system_prompt,
    )

    answer = response.content[0].text

    return {
        "question":  question,
        "chunks":    chunks,
        "distances": distances,
        "answer":    answer,
    }


# --- Run test questions ---
TEST_QUESTIONS = [
    "What is RAG and how does it work?",
    "How do agents decide which tools to use?",
    "What does ChromaDB store and why is it useful?",
    "What is the attention mechanism?",
]

print("[bold]--- Naive RAG Pipeline Results ---[/bold]\n")

for question in TEST_QUESTIONS:
    result = rag(question)

    print(Panel(
        f"[bold magenta]Q:[/bold magenta] {result['question']}\n\n"
        f"[bold yellow]Retrieved chunks:[/bold yellow]\n"
        + "\n".join([
            f"  [{i+1}] (dist={d:.4f}) {c}"
            for i, (c, d) in enumerate(zip(result["chunks"], result["distances"]))
        ])
        + f"\n\n[bold green]Answer:[/bold green] {result['answer']}",
        expand=False
    ))
    print()