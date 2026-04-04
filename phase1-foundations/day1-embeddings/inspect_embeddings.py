from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "The cat sat on the mat",
    "A feline rested on the rug",        # paraphrase -- should score HIGH
    "Python is a programming language",  # unrelated -- should score LOW
]

embeddings = model.encode(sentences)

print(f"Shape: {embeddings.shape}")
print(f"Each sentence = {embeddings.shape[1]} floats\n")

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

sim_paraphrase  = cosine_sim(embeddings[0], embeddings[1])
sim_unrelated   = cosine_sim(embeddings[0], embeddings[2])

print(f"'cat on mat'  vs  'feline on rug':   {sim_paraphrase:.4f}  <- should be HIGH (0.80+)")
print(f"'cat on mat'  vs  'Python language': {sim_unrelated:.4f}  <- should be LOW  (0.10-)")
print()
print("This gap is the entire mathematical foundation of RAG.")