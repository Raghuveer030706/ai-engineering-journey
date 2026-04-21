# phase5-production/day19-ragas-eval/questions.py

EVAL_SET = [
    {
        "question": "What is the attention mechanism in transformers?",
        "ground_truth": "The attention mechanism allows transformers to weigh the importance of different words in a sequence when encoding a representation. It computes query, key, and value vectors and uses scaled dot-product attention to produce weighted outputs."
    },
    {
        "question": "What are the main components of a transformer architecture?",
        "ground_truth": "The main components are multi-head self-attention layers, position-wise feed-forward networks, residual connections, layer normalization, and positional encodings."
    },
    {
        "question": "How does multi-head attention differ from single-head attention?",
        "ground_truth": "Multi-head attention runs multiple attention functions in parallel, each with different learned projections. This allows the model to attend to information from different representation subspaces at different positions."
    },
    {
        "question": "What is the role of positional encoding in transformers?",
        "ground_truth": "Positional encodings inject information about the position of tokens in the sequence since the attention mechanism itself is permutation invariant. They are added to the input embeddings before processing."
    },
    {
        "question": "What is the scaled dot-product attention formula?",
        "ground_truth": "Scaled dot-product attention is computed as Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V, where Q, K, V are query, key and value matrices and d_k is the dimension of the keys used for scaling."
    },
    {
        "question": "Why do transformers use residual connections?",
        "ground_truth": "Residual connections allow gradients to flow directly through the network without passing through attention or feed-forward layers, which helps avoid vanishing gradients and enables training of deeper networks."
    },
]