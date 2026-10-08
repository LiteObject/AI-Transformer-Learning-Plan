# AI Transformer Learning Plan

A practical, math-friendly study path from fundamentals to Transformers and LLMs. No prior machine learning or advanced mathematics background is required.

**The core idea:** don't learn "math → AI → Transformers." Learn **"AI concept → small amount of math → apply it → next concept."**

## Start Here

Read the [AI Transformer Learning Plan](AI_Transformer_Learning_Plan.md) first. It lays out the overall approach, the six-phase roadmap, the practice projects, and the recommended learning order. Then work through the phase guides in order, keeping [Abbreviations & Symbols](Abbreviations.md) open as a reference.

## Repository Contents

| File | What it covers |
|---|---|
| [AI_Transformer_Learning_Plan.md](AI_Transformer_Learning_Plan.md) | The master plan: roadmap, per-phase overviews, six practice projects, learning strategy, and the end goal. |
| [Phase1_Essential_Math.md](Phase1_Essential_Math.md) | Vectors, matrices, dot products, matrix multiplication, probability, derivatives/gradients, and statistics — only what later phases need. |
| [Phase2_Neural_Network_Fundamentals.md](Phase2_Neural_Network_Fundamentals.md) | Neurons, weights and bias, linear transformations, activation functions, softmax, loss functions, gradient descent, and backpropagation. |
| [Phase3_Embeddings.md](Phase3_Embeddings.md) | Tokenization, token IDs, embedding vectors, vector dimensions, and semantic similarity via dot product and cosine similarity. |
| [Phase4_Attention.md](Phase4_Attention.md) | What attention does, Query/Key/Value, dot-product scores, scaling, softmax, attention weights, and the weighted sum. |
| [Phase5_Transformer_Architecture.md](Phase5_Transformer_Architecture.md) | Self-attention, multi-head attention, feed-forward networks, residual connections, layer normalization, positional information/RoPE, encoder vs. decoder, and causal masking. |
| [Phase6_LLM_Concepts.md](Phase6_LLM_Concepts.md) | Next-token prediction, cross-entropy loss, KV cache, context windows, sampling, prefill/decode, quantization, and LoRA fine-tuning. |
| [Neural_Network_vs_Transformer.md](Neural_Network_vs_Transformer.md) | How a Transformer relates to other neural network architectures (feedforward, CNN, RNN/LSTM/GRU) and why self-attention replaced recurrence. |
| [Abbreviations.md](Abbreviations.md) | Quick lookup for every acronym and symbol used across the guides, each linked to the page where it is explained. |

## Learning Roadmap

| Phase | Focus | Key Outcome |
|---|---|---|
| 1 | Essential Math | Understand vectors, matrices, dot products, probability, and basic gradients. |
| 2 | Neural Networks | Understand neurons, weights, activations, loss, and gradient descent. |
| 3 | Embeddings | Understand how tokens become numerical vectors. |
| 4 | Attention | Understand Q, K, V, softmax, attention weights, and weighted sums. |
| 5 | Transformer Architecture | Understand Transformer blocks, multi-head attention, normalization, and positional information. |
| 6 | LLM Concepts | Understand next-token prediction, KV cache, context, sampling, and inference. |

## How to Use These Guides

1. **Read the phases in order.** Each one builds directly on the previous: math → neural networks → embeddings → attention → architecture → LLMs.
2. **Use just-in-time math.** Learn a mathematical concept when the AI concept requires it, rather than studying all of math upfront.
3. **Work with tiny numbers.** Two- or three-dimensional vectors are usually enough to understand the core idea.
4. **Build intuition before formulas.** Understand what an operation is doing before committing the notation to memory.
5. **Implement as you go.** A small Python/NumPy implementation turns an abstract equation into something concrete.

## Practice Projects

The [learning plan](AI_Transformer_Learning_Plan.md) defines six small projects that mirror the phases. Phases 1–5 also end with their own suggested mini-project.

| # | Project | What you build |
|---|---|---|
| 1 | Vector Playground | Vectors, dot products, matrix multiplication, and cosine similarity with Python/NumPy. |
| 2 | Tiny Neural Network | A small hand-traced network: linear → ReLU → linear → softmax. |
| 3 | Word Embeddings | A tiny vocabulary with embeddings, and similarity comparisons like `king` vs. `queen`. |
| 4 | Attention From Scratch | The full attention calculation (`QKᵀ`, scaling, softmax, weighted sum) in NumPy. |
| 5 | Multi-Head Attention | Splitting the attention implementation into multiple heads and combining their outputs. |
| 6 | Tiny Transformer | A miniature PyTorch Transformer trained to predict the next token. |

## Prerequisites

- Basic arithmetic and high-school algebra are enough to start; the guides introduce everything else.
- Python with NumPy for the early projects, and PyTorch for the Tiny Transformer work in Phases 5–6.
- No GPU required.

## Later Topics

After Phase 6, the plan previews: FlashAttention, GQA/MQA, Mixture of Experts, RoPE in greater depth, speculative decoding, distributed inference, and large-scale training.

## End Goal

By the end, you should be able to look at a Transformer diagram or implementation and understand what each component does, why it is there, what data flows through it, what the tensor dimensions mean, and how the mathematics maps to the code — making concepts like KV cache, context length, RoPE, and quantization much easier to reason about.
