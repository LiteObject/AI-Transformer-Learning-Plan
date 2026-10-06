# AI Transformer Architecture Learning Plan

A practical, math-friendly path from fundamentals to Transformers and LLMs.

## The Overall Approach

**Don't learn “math → AI → Transformers.” Learn “AI concept → small amount of math → apply it → next concept.”**

You do not need an advanced mathematics or statistics background to understand Transformers. The goal is to learn the minimum mathematics needed at each stage, immediately apply it, and gradually build intuition.

---

## Learning Roadmap

| Phase | Focus | Key Outcome |
|---|---|---|
| 1 | Essential Math | Understand vectors, matrices, dot products, probability, and basic gradients |
| 2 | Neural Networks | Understand neurons, weights, activations, loss, and gradient descent |
| 3 | Embeddings | Understand how tokens become numerical vectors |
| 4 | Attention | Understand Q, K, V, softmax, attention weights, and weighted sums |
| 5 | Transformer Architecture | Understand Transformer blocks, multi-head attention, normalization, and positional information |
| 6 | LLM Concepts | Understand next-token prediction, KV cache, context, sampling, and inference |

## Phase Files

Each phase has a dedicated, beginner-friendly guide — no prior knowledge required:

1. [Phase 1 — Essential Math](Phase1_Essential_Math.md)
2. [Phase 2 — Neural Network Fundamentals](Phase2_Neural_Network_Fundamentals.md)
3. [Phase 3 — Embeddings](Phase3_Embeddings.md)
4. [Phase 4 — Attention](Phase4_Attention.md)
5. [Phase 5 — Transformer Architecture](Phase5_Transformer_Architecture.md)
6. [Phase 6 — LLM Concepts](Phase6_LLM_Concepts.md)

**Reference:** [Abbreviations & Symbols](Abbreviations.md) — a quick lookup for every acronym and symbol used in the plan, each linked to where it's explained.

---

# Phase 1 — Essential Math

Learn only the mathematics needed for the AI concepts that follow. Avoid trying to master all of linear algebra, calculus, or statistics upfront.

- **Numbers, vectors, and matrices:** vectors, dimensions, matrices, matrix multiplication, and dot products.
- **Functions:** inputs → transformation → outputs; understand neural networks as compositions of functions.
- **Probability basics:** probability, distributions, conditional probability, and why models output probabilities.
- **Basic calculus:** derivatives, gradients, and why gradients are used to train neural networks.
- **Statistics basics:** mean, variance, standard deviation, distributions, and normalization.

---

# Phase 2 — Neural Network Fundamentals

Learn:

- Neurons, weights, and bias
- Linear transformations
- Activation functions
- ReLU and sigmoid
- Softmax
- Loss functions
- Gradient descent
- Backpropagation

**Target understanding:** Be able to follow the calculations in a very small neural network rather than treating training as a black box.

---

# Phase 3 — Embeddings

Embeddings are a critical bridge between basic neural networks and Transformers. Learn how text becomes numerical representations.

For example:

```text
"cat"
   ↓
token
   ↓
token ID
   ↓
embedding vector
   ↓
[0.21, -0.73, 0.14, ..., 0.82]
```

Learn:

- Tokenization
- Token IDs
- Embedding vectors
- Vector dimensions
- Semantic similarity
- Dot product
- Cosine similarity

---

# Phase 4 — Attention

Learn attention independently before studying the complete Transformer.

Start with intuition and then introduce the mathematics one piece at a time.

Learn:

1. What attention is trying to accomplish
2. Query, Key, and Value
3. Dot product
4. Scaling
5. Softmax
6. Attention weights
7. Weighted sum

For example, in:

> “The animal didn't cross the street because it was tired.”

Attention helps the model determine which other tokens are relevant to **“it.”**

Eventually understand:

```text
Attention(Q, K, V)
=
softmax(QKᵀ / √dₖ)V
```

**Important:** Do not start by memorizing this equation. Build it from the individual operations.

---

# Phase 5 — Transformer Architecture

Once embeddings and attention are comfortable, bring the pieces together.

```text
Tokens
   ↓
Token Embeddings
   +
Positional Information
   ↓
Transformer Blocks
   ↓
Output
```

Learn:

- Self-attention
- Multi-head attention
- Feed-forward network
- Residual connections
- Layer normalization
- Positional encoding and RoPE
- Encoder vs. decoder
- Causal masking
- Transformer blocks

---

# Phase 6 — LLM-Specific Concepts

After understanding the Transformer architecture, move into:

- Autoregressive generation
- Next-token prediction
- Cross-entropy loss
- KV cache
- Context window
- Temperature
- Top-k sampling
- Top-p sampling
- Prefill and decode
- Inference optimization
- Quantization
- LoRA and fine-tuning

Later topics can include:

- FlashAttention
- GQA/MQA
- Mixture of Experts
- RoPE in greater depth
- Speculative decoding
- Distributed inference
- Large-scale training

---

# Learn Through Small Projects

## Project 1 — Vector Playground

Implement:

```text
vector
dot product
matrix multiplication
cosine similarity
```

Use Python/NumPy.

The goal is to become comfortable with the basic operations that appear repeatedly in machine learning.

---

## Project 2 — Tiny Neural Network

Build a small neural network:

```text
Input
  ↓
Linear Layer
  ↓
ReLU
  ↓
Linear Layer
  ↓
Softmax
```

Follow every calculation.

The goal is to understand what the network is actually doing rather than treating the neural network as a black box.

---

## Project 3 — Word Embeddings

Create a tiny vocabulary and assign embeddings.

Then calculate similarities such as:

```text
similarity("king", "queen")
similarity("king", "car")
```

This develops intuition for how words and concepts can be represented as vectors.

---

## Project 4 — Attention From Scratch

Implement the basic attention calculation:

```python
scores = Q @ K.T
scores = scores / np.sqrt(K.shape[-1])   # scaling (see Phase 4)
weights = softmax(scores)
output = weights @ V
```

This is one of the most valuable exercises before studying the complete Transformer.

---

## Project 5 — Multi-Head Attention

Take the attention implementation and split the representation into multiple attention heads.

Understand:

- Why multiple heads are useful
- How each head can learn different relationships
- How the heads are combined

---

## Project 6 — Tiny Transformer

Build a miniature Transformer using PyTorch.

For example:

```text
Vocabulary: 1,000 tokens
Embedding size: 64
Attention heads: 4
Transformer layers: 2
Context length: 32
```

Train it to predict the next token.

At this point, a real LLM architecture should look much less mysterious.

---

# The Key Connection

The concepts are not isolated. They form a chain:

```text
Embeddings
    ↓
Token vectors
    ↓
Attention
    ↓
Q / K / V
    ↓
Matrix operations
    ↓
Softmax
    ↓
Attention weights
    ↓
Transformer block
    ↓
Next-token logits
    ↓
Token probabilities
    ↓
Next token
```

Once this chain is clear, Transformers stop feeling like a giant collection of mysterious terminology.

---

# Recommended Learning Strategy

### 1. Use Just-in-Time Math

Learn a mathematical concept when an AI concept requires it.

For example:

- Learning embeddings → vectors, dimensions, dot product
- Learning attention → matrix multiplication, Q/K/V, softmax
- Learning training → derivatives, gradients, gradient descent, backpropagation

### 2. Use Tiny Numerical Examples

Two- or three-dimensional vectors are often enough to understand the core idea.

You do not need huge matrices to learn the concepts.

### 3. Implement Concepts

A small Python/NumPy implementation can make an abstract equation concrete.

### 4. Build Intuition Before Formulas

Understand what an operation is doing before focusing heavily on mathematical notation.

For example, understand:

> “Attention determines how much each token should pay attention to other tokens.”

Then learn how Q, K, V and softmax implement that idea.

### 5. Don't Wait Until You “Finish Math”

Do not spend months studying mathematics before touching Transformers.

Keep progressing toward Transformers while filling mathematical gaps along the way.

---

# Suggested Learning Order

```text
Math foundations
       ↓
Vectors & matrices
       ↓
Neural networks
       ↓
Embeddings
       ↓
Attention
       ↓
Self-attention
       ↓
Multi-head attention
       ↓
Transformer block
       ↓
Full Transformer
       ↓
LLM
```

---

# End Goal

The goal is not simply to recognize Transformer terminology.

You should eventually be able to look at a Transformer diagram or implementation and understand:

- What each component does
- Why it is there
- What data flows through it
- What the dimensions of the tensors mean
- How the mathematics maps to the code
- How the Transformer produces the next-token prediction
- How the architecture is optimized for real-world LLM inference

Once you reach that point, concepts such as **KV cache, context length, RoPE, FlashAttention, GQA, quantization, speculative decoding, and inference engines** become much easier to understand because you have a solid mental model of the underlying architecture.
