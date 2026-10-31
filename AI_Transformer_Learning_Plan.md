# AI Transformer Architecture Learning Plan

A practical, math-friendly path from fundamentals to Transformers and LLMs.

## The Overall Approach

**Don't learn “math → AI → Transformers.” Learn “AI concept → small amount of math → apply it → next concept.”**

You do not need an advanced mathematics or statistics background to understand Transformers. The goal is to learn the minimum mathematics needed at each stage, immediately apply it, and gradually build intuition.

The coding exercises assume basic Python variables, functions, loops, and lists. See [prerequisites and setup](README.md#prerequisites-and-setup). Begin with vectors, matrix shapes, and probability; revisit gradients during neural-network training and statistics during layer normalization.

---

## Learning Roadmap

| Phase | Focus | Key Outcome |
|---|---|---|
| 1 | Essential Math | Calculate vector/matrix results, track shapes, and check a numerical derivative |
| 2 | Neural Networks | Trace forward/backward passes and train a tiny classifier |
| 3 | Embeddings | Map tokens to vectors; distinguish input embeddings from contextual hidden states |
| 4 | Attention | Implement Q/K/V projections, row-wise attention, and a verified causal mask |
| 5 | Transformer Architecture | Trace batched heads and assemble a causal pre-norm Transformer |
| 6 | LLM Concepts | Train/evaluate a small language model, generate within limits, and check cache equivalence |

## Phase Files

Each phase has a dedicated, beginner-friendly guide. No prior machine learning or advanced mathematics is required; basic Python is needed for the exercises:

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

- **Numbers, vectors, and matrices:** vectors, dimensions, matrix shapes, transposes, matrix multiplication, and dot products.
- **Functions:** inputs → transformation → outputs; understand neural networks as compositions of functions.
- **Probability basics:** probability, distributions, conditional probability, and why models output probabilities.
- **Basic calculus (revisit in Phase 2):** derivatives, numerical gradient checks, and why training subtracts the gradient.
- **Statistics basics (revisit in Phase 5):** mean, variance, standard deviation, distributions, and normalization.

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

**Target understanding:** Follow a forward pass, calculate a gradient and weight update, and train a small classifier with decreasing loss. Distinguish backpropagation from optimization, and pass raw logits to PyTorch cross-entropy.

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
- Static input embeddings vs. contextual hidden states

Handmade vector similarities and classic word analogies are illustrative, not guarantees about every trained embedding space.

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
8. Tensor shapes, row-wise softmax, and causal masking

For example, in:

> “The animal didn't cross the street because it was tired.”

Attention can route relevant contextual information toward **“it.”** A weight alone does not prove human-like reference resolution.

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

This diagram shows **additive** positional information. RoPE instead rotates Q/K inside attention; it is not added to the embedding vector and is not required for KV caching.

Learn:

- Self-attention
- Multi-head attention
- Batch/head/sequence dimensions and output projection
- Feed-forward network
- Residual connections
- Layer normalization
- Positional encoding and RoPE
- Encoder vs. decoder
- Causal masking
- Transformer blocks
- Pre-norm vs. post-norm layouts

---

# Phase 6 — LLM-Specific Concepts

After understanding the Transformer architecture, complete the required training/generation core:

- Autoregressive generation
- Next-token prediction with shifted targets and teacher forcing
- Cross-entropy on logits, optimizer updates, and held-out loss/perplexity
- Context window
- Temperature
- Top-k sampling
- Top-p sampling
- A runnable CPU training/evaluation/generation capstone
- Base language models vs. instruction-tuned chat models

Then connect that baseline to inference:

- Per-layer KV cache and a small cache-equivalence check
- Prefill and decode

Optional extensions:

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

The [Phase 2 exercise](Phase2_Neural_Network_Fundamentals.md) extends this into a backward pass and training loop. Check a weight's gradient numerically and verify decreasing loss on the tiny classification dataset.

---

## Project 3 — Word Embeddings

Create a tiny vocabulary and assign embeddings.

Then calculate similarities such as:

```text
similarity("king", "queen")
similarity("king", "car")
```

These are handmade examples, not learned relationships. Check that positive scaling changes the dot product but not cosine similarity, and distinguish a token's input embedding from its later contextual hidden states.

---

## Project 4 — Attention From Scratch

Implement the basic attention calculation. This is a schematic; the [Phase 4 exercise](Phase4_Attention.md) supplies imports, Q/K/V projections, stable row-wise softmax, and a boolean causal mask:

```text
scores = Q @ K.T
scores = scores / sqrt(K.shape[-1])
scores = mask_future_positions(scores)
weights = softmax_each_row(scores)
output = weights @ V
```

Check that rows sum to one and masked weights are zero. A changed future token must not affect an earlier causal output.

---

## Project 5 — Multi-Head Attention

Take the attention implementation and split the representation into multiple attention heads.

Understand:

- Why multiple heads are useful
- How each head can learn different relationships
- How the heads are combined
- How `(B, T, D)` becomes `(B, H, T, d_head)` and returns to `(B, T, D)`

The [Phase 5 exercise](Phase5_Transformer_Architecture.md) implements this explicitly in NumPy before assembling the reusable PyTorch model.

---

## Project 6 — Tiny Transformer

Train and evaluate the [miniature PyTorch Transformer](examples/tiny_transformer.py) introduced in Phase 5:

```powershell
.\.venv\Scripts\python.exe .\examples\tiny_transformer.py
```

```text
Vocabulary: characters found in the bundled training text
Embedding size: 32
Attention heads: 4
Transformer layers: 2
Context length: 32
```

The script builds shifted inputs/targets, trains on original toy text, records train and held-out validation loss/perplexity, and generates within an explicit token/context budget. It runs on CPU with no external model, dataset, or API key.

The [Phase 6 guide](Phase6_LLM_Concepts.md) explains each source function and decoding controls. This baseline deliberately has no KV cache; its separate NumPy exercise verifies the cache idea without pretending to be a production-serving implementation.

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

### 6. Verify Each Phase Before Moving On

Each guide ends with three checks:

1. **By hand:** calculate a small example and predict its shapes.
2. **In code:** reproduce the expected result or test the behavior.
3. **In your own words:** explain why the operation exists and what it does not guarantee.

Use the capstone's before/after held-out loss as evidence of learning on the toy task, not merely a correctly shaped tensor. Run the documented regression checks from [README.md](README.md#run-the-capstone-and-checks); they require at least a 20% held-out loss reduction for the seeded baseline and verify future-token isolation.

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
