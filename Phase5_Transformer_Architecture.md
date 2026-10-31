# Phase 5 — Transformer Architecture

> **Goal:** Now that you know embeddings (Phase 3) and attention (Phase 4), bring everything together into the full Transformer — the architecture behind modern language models.

> 🎨 **Interactive companion:** see this phase come to life — open **[Transformer_Architecture_Infographic.html](Transformer_Architecture_Infographic.html)** in a browser. It animates encoder-only, decoder-only, and encoder-decoder side by side, with step-by-step captions (use the ‹ › buttons or arrow keys), hover explanations on every block, and attention masks labeled with real tokens.

The companion illustrates a **canonical post-norm** layout. The block and runnable model below use **pre-norm**; Section 6 explains the difference. Both layouts are valid.

---

## 1. The Big Picture

A decoder-only Transformer with **additive positional embeddings** follows this flow:

```
Tokens
   ↓
Token Embeddings   (Phase 3: words become vectors)
   +
Positional Information  (where each token is in the sentence)
   ↓
Transformer Blocks  (attention + feed-forward, repeated several times)
   ↓
Output logits  (scores for the next token)
```

The **Transformer block** is the workhorse. It's repeated many times (for example, 12, 32, or even 100+ times in big models). Each block refines the token vectors, layer by layer, making them more and more context-aware.

---

## 2. Self-Attention

**Self-attention** is the attention from Phase 4 where Q, K, and V come from the **same sequence**. It can be bidirectional or restricted by a mask; "self" describes the source, not a requirement to attend to every position.

```
"the cat sat on the mat"
   ↑    ↑    ↑   ↑   ↑    ↑
each token looks at all others
```

With bidirectional attention, the words "the" at position 0 and position 4 can share context. With causal attention, position 4 can use position 0, but position 0 cannot use position 4.

In a block, Q, K, and V are all computed from the same current hidden states (normalized first in a pre-norm block). Later blocks do not start over from the original token embeddings.

---

## 3. Multi-Head Attention

One attention pass = one "view" of the relationships between tokens. But relationships are complex — maybe one type of relationship is "who does the pronoun refer to", another is "which words form a phrase", another is "is this a question?"

**Multi-head attention** runs attention **several times in parallel**, each with its own learned Q, K, V matrices:

```
input
  ↓
  ├── head 1: attention → looks for one kind of relationship
  ├── head 2: attention → looks for another kind
  ├── head 3: attention → looks for another kind
  └── ...
  ↓
concatenate all heads' outputs
  ↓
linear layer to blend them
  ↓
output
```

- Each head can learn different patterns, but it need not have a single, human-interpretable specialty.
- Their outputs are **concatenated** (glued together) and passed through a linear layer — the **output projection** — to combine the views.

> 📝 **Notation:** In the original Transformer paper and most courses, each head's projections are written $W_i^Q$, $W_i^K$, $W_i^V$ (the same idea as `W_Q`, `W_K`, `W_V` from Phase 4, but with a separate set per head), and the final blending layer is written $W^O$.

> 💡 **Analogy:** A team of detectives investigates the same sentence, each focusing on a different clue (grammar, references, tone...). Then they pool their findings.

Modern models use anywhere from 8 to 128 heads.

### Tensor shapes through multiple heads

Let `B` be batch size, `T` sequence length, `D` model width, and `H` head count. In this standard equal-width example, `D` must be divisible by `H`, and `d_head = D / H`.

```
input                       (B, T, D)
project Q, K, V              (B, T, D) each
split and transpose          (B, H, T, d_head) each
score each head              (B, H, T, T)
blend each head's values     (B, H, T, d_head)
transpose and concatenate    (B, T, D)
output projection            (B, T, D)
```

The full projection matrices have shape `(D, D)` in this example; their column slices contain each head's separate projections. This is equivalent to the per-head notation above. GQA/MQA use different K/V head counts and are later extensions, not this exercise.

---

## 4. Feed-Forward Network (FFN)

After attention, each token's vector passes through a small neural network — the **feed-forward network** (Phase 2: linear → activation → linear).

```
token vector
   ↓
Linear layer (expand)
   ↓
ReLU or other activation
   ↓
Linear layer (compress back)
   ↓
new token vector
```

What does it do? Attention **mixes information between tokens**. The feed-forward network **processes each token on its own** — like a deep thinker that transforms each token's meaning after it has absorbed context.

The **same FFN weights** are applied at every token position; it is not a separately trained network for each token. Its output width returns to `D` so residual addition has matching shapes.

> 💡 **Analogy:** First the group discusses (attention). Then each person thinks alone about what was learned (feed-forward).

---

## 5. Residual Connections (The "Skip" Paths)

Residual connections (also called **skip connections** or **shortcuts**) let information **bypass** a layer:

```
input ───────────────┐
        ↓            │
     [layer]         │
        ↓            │
     output ─────────┤
                     ↓
        output + input  (the residual addition)
```

So each sub-layer actually computes:

```
final = input + layer(input)
```

### Why?

1. **Training stability:** gradients can flow back through the "shortcut" even if the layer itself is troublesome (Phase 2 backpropagation).
2. **Identity preservation:** the model can choose to keep the original information and add only small refinements — it's easy to "do nothing" if needed.

> 💡 **Analogy:** A highway with a side road. Even if the main road is blocked, traffic can still get through. The model always has a direct route.

---

## 6. Layer Normalization

**Layer normalization** operates **per token vector**. For an input of shape `(B, T, D)`, compute the mean and variance across its last dimension, `D`, then apply learned per-feature scale and shift:

```
normalized = (x - mean) / sqrt(variance + epsilon)
output = normalized * gamma + beta
```

`epsilon` is a small positive number that prevents division by zero. `gamma` and `beta` each have shape `(D,)` and are shared across tokens. They let the model adjust the normalized features, but do **not** generally undo normalization: fixed learned parameters cannot recover every input's discarded mean and scale.

### Why?

- Keeps numbers stable as they flow through dozens of layers.
- Makes training faster and more reliable.
- Prevents "exploding" values (numbers growing out of control).

In a **pre-norm** Transformer block, the order is:

```
Normalize → Multi-Head Attention → Residual add
Normalize → Feed-Forward Network → Residual add
```

More precisely:

```
h = x + MultiHeadAttention(LayerNorm(x))
output = h + FFN(LayerNorm(h))
```

The original Transformer used **post-norm**: `LayerNorm(x + sublayer(x))`. PyTorch's `TransformerEncoderLayer` defaults to post-norm (`norm_first=False`); the capstone explicitly sets `norm_first=True` to match the pre-norm diagram here.

---

## 7. Positional Information: Knowing the Order

Attention treats all tokens equally — it doesn't naturally know that "the" at position 0 and "the" at position 4 are different. But word **order matters**:

> "dog bites man" ≠ "man bites dog"

So we must give the model information about **position**. There are two main approaches:

### Positional Encoding (original approach)

Add a **position vector** to each token's embedding:

```
final embedding for token i = token_embedding + position_vector(i)
```

The position vectors use sine and cosine waves of different frequencies, which create a unique, smooth pattern for each position — the model can tell where each token sits, and even that position 5 is "near" position 4.

### RoPE (Rotary Position Embedding, modern approach)

**RoPE** encodes position by applying position-dependent rotations to the query and key representations. It is widely used in modern LLMs, including Llama-family models, but it is not used by every model.

Advantages of RoPE:

- Its construction makes attention scores sensitive to **relative** positions, such as the distance between two tokens.
- It can support long-context extensions, but generalizing reliably beyond the training length requires additional design or training and is not guaranteed.
- It is compatible with KV caching, but is **not required** for caching. Causal models with additive positions can cache K/V too; the key requirement is that the cached past representations remain valid.

> 💡 The key idea in both approaches: **"I am the 5th token, and I know the token before me is the 4th."**

---

## 8. Encoder vs. Decoder

The original Transformer paper had two halves:

| Part | Job | Sees... |
|---|---|---|
| **Encoder** | Understand the whole input | All tokens (both directions) |
| **Decoder** | Generate output one token at a time | Past/current output tokens through causal self-attention; encoder output through cross-attention when an encoder is present |

- **Encoder-only** models (like BERT) are great at *understanding* (classification, search).
- **Decoder-only** models (like GPT and most modern LLMs) are great at *generating* text — they are the family that powers ChatGPT.
- There are also **encoder-decoder** models (like the original Transformer for translation).

In an encoder-decoder model, the decoder's **cross-attention** takes queries from its own hidden states and keys/values from the encoder output. A decoder-only model has no separate encoder or that cross-attention stage.

Many modern chat models use **decoder-only** Transformers: they read your prompt and generate the answer one token at a time.

---

## 9. Causal Masking: Don't Look Ahead

A decoder generates left to right. When predicting token 5, it must **not** be allowed to see token 6 (that would be cheating — the answer would be in the input!).

**Causal masking** forces the attention weights so that each token can **only attend to itself and earlier tokens**:

```
        attended to →
        1   2   3   4   5
token 1 ✓   ✗   ✗   ✗   ✗
token 2 ✓   ✓   ✗   ✗   ✗
token 3 ✓   ✓   ✓   ✗   ✗
token 4 ✓   ✓   ✓   ✓   ✗
token 5 ✓   ✓   ✓   ✓   ✓
```

The upper-right triangle is blocked with a boolean mask or scores set to negative infinity, so softmax gives it zero weight. A merely moderately negative finite score does not guarantee zero probability. The mask prevents future-token leakage during parallel training and prompt processing.

---

## 10. One Full Transformer Block (Summary)

This is the **pre-norm** variant used in the capstone:

```
input vectors
   ↓
┌─────────────────────────────┐
│  Layer Normalization        │
│  Multi-Head Self-Attention  │
│  + Residual connection      │
└─────────────────────────────┘
   ↓
┌─────────────────────────────┐
│  Layer Normalization        │
│  Feed-Forward Network       │
│  + Residual connection      │
└─────────────────────────────┘
   ↓
output vectors (more context-aware than input)
```

Repeat this block many times. **Additive positions** go into the embeddings up front; **RoPE** instead rotates Q/K inside attention. The capstone uses additive learned positions, then a final layer normalization and linear output head. The head returns **raw logits**; softmax converts them to probabilities for generation, while training passes logits directly to cross-entropy.

---

## Why It All Matters

Each layer of a Transformer:

1. **Attention** lets tokens exchange information with each other (with positional awareness and causal masking).
2. **Residual + normalization** keep training stable.
3. **Feed-forward** lets each token individually "think" about what it learned.

Stack enough of these, and the model develops rich, hierarchical understanding of language — from grammar, to references, to reasoning.

---

## Suggested Mini-Project: Multi-Head Attention, Then a Tiny Transformer

### Part A: multi-head attention without hiding the shapes

Run the Phase 4 Python example first, keeping its `attention` function in the same Python session. Then run:

```python
import numpy as np

def multi_head_attention(X, W_Q, W_K, W_V, W_O, heads, mask=None):
    if X.ndim != 3 or min(X.shape) <= 0:
        raise ValueError("X must have nonempty shape (batch, sequence, model_width)")
    B, T, D = X.shape
    if heads <= 0 or D % heads != 0:
        raise ValueError("model width must be divisible by a positive head count")
    if any(W.shape != (D, D) for W in (W_Q, W_K, W_V, W_O)):
        raise ValueError("this example requires four (model_width, model_width) projections")
    d_head = D // heads

    def split_heads(W):
        return (X @ W).reshape(B, T, heads, d_head).transpose(0, 2, 1, 3)

    Q, K, V = split_heads(W_Q), split_heads(W_K), split_heads(W_V)
    outputs = np.empty((B, heads, T, d_head))
    for batch in range(B):
        for head in range(heads):
            _, outputs[batch, head] = attention(
                Q[batch, head], K[batch, head], V[batch, head], mask=mask)
    concatenated = outputs.transpose(0, 2, 1, 3).reshape(B, T, D)
    return concatenated @ W_O

rng = np.random.default_rng(42)
mh_input = rng.normal(size=(2, 3, 4))
mh_projections = [rng.normal(size=(4, 4)) / np.sqrt(4) for _ in range(4)]
mh_mask = np.triu(np.ones((3, 3), dtype=bool), k=1)
multi_head_output = multi_head_attention(
    mh_input, *mh_projections, heads=2, mask=mh_mask)
print("multi-head output shape:", multi_head_output.shape)  # (2, 3, 4)
```

These projection weights are illustrative random values, not trained semantic specialists. Trace one head before following the entire batch.

### Part B: assemble the model in PyTorch

The complete, reusable implementation is in [examples/tiny_transformer.py](examples/tiny_transformer.py). From a Python session started at the repository root:

```python
import torch
from examples.tiny_transformer import TinyTransformer

model = TinyTransformer(vocab=1000, embed=32, heads=4, layers=2, ctx=32)
sample = torch.randint(0, 1000, (1, 8))
logits = model(sample)
print("logits shape:", logits.shape)   # (1, 8, 1000): one score per vocab per position
```

The last position's logits, after softmax, describe the model's distribution for the **next** token. At this point it is **untrained**: producing the right shape is not evidence that it has learned language. Phase 6 trains this same class on a small corpus, checks held-out loss, and generates a continuation.

Despite the PyTorch API name `TransformerEncoderLayer`, the explicit causal mask makes these blocks usable in a **decoder-only language model without cross-attention**. `norm_first=True` matches the diagram, and the learned position table is limited to `ctx` positions.

## Before Moving On

| Check | Completion criterion |
|---|---|
| By hand | For `(B, T, D) = (2, 3, 4)` and `H = 2`, derive `(2, 2, 3, 2)` per projected Q/K/V tensor and `(2, 2, 3, 3)` for scores. |
| In code | Produce multi-head output `(2, 3, 4)` and logits `(1, 8, 1000)`. Change future input tokens and confirm earlier causal logits stay unchanged in evaluation mode. |
| In your own words | Explain concatenation/output projection, shared FFN weights, pre-norm versus post-norm, and why additive positions and RoPE enter at different places. |


---

**Next step →** [Phase 6: LLM Concepts](Phase6_LLM_Concepts.md)
