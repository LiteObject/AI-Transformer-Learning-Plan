# Phase 5 — Transformer Architecture

> **Goal:** Now that you know embeddings (Phase 3) and attention (Phase 4), bring everything together into the full Transformer — the architecture behind modern language models.

---

## 1. The Big Picture

A Transformer processes text in this overall flow:

```
Tokens
   ↓
Token Embeddings   (Phase 3: words become vectors)
   +
Positional Information  (where each token is in the sentence)
   ↓
Transformer Blocks  (attention + feed-forward, repeated several times)
   ↓
Output  (predictions for the next token)
```

The **Transformer block** is the workhorse. It's repeated many times (for example, 12, 32, or even 100+ times in big models). Each block refines the token vectors, layer by layer, making them more and more context-aware.

---

## 2. Self-Attention

**Self-attention** is just the attention from Phase 4 where **every token attends to every other token in the same sequence** — the sequence attends to *itself*.

```
"the cat sat on the mat"
   ↑    ↑    ↑   ↑   ↑    ↑
each token looks at all others
```

The words "the" at position 0 and position 4 share context. Self-attention lets every token collect information from the whole sentence, no matter how far away.

In a block, the Q, K, and V are all computed from the same input (the token embeddings), which is why it's called **self**-attention: one source, three roles.

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

- Each head is a "specialist" that can learn different patterns.
- Their outputs are **concatenated** (glued together) and passed through a linear layer — the **output projection** — to combine the views.

> 📝 **Notation:** In the original Transformer paper and most courses, each head's projections are written $W_i^Q$, $W_i^K$, $W_i^V$ (the same idea as `W_Q`, `W_K`, `W_V` from Phase 4, but with a separate set per head), and the final blending layer is written $W^O$.

> 💡 **Analogy:** A team of detectives investigates the same sentence, each focusing on a different clue (grammar, references, tone...). Then they pool their findings.

Modern models use anywhere from 8 to 128 heads.

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

Remember Phase 1: normalization rescales numbers to a stable range. **Layer normalization** does this **per token vector**: it subtracts the mean and divides by the standard deviation of that token's numbers, then applies two learned scaling/shifting parameters.

```
normalized = (x − mean) / std × γ + β
```

(`γ` and `β` are learned numbers, so the model can undo the normalization if it wants.)

### Why?

- Keeps numbers stable as they flow through dozens of layers.
- Makes training faster and more reliable.
- Prevents "exploding" values (numbers growing out of control).

In a modern Transformer block, the typical order is:

```
Normalize → Multi-Head Attention → Residual add
Normalize → Feed-Forward Network → Residual add
```

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
- It enables clever optimizations (like the KV cache you'll meet in Phase 6).

> 💡 The key idea in both approaches: **"I am the 5th token, and I know the token before me is the 4th."**

---

## 8. Encoder vs. Decoder

The original Transformer paper had two halves:

| Part | Job | Sees... |
|---|---|---|
| **Encoder** | Understand the whole input | All tokens (both directions) |
| **Decoder** | Generate output one token at a time | Only tokens written so far (left to right) |

- **Encoder-only** models (like BERT) are great at *understanding* (classification, search).
- **Decoder-only** models (like GPT and most modern LLMs) are great at *generating* text — they are the family that powers ChatGPT.
- There are also **encoder-decoder** models (like the original Transformer for translation).

Modern chat models are **decoder-only**: they read your prompt and generate the answer one token at a time.

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

The upper-right triangle is set to "impossible" (a very negative number, so softmax gives it zero weight). This way the model can only use the past, never the future — exactly how real text generation works.

---

## 10. One Full Transformer Block (Summary)

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

Repeat this block many times. Add positional info up front (Phase 5, section 7). At the end, a final linear layer + softmax turns the last token's vector into probabilities for the next token (Phase 6 covers this).

---

## Why It All Matters

Each layer of a Transformer:

1. **Attention** lets tokens exchange information with each other (with positional awareness and causal masking).
2. **Residual + normalization** keep training stable.
3. **Feed-forward** lets each token individually "think" about what it learned.

Stack enough of these, and the model develops rich, hierarchical understanding of language — from grammar, to references, to reasoning.

---

## Suggested Mini-Project: Tiny Transformer

Use PyTorch to build a miniature Transformer and train it to predict the next token.

```python
import torch
import torch.nn as nn

class TinyTransformer(nn.Module):
    def __init__(self, vocab=1000, embed=64, heads=4, layers=2, ctx=32):
        super().__init__()
        self.embed = nn.Embedding(vocab, embed)
        self.pos   = nn.Embedding(ctx, embed)          # simple learned positions
        block = nn.TransformerEncoderLayer(
            d_model=embed, nhead=heads,
            dim_feedforward=embed * 4, batch_first=True)
        self.blocks = nn.TransformerEncoder(block, num_layers=layers)
        self.head = nn.Linear(embed, vocab)            # predict next token

    def forward(self, tokens):
        # tokens shape: (batch, sequence)
        pos = torch.arange(tokens.size(1), device=tokens.device)
        x = self.embed(tokens) + self.pos(pos)         # embeddings + positions
        # Causal mask: token i may attend only to tokens 0..i (no peeking ahead)
        seq = tokens.size(1)
        mask = torch.triu(torch.ones(seq, seq, device=tokens.device), diagonal=1).bool()
        x = self.blocks(x, mask=mask)                  # transformer blocks
        return self.head(x)                            # logits for next token

model = TinyTransformer()
# A tiny batch: batch=1, sequence of 8 token IDs
sample = torch.randint(0, 1000, (1, 8))
logits = model(sample)
print("logits shape:", logits.shape)   # (1, 8, 1000): one score per vocab per position
```

The last position's logits, after softmax, tell you the model's guess for the **next** token. Training it to match the real next token is exactly how GPT-style models learn. The causal mask inside `forward` is what stops the model from peeking at future tokens (Section 9) — without it, this wouldn't be a valid next-token model.

---

**Next step →** [Phase 6: LLM Concepts](Phase6_LLM_Concepts.md)
