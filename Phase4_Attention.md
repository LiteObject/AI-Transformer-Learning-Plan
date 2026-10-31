# Phase 4 — Attention

> **Goal:** Understand attention — the single most important idea in the Transformer — piece by piece, with intuition first and math second. Do **not** memorize the formula. Build it from small operations.

Start with one head and unmasked attention. The mini-project then adds a causal mask; Phase 5 expands the same calculation into multiple heads and batches.

---

## 1. What Is Attention Trying to Do?

When you read a sentence, some words matter more than others for understanding each word.

> "The animal didn't cross the street because **it** was too tired."

What does **"it"** refer to? A model may learn to use information from **"animal"**, rather than **"street"**, when processing this sentence. Attention helps tokens exchange information, but an attention weight by itself does not prove that the model has resolved a reference exactly as a person would. 

```
"it" → looks at all other words →
      "animal" gets high attention  (an illustrative learned pattern)
      "street" gets low attention
      "cross"  gets medium attention
```

In general:

> **Attention determines how much each token should pay attention to other tokens, then blends their information together weighted by those amounts.**

---

## 2. The Cast: Query, Key, and Value

Attention works like a **library search**. Every token gets three roles — but here's the key idea: the same token plays different roles depending on whether it's *asking* or *being asked*.

| Role | Analogy | What it does |
|---|---|---|
| **Query (Q)** | A question you're asking | "Who should I look at?" |
| **Key (K)** | A label on a library book | "Here's what I contain / what I am about" |
| **Value (V)** | The book's actual content | "Here's my actual information" |

**How it works:** for each token, you take its **Query** and compare it against every other token's **Key**. If a Key matches your Query well, you pull a lot of that token's **Value** into your understanding.

### Where do Q, K, V come from?

They're created by multiplying each token's embedding by **learned weight matrices** (Phase 2!). Each token starts with one embedding vector and gets turned into three vectors:

```
Q = embedding × W_Q
K = embedding × W_K
V = embedding × W_V
```

The model learns what makes a good question (Q), a good label (K), and useful content (V).

In the first block, the input comes from token embeddings plus any additive positions. In later blocks, Q/K/V are projections of that block's **input hidden states**, not fresh lookups of the original token embeddings.

---

## 3. Step 1: Score How Well Queries Match Keys (Dot Product)

To find out how much token A should pay attention to token B, we ask: *"Does A's question match B's label?"*

We measure that match with the **dot product** (Phase 1):

```
score(A, B) = Q_A · K_B
```

A big score → A's question matches B's label → B should get a lot of A's attention.

**Tiny example:**

```
Q for "it"     = [0.8, 0.2]
K for "animal" = [0.9, 0.1]   →  0.8×0.9 + 0.2×0.1 = 0.74   (high)
K for "street" = [0.1, 0.8]   →  0.8×0.1 + 0.2×0.8 = 0.24   (low)
```

So "it" matches "animal" much more than "street". Intuition confirmed by the math.

**Doing all tokens at once:** with matrix multiplication, we compute every score at once:

```
Scores = Q × Kᵀ
```

Here `Kᵀ` is the Key matrix flipped (transposed) so the shapes line up. The result is a table where row `i`, column `j` = "how much token i should attend to token j."

### Follow the shapes

For one self-attention head, let `T` be the token count, `D` the input width, `d_k` the query/key width, and `d_v` the value width:

| Tensor / operation | Shape |
|---|---|
| Input `X` | `(T, D)` |
| `W_Q`, `W_K` | `(D, d_k)` |
| `W_V` | `(D, d_v)` |
| `Q = X @ W_Q`, `K = X @ W_K` | `(T, d_k)` |
| `V = X @ W_V` | `(T, d_v)` |
| `Q @ K.T` | `(T, T)` |
| `weights @ V` | `(T, d_v)` |

Softmax runs **across the keys in each row**, not across the whole matrix. In cross-attention, queries and keys may have different token counts; the score table is then `(query_tokens, key_tokens)`.

---

## 4. Step 2: Scaling (Why Do We Divide by √dₖ?)

As the **query/key width** grows, dot products can have larger variance. Under the simple assumption of independent, zero-mean, unit-variance components, that variance is `d_k`; dividing by `sqrt(d_k)` brings it back to approximately `1`. Very large score differences can saturate softmax and make its gradients small.

**The fix:** divide every score by the square root of the key dimension, `√dₖ` (read: "the square root of the key dimension").

```
scaled_score = (Q · K) / √dₖ
```

It's just a **volume knob** that keeps the scores in a comfortable range so softmax behaves well. Nothing more mysterious than that.

---

## 5. Step 3: Softmax — Turning Scores into Weights

Now we have raw scores. We convert them into **attention weights** (numbers between 0 and 1 that add up to 1) using **softmax** (from Phase 2).

```
raw scores for "it"  →  [0.74, 0.24, 0.30]
divide by sqrt(2)   →  [0.523, 0.170, 0.212]
after softmax       →  [0.411, 0.288, 0.301]   (rounded; adds up to 1.0)
```

The attention weights answer: *"Out of all the attention I have to give, how much goes to each other token?"*

- They add to 1, so the total is always balanced.
- They're nonnegative, so they form a weighted average of the Value vectors. The Value vectors themselves, and therefore the resulting output, can still contain negative numbers.

---

## 6. Step 4: The Weighted Sum — Gathering Information

Finally, each token gathers information from all other tokens by taking a **weighted sum** of their **Values**:

```
output for token A = (weight₁ × V₁) + (weight₂ × V₂) + (weight₃ × V₃) + ...
```

- Tokens with **high attention weight** contribute a lot to the result.
- Tokens with **low attention weight** contribute almost nothing.

> 💡 **Analogy:** You're blending a smoothie. Each ingredient (other token's Value) is added in proportion to how much the recipe (attention weight) calls for it. "Animal" is the star ingredient for "it"; "street" is barely a splash.

The result: every token now carries **context** — it's no longer just itself, it's a blend of itself and the tokens it found relevant.

---

## 7. The Whole Equation, Now That You Know Each Part

```
Attention(Q, K, V) = softmax( QKᵀ / √dₖ ) V
```

Read it left to right with what you now know:

1. `QKᵀ` → how well every query matches every key (the scores table).
2. `÷ √dₖ` → scale the scores to keep them stable.
3. `softmax(...)` → turn scores into attention weights (rows sum to 1).
4. `× V` → blend the values using those weights.

**Don't memorize it — you've already built it.**

---

## 8. What Comes Out of Attention

- Every token gets a **new vector** that is a context-aware blend of other tokens.
- The model can see **long-range relationships**: "it" connects to "animal" even though they're far apart in the sentence.
- **Full attention can directly connect distant tokens within its available context.** Masks, sliding-window designs, and context limits can restrict those connections.
- RNNs can also carry long-range information through hidden states, and attention existed before Transformers. Transformers make direct token interactions possible without recurrent processing, allowing parallel computation across token positions during training and prefill.

---

## Suggested Mini-Project: Attention From Scratch

Implement the whole thing in a few lines of NumPy:

```python
import numpy as np

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def attention(Q, K, V, mask=None):
    if any(x.ndim != 2 or x.size == 0 for x in (Q, K, V)):
        raise ValueError("Q, K, V must be nonempty two-dimensional arrays")
    if Q.shape[1] != K.shape[1] or K.shape[0] != V.shape[0]:
        raise ValueError("query/key widths and key/value token counts must match")
    if not all(np.isfinite(x).all() for x in (Q, K, V)):
        raise ValueError("Q, K, V must contain finite numbers")
    d_k = K.shape[-1]                       # key dimension
    scores = Q @ K.T                        # step 1: match queries to keys
    scores = scores / np.sqrt(d_k)          # step 2: scale
    if mask is not None:
        if mask.shape != scores.shape or mask.dtype != np.bool_:
            raise ValueError("mask must be boolean and have the score-table shape")
        if mask.all(axis=-1).any():
            raise ValueError("each query must have at least one unmasked key")
        scores = np.where(mask, -np.inf, scores)
    weights = softmax(scores)               # step 3: to probabilities
    output = weights @ V                    # step 4: weighted sum of values
    return weights, output

# Tiny illustrative projections; training would learn these weight matrices.
X = np.array([[0.8, 0.2], [0.5, 0.5], [0.3, 0.7]])
W_Q = np.eye(2)
W_K = np.array([[0.9, 0.1], [0.1, 0.9]])
W_V = np.eye(2)
Q, K, V = X @ W_Q, X @ W_K, X @ W_V

weights, output = attention(Q, K, V)
print("attention weights:\n", np.round(weights, 3))
print("output:\n", np.round(output, 3))

mask = np.triu(np.ones((len(X), len(X)), dtype=bool), k=1)
causal_weights, causal_output = attention(Q, K, V, mask=mask)
assert np.allclose(causal_weights.sum(axis=-1), 1)
assert np.all(causal_weights[mask] == 0)
print("causal weights:\n", np.round(causal_weights, 3))
```

Look at the weights table: each row shows how one token spreads its attention across the others. This tiny program is the heart of the entire Transformer — everything after this builds on it.

Here, `True` means **blocked**, matching the PyTorch boolean-mask convention used in Phase 5. The diagonal stays available: a token may attend to itself. A fully blocked row is rejected because softmax cannot form a probability distribution from it.

## Before Moving On

| Check | Completion criterion |
|---|---|
| By hand | Reproduce the score `0.74` for the earlier query/key example and its scaled value of approximately `0.523` when `d_k = 2`. |
| In code | Compute Q/K/V from `X`, verify every attention row sums to `1`, and verify every masked weight is `0`. Changing a future value must not change an earlier causal output; confirm the unmasked version can change. |
| In your own words | Explain why Q/K determine weights but V supplies the blended information, why softmax is row-wise, and how a mask limits otherwise available connections. |

---

**Next step →** [Phase 5: Transformer Architecture](Phase5_Transformer_Architecture.md)
