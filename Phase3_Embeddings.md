# Phase 3 — Embeddings

> **Goal:** Understand how words become numbers that a computer can work with. This is the bridge between basic neural networks and Transformers.

---

## 1. The Problem: Computers Can't Read Words

A neural network can only do math with numbers. So before anything else, we must convert text into numbers.

The chain looks like this:

```
"cat"
   ↓  (tokenization)
token
   ↓  (lookup table)
token ID
   ↓  (embedding table)
embedding vector
   ↓
[0.21, -0.73, 0.14, ..., 0.82]
```

We'll unpack each step.

---

## 2. Tokenization: Splitting Text into Pieces

**Tokenization** is the process of splitting text into small pieces called **tokens**.

A token can be:

- A whole word: `"cat"` → one token
- A piece of a word: `"unhappiness"` → `"un"` + `"happi"` + `"ness"` (three tokens)
- A single character, punctuation, or even a space

There are four main granularities for tokenization:

| Type | Pros | Cons |
|---|---|---|
| **Word** | Easy to interpret; short sequences | Large vocabulary; word variations (`"cat"` vs. `"cats"`) are not handled |
| **Subword** | Word roots are leveraged (e.g. `"un-"`, `"-ness"`); intuitive embeddings | Longer sequences; tokenization is more complex |
| **Character** | No "unknown word" concerns | Much longer sequences; patterns are harder to learn |
| **Byte** | Very small vocabulary; can represent any text (even emoji) | Very long sequences; works below the level of meaning |

Modern models almost all use **subword** tokenization: common words stay whole, and rare words get split into smaller pieces. It's the sweet spot — a manageable vocabulary that can still represent any word.

The most common subword algorithms you'll hear about are:

- **Byte-Pair Encoding (BPE)** — starts from characters and repeatedly merges the most frequent pair into a new token.
- **Unigram** — starts from a large candidate list of tokens and prunes it down, keeping the most useful pieces.

(You may also see **WordPiece** — the variant used by BERT.)

Real tokenizers also reserve **special tokens**, such as `[UNK]` ("unknown" — a fallback used when a piece of text isn't in the vocabulary) and `[PAD]` (padding — fills empty positions so sequences in a batch have the same length).

**Example:**

```
"the cat sat"  →  ["the", "cat", "sat"]
"unhappiness"  →  ["un", "happi", "ness"]
```

### Why do we care?

Tokenization is the very first thing a model does with your text. Everything the model "knows" is built from these tokens.

---

## 3. Token IDs: Giving Each Token a Number

Next, every token in the model's **vocabulary** (the list of all tokens it knows) gets a unique **ID** — just an integer.

```
vocabulary (tiny example):
  "the"   → 0
  "cat"   → 1
  "sat"   → 2
  "mat"   → 3
  ...
```

So the sentence `"the cat sat"` becomes the IDs `[0, 1, 2]`.

The IDs are just **labels** — they carry no meaning about the words themselves. ID 1 and ID 2 aren't "closer" because their numbers are close. That's exactly why we need embeddings next.

---

## 4. Embedding Vectors: Giving Meaning to IDs

An **embedding** is a vector of numbers that represents a token. This is where meaning lives.

Each token ID maps to one row in an **embedding table** — a big matrix where:

- Each **row** = one token in the vocabulary.
- Each **column** = one dimension of the embedding.

```
embedding dimension = 4 (tiny example)

"the" → [0.10, 0.82, -0.30, 0.05]
"cat" → [0.21, -0.73, 0.14, 0.82]
"dog" → [0.19, -0.71, 0.15, 0.80]
"car" → [0.90, 0.10, 0.60, -0.25]
```

Notice something important: in this illustrative table, **"cat" and "dog" look similar**, while **"car" looks different**. During training, related usage patterns can lead to similar vectors, but this is not guaranteed for every model, token, layer, or kind of meaning.

### How are these numbers chosen?

**They're learned during training.** The embedding table is just another set of learned **parameters** — the same idea as the weights from Phase 2 — so training updates it along with everything else. The model starts with random numbers, then training reshapes them so that:

- Words used in similar ways end up **close together** in vector space.
- Unrelated words end up **far apart**.

**Why does similar usage push vectors together?** Because during training the model keeps making predictions ("what word comes next?", "what word is nearby?") that depend on the embeddings. When two words appear in the same kinds of contexts, the model's predictions improve if those words share features — so gradient descent (Phase 2) gradually nudges their vectors closer. It's a side effect of optimizing predictions, not something a person hand-designs.

A famous classic example: after training on lots of text, the model learns relationships like:

```
vector("king") − vector("man") + vector("woman") ≈ vector("queen")
```

Meaning the directions of the vectors encode relationships (gender, royalty, etc.). Treat this as a neat side effect of how embeddings organize themselves — the model isn't deliberately "doing arithmetic on concepts," and no single dimension cleanly means "royalty" or "gender."

### Embedding dimensions

Real models use embeddings with hundreds or thousands of dimensions (for example, 512 or 768 or 4096). More dimensions = more room to store subtle meaning, but also more computation.

---

## 5. Semantic Similarity: Measuring "Closeness" of Meaning

Once words are vectors, we can **measure how related two words are** by comparing their vectors.

The two standard tools:

### Dot product (from Phase 1)

```
a · b = sum of (a_i × b_i)
```

Large positive = pointing in the same direction = similar meaning.

### Cosine similarity

The dot product can be inflated just because vectors are long. **Cosine similarity** fixes this by measuring the **angle** between the vectors, ignoring their length.

```
cosine_similarity(a, b) = (a · b) / (|a| × |b|)
```

- `1` → vectors point exactly the same way (very similar).
- `0` → vectors point in unrelated directions.
- `-1` → vectors point in exactly opposite directions (usually very different meanings — though true antonyms like "hot"/"cold" can still end up with high similarity in some models, because they appear in similar contexts).

It's a common way to compare embedding vectors, although the best similarity measure depends on the model and the task.

---

## 6. What the Numbers "Mean" (Intuition)

Each dimension of an embedding is like a **feature knob**. The model learns what each knob represents (though it rarely maps neatly to a single human word like "animal-ness"). Some dimensions may respond to:

- Whether the word is an animal
- Whether it's a verb or a noun
- Whether it's positive or negative
- Abstract combinations of all of the above

You don't need to know what each dimension means. The important mental model:

> **A word is a point in space. Similar words live near each other.**

---

## 7. From Words to Sentences

When a model reads a sentence, it looks up the embedding for **every token** and stacks them into a matrix (one row per token).

```
"the cat sat"
   ↓
| embedding("the") |   ← row 0
| embedding("cat") |   ← row 1
| embedding("sat") |   ← row 2
```

This matrix is the model's initial understanding of the sentence. Every later phase (attention, the Transformer blocks) will refine these vectors, moving them around in space to capture more and more context.

---

## Summary

| Term | Meaning |
|---|---|
| Token | A piece of text (word or subword) |
| Vocabulary | The list of all tokens the model knows |
| Token ID | A unique integer label for each token |
| Embedding | A vector of numbers representing a token's meaning |
| Embedding table | A matrix: one row of numbers per token |
| Embedding dimension | How many numbers each token gets |
| Semantic similarity | How related two words are, measured by vector closeness |
| Cosine similarity | Similarity measured by the angle between vectors (ignores length) |

---

## Suggested Mini-Project: Word Embeddings

Create a tiny vocabulary with hand-made embeddings and compute similarities.

```python
import numpy as np

# Tiny hand-made embeddings (dimension = 3)
embeddings = {
    "king":   np.array([0.9, 0.9, 0.1]),
    "queen":  np.array([0.9, 0.8, 0.1]),
    "man":    np.array([0.8, 0.1, 0.1]),
    "woman":  np.array([0.8, 0.1, 0.0]),
    "car":    np.array([0.1, 0.9, 0.9]),
    "bicycle":np.array([0.1, 0.8, 0.9]),
}

def cosine_similarity(a, b):
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))

print('king vs queen:',  round(cosine_similarity(embeddings["king"], embeddings["queen"]), 3))
print('king vs car:',    round(cosine_similarity(embeddings["king"], embeddings["car"]), 3))
print('car vs bicycle:', round(cosine_similarity(embeddings["car"], embeddings["bicycle"]), 3))
```

Notice that `king`–`queen` and `car`–`bicycle` score high (similar), while `king`–`car` scores low (unrelated). This is the core idea that real language models build on at massive scale.

---

**Next step →** [Phase 4: Attention](Phase4_Attention.md)
