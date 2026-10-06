# Phase 6 — LLM-Specific Concepts

> **Goal:** Understand how the Transformer turns into a real Large Language Model (LLM) — how it generates text, how it's trained, and how it's made fast enough to run in production.

---

## 1. The Big Idea: Predicting the Next Token

An LLM is, at its core, a **next-token predictor** (Phase 5). It reads a sequence of tokens and predicts which token comes next.

```
input:  "The capital of France is"
output: probabilities for the next token:
        "Paris"   → 0.85
        "Lyon"    → 0.04
        "French"  → 0.02
        ...
```

### Autoregressive generation

To write a full answer, the model **feeds its own output back in** — one token at a time:

```
"The capital of France is"  → predicts "Paris"
"The capital of France is Paris"  → predicts "."
"The capital of France is Paris."  → predicts "It"
... and so on ...
```

This loop — predict one token, append it, predict again — is called **autoregressive generation** ("auto" = self, "regressive" = using its own past output). Every ChatGPT response is built this way, token by token.

---

## 2. How Training Works: Next-Token Prediction + Cross-Entropy

During training, the model is shown **many real sentences**. For every position, it tries to predict the next token, and we measure how wrong it was.

**Cross-entropy loss** (from Phase 2) is the scoreboard:

- Correct word gets high probability → small loss.
- Correct word gets low probability → big loss.

```
training sentence: "The cat sat on the mat"
at position 3 ("sat"), the model should predict "on":
  model says:  "on" → 0.03   (oops, very wrong → big loss)
after more training:
  model says:  "on" → 0.92   (good → small loss)
```

Backpropagation (Phase 2) then nudges the weights so the model gets better at predicting. Trained on trillions of tokens of text, the model learns grammar, facts, patterns, and reasoning — all as a side effect of getting really good at next-token prediction.

> 💡 **Key insight:** An LLM doesn't "look up" answers. It has learned statistical patterns from text, and it generates what is *most likely* next.

---

## 3. The KV Cache: Reusing Work

When generating, the model processes the whole sequence at **every step** — but most of it is repeated work!

At step 5, the model attends to tokens 1–4 again, even though it already computed their Key and Value vectors at earlier steps.

**The KV cache** is simply a memory that stores the Keys and Values of all tokens computed so far:

```
step 1: compute K,V for token 1      → save in cache
step 2: compute K,V for token 2      → reuse token 1 from cache
step 3: compute K,V for token 3      → reuse tokens 1-2 from cache
...
```

Now each step only computes attention for the **new** token, making generation dramatically faster. This is why the KV cache matters: **it turns repeated computation into a fast lookup.**

The KV cache is also why **context length** has a cost — the longer the conversation, the bigger the cache in memory.

---

## 4. Context Window

The **context window** is the maximum amount of text (in tokens) the model can "see" at once — your prompt plus everything it has generated so far.

- Modern models: 8K, 32K, 128K, even millions of tokens.
- When the context fills up, old information must be dropped or summarized.

> 💡 **Analogy:** The context window is the model's working desk. Everything on the desk can be referenced. When the desk is full, you have to clear space.

A longer context window lets the model remember more of a long document or conversation — but it uses more memory (especially the KV cache) and more computation.

---

## 5. Sampling: Choosing the Next Token

The model outputs **probabilities** for every possible next token. But which one should it actually pick?

- **Always pick the most likely** (greedy) → predictable, but often boring and repetitive.
- **Pick randomly from the distribution** → more varied, but can go off the rails.

Real LLMs use tricks to balance creativity and coherence:

### Temperature

Temperature controls **how "confident" or "flat"** the probability distribution is.

```
probability_i ∝ exp(score_i / temperature)
```

- **Low temperature (e.g., 0.2):** sharpens the distribution → the model almost always picks the top choice → focused, predictable output.
- **High temperature (e.g., 1.5):** flattens the distribution → unlikely words become more likely → creative, surprising, sometimes chaotic output.
- **Temperature 1.0:** the natural distribution (no change).

> 💡 **Analogy:** Low temperature = a careful, conservative speaker. High temperature = an adventurous, freewheeling one.

### Top-k sampling

Only consider the **k most likely** tokens, and pick randomly among them. If `k = 50`, the model ignores the long tail of very unlikely tokens (which are often nonsense).

### Top-p sampling (nucleus sampling)

Consider only the smallest set of tokens whose **combined probability reaches `p`** (e.g., 0.9), and pick among those.

- When the model is confident, that set is small (few good options).
- When it's uncertain, the set is larger (more options).

> 💡 **Analogy:** Top-k says "keep the 10 best candidates." Top-p says "keep as many candidates as needed to cover 90% of the probability."

Systems may use temperature, top-k, top-p, greedy decoding, beam search, or other decoding strategies. These controls are configurable and are not necessarily used together.

---

## 6. Prefill and Decode: The Two Phases of Generation

Every generation request has two phases:

| Phase | What happens | Speed |
|---|---|---|
| **Prefill** | The model processes your **entire prompt at once**, computing all K/V in parallel | Very fast (parallel) |
| **Decode** | The model generates tokens **one at a time**, each depending on the previous | Slower (sequential) |

The KV cache is filled during prefill, then reused during decode. Since decode is one token at a time, total generation time scales with the number of tokens you ask for.

---

## 7. Inference Optimization: Making It Fast

Running an LLM costs money and time, so there's a whole industry of tricks to make it faster:

- **KV cache** (above): avoid recomputing.
- **Quantization** (below): smaller numbers, faster math.
- **FlashAttention:** a memory-efficient way to compute attention that reduces unnecessary reads and writes. It can improve speed, but overall serving speed also depends on hardware, kernels, batching, precision, and model architecture. (More in "later topics".)
- **GQA / MQA:** share Keys and Values across attention heads to shrink the KV cache. (More in "later topics".)
- **Speculative decoding:** use a small, fast model to draft several tokens, then have the larger model verify them in parallel. It can improve latency, but the speedup depends on the models and workload. (More in "later topics".)

---

## 8. Quantization: Smaller Numbers, Faster Models

The weights of a trained model are usually stored as **32-bit or 16-bit floating-point numbers**. **Quantization** stores them with fewer bits (8-bit or 4-bit), which:

- **Shrinks the model** (4-bit models can be ~4× smaller).
- **Can speed up computation** when the hardware and kernels support the quantized format; otherwise its main benefit may be reduced memory use.
- **Uses less memory**, so bigger models fit on smaller hardware.

The trade-off: quantization can reduce accuracy, with the size of the loss depending on the model, data, method, and bit width.

> 💡 **Analogy:** Writing a phone number with fewer digits saves space, but rounding too much loses information. Quantization finds the sweet spot.

This is why you can run surprisingly large LLMs on a laptop — they're often 4-bit quantized versions.

---

## 9. Fine-Tuning and LoRA: Teaching an LLM a New Skill

**Fine-tuning** means taking a pre-trained model (which already knows language) and training it a bit more on specialized data:

- A customer-support chat dataset → makes it better at support.
- Medical texts → makes it better at medical topics.

Pre-training is like going to school for years; fine-tuning is like a short specialist course.

### LoRA (Low-Rank Adaptation)

Full fine-tuning changes **all** billions of weights — expensive. **LoRA** is a clever shortcut:

- Instead of updating the big weight matrices, LoRA trains **two small matrices** whose product approximates the change.
- The original weights stay frozen; the small LoRA matrices are added on top.

```
new_weights ≈ original_weights + (small_A × small_B)
                    frozen            trained (tiny)
```

Benefits:

- Trains **much fewer** parameters (often < 1% of the model).
- Produces **tiny** update files you can share (a few MB instead of GBs).
- Multiple LoRA "add-ons" can be swapped onto one base model.

> 💡 **Analogy:** Instead of renovating the whole building, you add removable furniture (LoRA) that changes how rooms feel — and you can swap the furniture per task.

---

## 10. Putting It All Together: What Happens When You Ask a Question

```
1. Your prompt is tokenized into token IDs.
2. Embeddings + positional info turn them into vectors.
3. Transformer blocks (attention + feed-forward) process them.
   → prefill: KV cache is built for your prompt.
4. The model outputs probabilities for the next token.
5. Sampling (temperature, top-k, top-p) picks a token.
6. The token is appended; the KV cache is updated.
   → decode: repeat steps 4-6, one token at a time.
7. Generation stops when the model outputs an end-of-text token.
```

That's the core text-generation loop of a decoder-based LLM. A production chat system may also add retrieval, tool calls, safety checks, multimodal processing, and other application logic around the model.

---

## Later Topics (a Quick Preview)

Once the above is comfortable, these become much easier:

- **FlashAttention** — faster attention via smarter memory use.
- **GQA / MQA** — sharing K/V across heads to shrink the cache.
- **Mixture of Experts (MoE)** — many specialized sub-networks, only some active per token.
- **RoPE in depth** — rotary positions and how they enable long contexts.
- **Speculative decoding** — small model drafts, big model verifies.
- **Distributed inference** — splitting a giant model across many GPUs.
- **Large-scale training** — data, parallelism, and the cost of building LLMs.

---

## Summary Cheat Sheet

| Term | One-line meaning |
|---|---|
| Next-token prediction | The core task: predict the token that comes next |
| Autoregressive generation | Feed output back in, one token at a time |
| Cross-entropy loss | Measures how wrong the token probabilities are |
| KV cache | Stored Keys/Values so we don't recompute them |
| Context window | Max tokens the model can see at once |
| Temperature | Controls confidence / randomness of sampling |
| Top-k / Top-p | Limit which tokens are allowed to be chosen |
| Prefill / Decode | Process prompt in parallel, then generate one token at a time |
| Quantization | Fewer bits per weight → smaller, faster models |
| LoRA | Train tiny add-on matrices instead of all weights |
| Fine-tuning | Extra training on specialized data |

---

**That's the end of the roadmap!** You now have a complete mental model: from vectors and matrices, through neural networks, embeddings, attention, the full Transformer, and finally how real LLMs generate text. Keep building the mini-projects — hands-on practice turns this understanding into skill.
