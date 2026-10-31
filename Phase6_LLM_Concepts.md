# Phase 6 — LLM-Specific Concepts

> **Goal:** Understand how the Transformer turns into a real Large Language Model (LLM) — how it generates text, how it's trained, and how it's made fast enough to run in production.

**Required core:** next-token targets and loss, training/validation, context limits, sampling, and the runnable capstone. Then study KV caching and prefill/decode. **Optional extensions:** Sections 7-9 and the later topics. The capstone is a tiny educational base language model, not a production LLM or chat assistant.

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

> 💡 **Key insight:** The model generates from a conditional next-token distribution rather than looking up an answer in a database. Greedy decoding chooses its largest probability; sampling can choose another token. Learned patterns do not guarantee factual accuracy or correct reasoning.

### Shift the targets, not the loss

For a six-token training sequence:

```
tokens:   [the, cat, sat, on, the, mat]
inputs:   [the, cat, sat, on, the]
targets:  [cat, sat, on, the, mat]
```

Each input position predicts the token **one place to its right**. The causal mask prevents that position from reading its target, even though all training positions are processed in parallel.

Training uses the true preceding tokens (**teacher forcing**); generation uses the model's own previously selected tokens. For logits `(B, T, V)` and integer targets `(B, T)`, flatten to `(B*T, V)` and `(B*T,)` for cross-entropy, with **no preceding softmax**. Here `V` is vocabulary size, not attention's Value matrix. Mean loss averages `-log(p_correct)` over the target positions.

### Base language model vs. instruction-tuned chat model

A base model learns text continuation. **Supervised fine-tuning (SFT)** on instructions and responses can teach it to respond as an assistant; other post-training methods may further shape preferences and behavior. A **chat template** marks roles and message boundaries using the format expected by that model. Next-token prediction alone does not imply that a tiny base model will follow instructions.

---

## 3. The KV Cache: Reusing Work

**Without caching**, a simple generation loop processes the whole prefix at every step. Much of that computation repeats.

At step 5, the model attends to tokens 1–4 again, even though it already computed their Key and Value vectors at earlier steps.

**The KV cache** stores past Keys and Values **separately in each attention layer**. In a causal model at inference time, adding a future token does not change past token representations, so those projections can be reused:

```
step 1: compute K,V for token 1      → save in cache
step 2: compute K,V for token 2      → reuse token 1 from cache
step 3: compute K,V for token 3      → reuse tokens 1-2 from cache
...
```

Now each step computes new projections and attention outputs only for the **new** token. Its query still attends to the cached past K/V: caching does not eliminate that growing amount of attention work, nor does it store answers.

The KV cache is also why **context length** has a cost — the longer the conversation, the bigger the cache in memory.

For standard attention, each layer's K or V cache has shape `(B, H, T, d_head)`. With `L` layers, nominal K/V storage is:

```
2 * L * B * H * T * d_head * bytes_per_element
```

The `2` counts K and V. GQA/MQA use the number of **KV heads**, not query heads, in this expression. Caching works with additive positions too; RoPE is not a prerequisite.

### Small cache-equivalence exercise

Reuse `attention` from Phase 4 in the same Python session. This isolates **one attention layer** with already-computed projections; a real Transformer maintains a cache for every layer.

```python
import numpy as np

cache_Q = np.array([[0.8, 0.2], [0.5, 0.5], [0.3, 0.7]])
cache_K = np.array([[0.9, 0.1], [0.4, 0.6], [0.1, 0.9]])
cache_V = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
cached_keys = np.empty((0, 2))
cached_values = np.empty((0, 2))
cached_outputs, full_outputs = [], []

for position in range(len(cache_Q)):
    cached_keys = np.concatenate((cached_keys, cache_K[position:position + 1]))
    cached_values = np.concatenate((cached_values, cache_V[position:position + 1]))
    _, cached = attention(cache_Q[position:position + 1], cached_keys, cached_values)
    prefix_mask = np.triu(np.ones((position + 1, position + 1), dtype=bool), k=1)
    _, full = attention(
        cache_Q[:position + 1], cache_K[:position + 1], cache_V[:position + 1],
        mask=prefix_mask)
    cached_outputs.append(cached[0])
    full_outputs.append(full[-1])

np.testing.assert_allclose(cached_outputs, full_outputs, rtol=0, atol=1e-12)
print("cached and full-prefix attention outputs match")
```

Check that only one query is needed for the new token and that the cache grows one row at a time. This is a correctness check, not a serving-speed benchmark.

---

## 4. Context Window

The **context window** is the maximum amount of text (in tokens) the model can "see" at once — your prompt plus everything it has generated so far.

- Modern models: 8K, 32K, 128K, even millions of tokens.
- When the context fills up, the system must stop, select a window, or explicitly truncate/summarize history. The model does not automatically summarize discarded text.

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

The formula requires a positive temperature. In the capstone, **temperature `0` is an explicit greedy-decoding switch**, not division by zero. Temperature changes sampling sharpness, not the truthfulness of an answer.

> 💡 **Analogy:** Low temperature = a careful, conservative speaker. High temperature = an adventurous, freewheeling one.

### Top-k sampling

Only consider the **k most likely** tokens, renormalize their probabilities, and sample from that distribution. This is not a uniform choice among the retained tokens.

### Top-p sampling (nucleus sampling)

Sort tokens by descending probability, retain the smallest set whose **combined probability reaches `p`** (e.g., 0.9), renormalize, and sample. Include the token that crosses the threshold.

- When the model is confident, that set is small (few good options).
- When it's uncertain, the set is larger (more options).

> 💡 **Analogy:** Top-k says "keep the 10 best candidates." Top-p says "keep as many candidates as needed to cover 90% of the probability."

Systems may use temperature, top-k, top-p, greedy decoding, beam search, or other decoding strategies. These controls are configurable and are not necessarily used together.

---

## 6. Prefill and Decode: The Two Phases of Generation

Every generation request has two phases:

| Phase | What happens | Computation pattern |
|---|---|---|
| **Prefill** | Process the prompt and, when using a cache, populate its per-layer K/V | Many positions can be processed in parallel, sometimes in chunks; long prompts can still have substantial latency |
| **Decode** | Process selected new tokens and produce subsequent predictions | Ordinary autoregressive decoding is sequential across generated tokens; each query still attends to its available context |

Prefill produces the logits used to select the first new token. The KV cache is then reused and extended during decode. Parallelism does not guarantee that prefill is always faster than decode; latency depends on prompt length, output length, batching, model, and hardware.

---

## 7. Inference Optimization: Making It Fast

**Optional extension:** study this after training and generating with the baseline.

Running an LLM costs money and time, so there's a whole industry of tricks to make it faster:

- **KV cache** (above): avoid recomputing.
- **Quantization** (below): smaller numbers, faster math.
- **FlashAttention:** a memory-efficient way to compute attention that reduces unnecessary reads and writes. It can improve speed, but overall serving speed also depends on hardware, kernels, batching, precision, and model architecture. (More in "later topics".)
- **GQA / MQA:** share Keys and Values across attention heads to shrink the KV cache. (More in "later topics".)
- **Speculative decoding:** use a small, fast model to draft several tokens, then have the larger model verify them in parallel. It can improve latency, but the speedup depends on the models and workload. (More in "later topics".)

---

## 8. Quantization: Smaller Numbers, Faster Models

**Optional extension:** the capstone uses ordinary floating-point weights, not quantization.

The weights of a trained model are usually stored as **32-bit or 16-bit floating-point numbers**. **Quantization** stores them with fewer bits (8-bit or 4-bit), which:

- **Shrinks weight storage** (4-bit versus 16-bit is nominally 4× smaller, before scales, metadata, and unquantized components).
- **Can speed up computation** when the hardware and kernels support the quantized format; otherwise its main benefit may be reduced memory use.
- **Uses less memory**, so bigger models fit on smaller hardware.

The trade-off: quantization can reduce accuracy, with the size of the loss depending on the model, data, method, and bit width.

Weight quantization does not automatically quantize the KV cache.

> 💡 **Analogy:** Writing a phone number with fewer digits saves space, but rounding too much loses information. Quantization finds the sweet spot.

This is why you can run surprisingly large LLMs on a laptop — they're often 4-bit quantized versions.

---

## 9. Fine-Tuning and LoRA: Teaching an LLM a New Skill

**Optional extension:** first establish a trained baseline and evaluate it on held-out data.

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
4. The output head produces logits; the decoding rule turns them into a selected token.
5. Append that token to the sequence.
6. Unless stopping, process the new token, extend the per-layer KV cache,
   and produce the next logits.
   → decode: repeat steps 4-6.
7. Stop at a configured end/stop token, generated-token limit, or context/budget limit.
```

That's the core text-generation loop of a decoder-based LLM. A production chat system may also add retrieval, tool calls, safety checks, multimodal processing, and other application logic around the model.

## Suggested Mini-Project: Train, Evaluate, and Generate

Use the [runnable capstone](examples/tiny_transformer.py) and the [Python setup](README.md#prerequisites-and-setup). From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe .\examples\tiny_transformer.py
```

This runs a **CPU-only** model with character tokens, width `32`, four heads, two pre-norm blocks, and context length `32`. The vocabulary comes from the training text. Its original toy sentences are bundled in the script; no dataset/model download, GPU, or API key is needed.

Follow these steps in the source:

1. `encode` converts characters to integer IDs. Validation text uses a separate set of sentences and is never used for optimizer updates.
2. `make_examples` takes windows of `context + 1` tokens and returns shifted inputs/targets, each `(B, T)`.
3. `TinyTransformer` produces raw logits `(B, T, V)` with learned additive positions and causal masking.
4. `next_token_loss` computes cross-entropy on logits. `train` clears gradients, calls `backward`, and updates parameters with AdamW.
5. `evaluate` records train and held-out validation loss before and after training. **Perplexity** is `exp(mean cross-entropy)` for this tokenizer and corpus; it is not comparable across arbitrary tokenizations or datasets.
6. `generate` repeatedly evaluates the current prefix and appends a selected token. It stops at a newline (this toy model's configured stop token) or the requested token limit.

The script prints initial/final losses, validation perplexity, and a generated continuation. The default seeded run should reduce both losses; inspect the numbers rather than treating a correctly shaped output as proof of learning. Exact values can vary across library versions.

**Deliberate limits:** this baseline has **no KV cache**, uses a tiny highly patterned corpus, and trains from scratch on each run. Lower held-out loss here does not establish broad language understanding. Prompt plus generation budget must fit its 32-token context; unsupported characters and oversized requests produce explicit errors rather than silently dropping input.

Experiment with decoding:

```powershell
.\.venv\Scripts\python.exe .\examples\tiny_transformer.py --temperature 0
.\.venv\Scripts\python.exe .\examples\tiny_transformer.py --temperature 0.8 --top-k 5 --top-p 0.9
```

`top-k 0` disables top-k filtering; `top-p 1` disables nucleus filtering. Each command retrains the same seeded baseline before generating.

Run the regression checks:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Before Moving On

| Check | Completion criterion |
|---|---|
| By hand | Shift the six-token example into five inputs/targets and calculate `-log(0.02) ≈ 3.912` for one correct-token probability. |
| In code | Record initial/final train and validation loss, generate within the token/context budget, and reproduce the cache-equivalence check. The seeded capstone regression test requires at least a 20% reduction in held-out loss on this toy corpus. |
| In your own words | Explain teacher forcing versus generation, logits versus probabilities, what per-layer K/V stores, and why neither lower loss nor lower temperature guarantees a correct answer. |

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
| Temperature | Controls distribution sharpness / randomness of sampling, not factual correctness |
| Top-k / Top-p | Limit which tokens are allowed to be chosen |
| Prefill / Decode | Process prompt in parallel, then generate one token at a time |
| Quantization | Fewer bits per weight → less storage; speed depends on hardware and kernels |
| LoRA | Train tiny add-on matrices instead of all weights |
| Fine-tuning | Extra training on specialized data |

---

**That's the end of the core roadmap!** Use the phase checks and capstone results to establish that you can trace, implement, train, and test the pipeline. Reading the terms alone does not establish mastery, and production-scale training and serving remain further study.
