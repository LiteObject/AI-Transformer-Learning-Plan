# AI Transformer Learning Plan

A practical, math-friendly study path from fundamentals to Transformers and LLMs. No prior machine learning or advanced mathematics background is required; the exercises assume basic Python.

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
| [Transformer_Architecture_Infographic.html](Transformer_Architecture_Infographic.html) | Interactive animated companion to Phase 5: the three Transformer designs (encoder-only, decoder-only, encoder-decoder) side by side, with step-through captions, hover explanations, and labeled attention masks. |
| [Phase6_LLM_Concepts.md](Phase6_LLM_Concepts.md) | Shifted next-token targets, training/validation, generation, KV cache, context windows, sampling, and prefill/decode; quantization and LoRA are optional extensions. |
| [How_LLMs_Generate_Text_Infographic.html](How_LLMs_Generate_Text_Infographic.html) | Visual companion for token generation, attention shapes, KV caching, and prefill/decode. |
| [examples/tiny_transformer.py](examples/tiny_transformer.py) | Runnable CPU-only capstone: character tokenization, a causal pre-norm Transformer, training, held-out evaluation, and bounded generation. |
| [tests/test_learning_path.py](tests/test_learning_path.py) | Regression checks for the documented exercises, tensor shapes, causal masking, sampling, and capstone learning. |
| [requirements.txt](requirements.txt) | NumPy and PyTorch dependencies for the exercises. |
| [Neural_Network_vs_Transformer.md](Neural_Network_vs_Transformer.md) | How a Transformer relates to other neural network architectures (feedforward, CNN, RNN/LSTM/GRU) and why self-attention replaced recurrence. |
| [Abbreviations.md](Abbreviations.md) | Quick lookup for every acronym and symbol used across the guides, each linked to the page where it is explained. |

## Learning Roadmap

| Phase | Focus | Key Outcome |
|---|---|---|
| 1 | Essential Math | Calculate dot products and matrix outputs, track shapes, and check a numerical derivative. |
| 2 | Neural Networks | Trace forward/backward passes and train a tiny classifier with decreasing loss. |
| 3 | Embeddings | Map tokens to vectors and distinguish input embeddings from contextual hidden states. |
| 4 | Attention | Implement projected Q/K/V, row-wise softmax, value mixing, and a verified causal mask. |
| 5 | Transformer Architecture | Trace batched multi-head shapes and assemble a causal pre-norm Transformer. |
| 6 | LLM Concepts | Train/evaluate a small language model, generate within limits, and verify a cache-equivalence example. |

## How to Use These Guides

1. **Read the phases in order.** Each one builds directly on the previous: math → neural networks → embeddings → attention → architecture → LLMs.
2. **Use just-in-time math.** Start with Phase 1's vectors, shapes, and probability; revisit calculus in Phase 2 and statistics in Phase 5 rather than waiting to master all the math.
3. **Work with tiny numbers.** Two- or three-dimensional vectors are usually enough to understand the core idea.
4. **Build intuition before formulas.** Understand what an operation is doing before committing the notation to memory.
5. **Implement as you go.** A small Python/NumPy implementation turns an abstract equation into something concrete.
6. **Complete each phase's "Before Moving On" checks.** Do one calculation by hand, reproduce the code results, and explain the idea without copying the guide.

## Practice Projects

The [learning plan](AI_Transformer_Learning_Plan.md) defines six projects that mirror the phases. Every phase includes practical work and explicit completion checks.

| # | Project | What you build |
|---|---|---|
| 1 | Vector Playground | Vectors, dot products, matrix multiplication, and cosine similarity with Python/NumPy. |
| 2 | Tiny Neural Network | A hand-traced forward/backward pass and a small trained classifier. |
| 3 | Word Embeddings | A handmade vocabulary, similarity comparisons, and cosine's invariance to positive scaling. |
| 4 | Attention From Scratch | Q/K/V projections, scaled scores, row-wise softmax, value mixing, and causal-mask checks. |
| 5 | Multi-Head Attention | Batched head splitting/combining, followed by assembly of the reusable PyTorch model. |
| 6 | Tiny Transformer | Shifted targets, CPU training, separate validation loss/perplexity, bounded generation, and a small cache-equivalence check. |

## Prerequisites and Setup

- Basic arithmetic and high-school algebra are enough to start; the guides introduce everything else.
- Basic Python: variables, lists, loops, functions, and running a script. If those are new, learn them before the coding exercises; no prior NumPy or machine learning is assumed.
- Python **3.12** is the setup used for validation here. The exercises use NumPy, then PyTorch in Phases 5-6.
- No GPU, model/dataset download, or API key is required. The capstone bundles original toy text and runs on CPU.

From the repository root in **PowerShell**:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Installing CPU PyTorch first avoids fetching an unnecessary GPU build. These commands use the environment's executable directly, so activation is not required.

For interactive guide examples, start `.\.venv\Scripts\python.exe`. Phase 5's NumPy example and Phase 6's cache example reuse the `attention` function: run the Phase 4 Python block first in that same session.

## Run the Capstone and Checks

```powershell
.\.venv\Scripts\python.exe .\examples\tiny_transformer.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The capstone reports train/validation loss before and after training, validation perplexity, and a generated continuation. Its regression test requires at least a **20% reduction in held-out loss** for the default seeded toy setup. Tests also check documented calculations, shapes, future-token isolation, cache equivalence, sampling boundaries, and local documentation links.

This is a small base language model, not an instruction-following assistant. The held-out sentences are highly patterned; improvement here does not establish broad language competence. Each run trains from scratch and deliberately recomputes the prefix rather than implementing a production KV cache.

## Later Topics

After the required Phase 6 capstone, study its optional quantization/LoRA sections and previews of FlashAttention, GQA/MQA, Mixture of Experts, RoPE in greater depth, speculative decoding, distributed inference, and large-scale training.

## End Goal

By completing the calculations, implementations, and checks, you should be able to trace a Transformer from input IDs to logits, train a small model, measure its held-out loss, and explain the tensor dimensions. That foundation makes KV cache, context length, RoPE, and later optimizations easier to reason about; reading alone is not a completion criterion.
