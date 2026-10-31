# Neural Network vs. Transformer

> **Key insight:** A transformer **is** a neural network — it's a specific *architecture* of neural network, not a separate thing. The real question is how it differs from the *other* neural network architectures.

## The relationship

```
Neural Networks
├── Feedforward (MLP) — the "vanilla" network
├── Convolutional (CNN) — for images/grids
├── Recurrent (RNN/LSTM/GRU) — for sequences
└── Transformer — for sequences (and more)
```

## The core differences

| Aspect | Traditional NN (feedforward/RNN) | Transformer |
|---|---|---|
| **Core building block** | Neurons → layers, weighted sums + activation | **Self-attention** + feedforward layers |
| **Processing order** | RNNs read tokens one at a time, in order | Training/prefill can process many positions in parallel; ordinary autoregressive generation still proceeds one new token at a time |
| **Memory of context** | RNNs compress history into a hidden state (information loss) | Attention can look at **every token directly**, no matter how far apart |
| **Long-range dependencies** | RNNs can carry distant information through hidden states, though compression and training can make it difficult | Direct connections to distant tokens within the available context and permitted attention mask; not guaranteed perfect recall |
| **Position info** | Built in (order is implicit) | Must be **injected** via positional encodings |
| **Parallelization / training speed** | Sequential = slow | Parallel = fast, scales to huge models |
| **Scale** | Struggles beyond certain sizes | Scales extremely well (that's why LLMs work) |

## The intuition in one sentence

An RNN passes information along like a bucket brigade — each person can only hand things to the person next to them. A transformer is more like a room where **everyone can talk directly to everyone else at the same time**, and then each person summarizes what they heard.

The brigade can still transmit information from far away: RNNs are not restricted to nearby-word context. For a Transformer, the room analogy describes full bidirectional attention; causal or local masks restrict who can communicate. Attention also existed in recurrent models before Transformers.

## The "aha" moment (Phase 4/5 of the learning plan)

The defining idea of the transformer is **self-attention**:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Each token generates a *query*, *key*, and *value*:

- **Query (Q)** — "asks" how relevant other tokens are
- **Key (K)** — what each token "advertises" about itself
- **Value (V)** — the actual content that gets mixed together

The query is matched against all keys (via the dot product), the scores are normalized with softmax, and the values of the relevant tokens get mixed together. This single mechanism **replaces recurrence entirely**.

## Where this fits in the learning plan

| Phase | Topic | Connection to this doc |
|---|---|---|
| [Phase1_Essential_Math.md](Phase1_Essential_Math.md) | Math foundations | The $QK^T$ dot products and softmax in attention |
| [Phase2_Neural_Network_Fundamentals.md](Phase2_Neural_Network_Fundamentals.md) | Neurons, layers, activations | The baseline forward/backward calculations to compare against |
| [Phase3_Embeddings.md](Phase3_Embeddings.md) | Tokens → vectors | Input embeddings become the hidden states used by attention |
| [Phase4_Attention.md](Phase4_Attention.md) | The attention mechanism | Projections, row-wise weights, and causal masking |
| [Phase5_Transformer_Architecture.md](Phase5_Transformer_Architecture.md) | Full encoder/decoder stack | How attention + feedforward layers assemble into the transformer |
| [Phase6_LLM_Concepts.md](Phase6_LLM_Concepts.md) | LLMs built on transformers | Training, held-out evaluation, and sequential generation |

## Quick summary

- **A transformer is a neural network** — specifically, a sequence model built on self-attention instead of recurrence or convolution.
- **RNNs** process sequences through hidden states; distant information is transmitted indirectly, not excluded by a nearby-word-only rule.
- **Transformers** can process training/prefill positions in parallel and directly mix permitted context. Causal generation remains sequential across new tokens, and masks/context limits restrict available connections.
