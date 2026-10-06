# Abbreviations & Symbols

A quick lookup for every abbreviation, acronym, and symbol used across these guides. Each entry points to the page where the idea is explained in depth — if a term shows up mid-phase and you don't remember it, find it here, then jump to the explanation.

## Abbreviations & Acronyms

| Abbreviation | Stands for | In one line | Explained in |
|---|---|---|---|
| AI | Artificial Intelligence | The broad field of getting computers to do tasks that normally require human intelligence. | [Learning plan](AI_Transformer_Learning_Plan.md) |
| BERT | Bidirectional Encoder Representations from Transformers | An "encoder-only" Transformer model family, good at understanding tasks like classification and search. | [Phase 5](Phase5_Transformer_Architecture.md) |
| BPE | Byte-Pair Encoding | A tokenization method that starts from characters and repeatedly merges the most frequent pair into a new token. | [Phase 3](Phase3_Embeddings.md) |
| CNN | Convolutional Neural Network | A neural network family built for images/grids; an alternative to Transformers for certain tasks. | [NN vs Transformer](Neural_Network_vs_Transformer.md) |
| FFN | Feed-Forward Network | The two-layer, per-token network inside every Transformer block (also called an MLP). | [Phase 5](Phase5_Transformer_Architecture.md) |
| GELU | Gaussian Error Linear Unit | A smooth activation function often used in Transformer-based language models. | [Phase 2](Phase2_Neural_Network_Fundamentals.md) |
| GPT | Generative Pre-trained Transformer | The decoder-only model family behind ChatGPT-style generation. | [Phase 5](Phase5_Transformer_Architecture.md) |
| GQA | Grouped-Query Attention | Shares Keys and Values across groups of attention heads to shrink the KV cache. | [Phase 6](Phase6_LLM_Concepts.md) |
| GRU | Gated Recurrent Unit | A recurrent sequence architecture; a simplified cousin of LSTM. | [NN vs Transformer](Neural_Network_vs_Transformer.md) |
| ID | Identifier | A unique integer label — for example, the number assigned to each token in a vocabulary. | [Phase 3](Phase3_Embeddings.md) |
| KV cache | Key/Value cache | The stored Keys and Values of previous tokens, so attention doesn't recompute them at every generation step. | [Phase 6](Phase6_LLM_Concepts.md) |
| LLM | Large Language Model | A very large, usually decoder-only Transformer trained to predict the next token. | [Phase 6](Phase6_LLM_Concepts.md) |
| LoRA | Low-Rank Adaptation | A fine-tuning shortcut that trains two small matrices instead of updating all model weights. | [Phase 6](Phase6_LLM_Concepts.md) |
| LSTM | Long Short-Term Memory | A recurrent network designed to carry information over long sequences; superseded by attention in Transformers. | [NN vs Transformer](Neural_Network_vs_Transformer.md) |
| MLP | Multi-Layer Perceptron | A plain feed-forward network; in a Transformer, the FFN is an MLP. | [NN vs Transformer](Neural_Network_vs_Transformer.md) |
| MQA | Multi-Query Attention | Shares one Key/Value across all attention heads — an even smaller KV cache than GQA. | [Phase 6](Phase6_LLM_Concepts.md) |
| MSE | Mean Squared Error | A loss function that averages the squared differences between predictions and targets. | [Phase 2](Phase2_Neural_Network_Fundamentals.md) |
| NN | Neural Network | The broad model family that Transformers belong to. | [NN vs Transformer](Neural_Network_vs_Transformer.md) |
| PAD | Padding token | A special token (`[PAD]`) that fills empty positions so sequences in a batch have equal length. | [Phase 3](Phase3_Embeddings.md) |
| ReLU | Rectified Linear Unit | An activation function that keeps positive values and replaces negatives with zero. | [Phase 2](Phase2_Neural_Network_Fundamentals.md) |
| RNN | Recurrent Neural Network | A sequence model that passes information step by step; Transformers replaced recurrence with attention. | [NN vs Transformer](Neural_Network_vs_Transformer.md) |
| RoPE | Rotary Position Embedding | A modern way to encode token order by rotating Q/K vectors by position-dependent angles. | [Phase 5](Phase5_Transformer_Architecture.md) |
| SwiGLU | Swish-Gated Linear Unit | A gated activation often used in modern Transformer language models. | [Phase 2](Phase2_Neural_Network_Fundamentals.md) |
| UNK | Unknown token | A special token (`[UNK]`) used as a fallback when text isn't in the vocabulary. | [Phase 3](Phase3_Embeddings.md) |

## Symbols & Notation

| Symbol | Name | In one line | Explained in |
|---|---|---|---|
| `Q`, `K`, `V` | Query, Key, Value | The three roles each token plays in attention: what am I asking, what do I contain, what do I pass on. | [Phase 4](Phase4_Attention.md) |
| `W_Q`, `W_K`, `W_V` | Query/Key/Value weight matrices | The learned matrices that turn each token's embedding into its Q, K, and V vectors (per-head versions are written $W_i^Q$, $W_i^K$, $W_i^V$). | [Phase 4](Phase4_Attention.md) |
| `W^O` | Output weight matrix | The final blending layer that combines all attention heads' outputs back into one vector. | [Phase 5](Phase5_Transformer_Architecture.md) |
| `QKᵀ` | Query–Key score matrix | The dot products between every query and every key — how well each token matches every other token (ᵀ means transpose). | [Phase 4](Phase4_Attention.md) |
| `d_k` | Key dimension | The width of each query/key vector; scores are divided by √d_k to keep softmax stable. | [Phase 4](Phase4_Attention.md) |
| `d_model` | Model dimension | The width of each token's vector inside the model — it stays constant through every block. | [Phase 5](Phase5_Transformer_Architecture.md) |
