"""CPU-only character language model for Phases 5-6; deliberately no KV cache."""

import argparse
import math

import torch
from torch import nn
from torch.nn import functional as F


TRAIN_TEXT = (
    "the cat sat on the mat.\n"
    "the dog sat on the rug.\n"
    "the cat slept on the rug.\n"
    "the dog slept on the mat.\n"
    "the cat ran to the dog.\n"
    "the dog ran to the cat.\n"
    "the cat saw the dog.\n"
    "the dog saw the cat.\n"
)
VALID_TEXT = (
    "the cat sat on the rug.\n"
    "the dog sat on the mat.\n"
    "the cat slept on the mat.\n"
    "the dog slept on the rug.\n"
)


class TinyTransformer(nn.Module):
    def __init__(
        self, vocab: int = 1000, embed: int = 32, heads: int = 4,
        layers: int = 2, ctx: int = 32,
    ) -> None:
        super().__init__()
        if min(vocab, embed, heads, layers, ctx) <= 0 or embed % heads != 0:
            raise ValueError("dimensions must be positive and embed divisible by heads")
        self.ctx = ctx
        self.embed = nn.Embedding(vocab, embed)
        self.pos = nn.Embedding(ctx, embed)
        block = nn.TransformerEncoderLayer(
            d_model=embed, nhead=heads, dim_feedforward=embed * 4,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            block, num_layers=layers, enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(embed)
        self.head = nn.Linear(embed, vocab)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.size(0) == 0:
            raise ValueError("tokens must have shape (nonempty batch, sequence)")
        seq = tokens.size(1)
        if not 0 < seq <= self.ctx:
            raise ValueError(f"sequence length must be between 1 and {self.ctx}")
        positions = torch.arange(seq, device=tokens.device)
        x = self.embed(tokens) + self.pos(positions)
        # True blocks future positions, matching PyTorch's attention-mask convention.
        mask = torch.triu(
            torch.ones(seq, seq, dtype=torch.bool, device=tokens.device), diagonal=1,
        )
        x = self.blocks(x, mask=mask)
        return self.head(self.norm(x))


def encode(text: str, vocabulary: dict[str, int]) -> torch.Tensor:
    unknown = set(text).difference(vocabulary)
    if unknown:
        raise ValueError(f"text contains characters outside the vocabulary: {sorted(unknown)!r}")
    return torch.tensor([vocabulary[char] for char in text], dtype=torch.long)


def make_examples(tokens: torch.Tensor, context: int) -> tuple[torch.Tensor, torch.Tensor]:
    if tokens.ndim != 1 or context <= 0 or tokens.numel() <= context:
        raise ValueError("need a one-dimensional token sequence longer than the context")
    windows = tokens.unfold(0, context + 1, 1)
    return windows[:, :-1].contiguous(), windows[:, 1:].contiguous()


def next_token_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3 or targets.shape != logits.shape[:2]:
        raise ValueError("expected logits (batch, sequence, vocab) and targets (batch, sequence)")
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


@torch.no_grad()
def evaluate(
    model: TinyTransformer, inputs: torch.Tensor, targets: torch.Tensor,
    batch_size: int = 64,
) -> float:
    if batch_size <= 0 or inputs.size(0) == 0 or inputs.shape != targets.shape:
        raise ValueError("evaluation needs matching, nonempty inputs/targets and a positive batch size")
    was_training = model.training
    model.eval()
    try:
        total_loss = 0.0
        for start in range(0, inputs.size(0), batch_size):
            x = inputs[start:start + batch_size]
            y = targets[start:start + batch_size]
            loss = next_token_loss(model(x), y)
            if not torch.isfinite(loss):
                raise RuntimeError("evaluation produced a non-finite loss")
            total_loss += loss.item() * y.numel()
        return total_loss / targets.numel()
    finally:
        model.train(was_training)


def train(
    model: TinyTransformer, inputs: torch.Tensor, targets: torch.Tensor,
    steps: int = 200, batch_size: int = 16, learning_rate: float = 0.003,
) -> None:
    if steps <= 0 or batch_size <= 0 or inputs.size(0) == 0 or inputs.shape != targets.shape:
        raise ValueError("training needs matching, nonempty data and positive steps/batch size")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning rate must be finite and positive")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(42)
    model.train()
    for step in range(steps):
        indices = torch.randint(inputs.size(0), (batch_size,), generator=generator)
        logits = model(inputs[indices])
        loss = next_token_loss(logits, targets[indices])
        if not torch.isfinite(loss):
            raise RuntimeError(f"training produced a non-finite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def _validate_sampling_settings(
    temperature: float, top_k: int, top_p: float, vocab: int,
) -> None:
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative; zero selects greedy decoding")
    if not 0 <= top_k <= vocab:
        raise ValueError("top_k must be between zero and the vocabulary size")
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p must be finite and in (0, 1]")


def sample_token(
    logits: torch.Tensor, temperature: float = 0.8, top_k: int = 0,
    top_p: float = 1.0, generator: torch.Generator | None = None,
) -> int:
    if logits.ndim != 1 or logits.numel() == 0 or not torch.isfinite(logits).all():
        raise ValueError("sampling needs a finite, nonempty vector of logits")
    _validate_sampling_settings(temperature, top_k, top_p, logits.numel())
    if temperature == 0:
        return int(logits.argmax().item())

    scores = (logits - logits.max()) / temperature
    if torch.isnan(scores).any():
        raise ValueError("temperature is too small for the logits' numeric precision")
    if top_k:
        values, indices = scores.topk(top_k)
        scores = torch.full_like(scores, -torch.inf)
        scores[indices] = values
    sorted_scores, sorted_indices = scores.sort(descending=True)
    probabilities = sorted_scores.softmax(dim=-1)
    # Keep the token that crosses p, not just the tokens strictly below p.
    remove = probabilities.cumsum(dim=-1) - probabilities >= top_p
    sorted_scores = sorted_scores.masked_fill(remove, -torch.inf)
    choice = torch.multinomial(sorted_scores.softmax(dim=-1), 1, generator=generator)
    return int(sorted_indices[choice].item())


@torch.no_grad()
def generate(
    model: TinyTransformer, tokens: torch.Tensor, max_new_tokens: int = 24,
    temperature: float = 0.8, top_k: int = 0, top_p: float = 1.0,
    stop_token_id: int | None = None,
) -> torch.Tensor:
    if tokens.ndim != 2 or tokens.size(0) != 1 or tokens.size(1) == 0:
        raise ValueError("generation needs a nonempty prompt with shape (1, sequence)")
    if tokens.dtype not in (torch.int32, torch.int64):
        raise ValueError("prompt token IDs must be integers")
    if tokens.min() < 0 or tokens.max() >= model.embed.num_embeddings:
        raise ValueError("prompt token IDs must be in the vocabulary")
    if max_new_tokens < 0 or tokens.size(1) + max_new_tokens > model.ctx:
        raise ValueError("prompt plus generation budget must fit the model's context")
    _validate_sampling_settings(temperature, top_k, top_p, model.embed.num_embeddings)
    if stop_token_id is not None and not 0 <= stop_token_id < model.embed.num_embeddings:
        raise ValueError("stop token must be in the vocabulary")
    was_training = model.training
    model.eval()
    generator = torch.Generator(device=tokens.device).manual_seed(42)
    try:
        for _ in range(max_new_tokens):
            # This baseline recomputes the prefix; Phase 6 contrasts it with caching.
            logits = model(tokens)[0, -1]
            next_id = sample_token(logits, temperature, top_k, top_p, generator)
            tokens = torch.cat((tokens, tokens.new_tensor([[next_id]])), dim=1)
            if next_id == stop_token_id:
                break
        return tokens
    finally:
        model.train(was_training)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--prompt", default="the cat")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    args = parser.parse_args()

    characters = sorted(set(TRAIN_TEXT))
    vocabulary = {char: index for index, char in enumerate(characters)}
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if not args.prompt or set(args.prompt).difference(vocabulary):
        parser.error("--prompt must be nonempty and use only characters in the training text")
    if args.max_new_tokens < 0 or len(args.prompt) + args.max_new_tokens > 32:
        parser.error("prompt plus --max-new-tokens must fit the 32-token context")
    try:
        _validate_sampling_settings(args.temperature, args.top_k, args.top_p, len(vocabulary))
    except ValueError as error:
        parser.error(str(error))

    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = TinyTransformer(vocab=len(vocabulary))
    train_inputs, train_targets = make_examples(encode(TRAIN_TEXT, vocabulary), model.ctx)
    valid_inputs, valid_targets = make_examples(encode(VALID_TEXT, vocabulary), model.ctx)
    initial_train = evaluate(model, train_inputs, train_targets)
    initial_valid = evaluate(model, valid_inputs, valid_targets)
    train(model, train_inputs, train_targets, steps=args.steps)
    final_train = evaluate(model, train_inputs, train_targets)
    final_valid = evaluate(model, valid_inputs, valid_targets)
    print(f"CPU; character tokens; context={model.ctx}; vocabulary={len(vocabulary)}")
    print(f"train loss: {initial_train:.3f} -> {final_train:.3f}")
    print(f"validation loss: {initial_valid:.3f} -> {final_valid:.3f}")
    print(f"validation perplexity: {math.exp(initial_valid):.3f} -> {math.exp(final_valid):.3f}")
    prompt = encode(args.prompt, vocabulary).unsqueeze(0)
    output = generate(
        model, prompt, args.max_new_tokens, args.temperature, args.top_k, args.top_p,
        stop_token_id=vocabulary["\n"],
    )
    print("generated:", repr("".join(characters[index] for index in output[0].tolist())))


if __name__ == "__main__":
    main()
