"""Execute the documented examples and test the CPU capstone's key behaviors."""

import contextlib
import io
import re
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

from examples.tiny_transformer import (
    TRAIN_TEXT,
    VALID_TEXT,
    TinyTransformer,
    encode,
    evaluate,
    generate,
    make_examples,
    next_token_loss,
    sample_token,
    train,
)


ROOT = Path(__file__).resolve().parents[1]
PYTHON_BLOCK = re.compile(r"^```python[ \t]*\n(.*?)^```[ \t]*$", re.MULTILINE | re.DOTALL)


def run_guides(*filenames):
    namespace = {}
    for filename in filenames:
        path = ROOT / filename
        text = path.read_text(encoding="utf-8")
        for match in PYTHON_BLOCK.finditer(text):
            code = "\n" * text[:match.start(1)].count("\n") + match.group(1)
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(code, str(path), "exec"), namespace)
    return namespace


class GuideExamplesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_phase1_calculations_and_shapes(self):
        values = run_guides("Phase1_Essential_Math.md")
        self.assertEqual(values["a"] @ values["b"], 32)
        np.testing.assert_array_equal(values["M"] @ values["x"], [11, 25, 39])
        self.assertEqual(values["M"].T.shape, (2, 3))
        self.assertAlmostEqual(values["normalized"].mean(), 0)
        self.assertAlmostEqual(values["normalized"].std(), 1)
        self.assertAlmostEqual(values["derivative"], 6, places=7)

    def test_phase2_trains_and_matches_a_numerical_gradient(self):
        values = run_guides("Phase2_Neural_Network_Fundamentals.md")
        self.assertLess(values["final_loss"], values["initial_loss"])
        np.testing.assert_array_equal(values["predictions"], [0, 0, 1, 1])
        x = np.array([0.3, 0.7])
        loss_and_gradients = values["loss_and_gradients"]
        _, gradients = loss_and_gradients(x, 0)
        epsilon = 1e-5
        for parameter, gradient in zip(values["parameters"], gradients):
            numerical = np.empty_like(parameter)
            for index in np.ndindex(parameter.shape):
                original = parameter[index]
                try:
                    parameter[index] = original + epsilon
                    upper = loss_and_gradients(x, 0)[0]
                    parameter[index] = original - epsilon
                    lower = loss_and_gradients(x, 0)[0]
                finally:
                    parameter[index] = original
                numerical[index] = (upper - lower) / (2 * epsilon)
            np.testing.assert_allclose(gradient, numerical, rtol=1e-4, atol=1e-6)

    def test_phase3_cosine_and_positive_scaling(self):
        values = run_guides("Phase3_Embeddings.md")
        vectors, cosine = values["embeddings"], values["cosine_similarity"]
        score = cosine(vectors["king"], vectors["queen"])
        self.assertAlmostEqual(score, values["scaled_similarity"])
        self.assertAlmostEqual(cosine(vectors["king"], vectors["car"]), 0.99 / 1.63)
        self.assertAlmostEqual(
            (10 * vectors["king"]) @ vectors["queen"],
            10 * (vectors["king"] @ vectors["queen"]),
        )

    def test_phase4_mask_and_future_value_isolation(self):
        values = run_guides("Phase4_Attention.md")
        attention = values["attention"]
        Q, K, V, mask = (values[name] for name in ("Q", "K", "V", "mask"))
        weights, original = attention(Q, K, V, mask)
        np.testing.assert_allclose(weights.sum(axis=-1), 1)
        self.assertTrue(np.all(weights[mask] == 0))
        changed = V.copy()
        changed[-1] += 100
        _, causal_changed = attention(Q, K, changed, mask)
        np.testing.assert_allclose(original[:-1], causal_changed[:-1], rtol=0, atol=1e-12)
        _, unmasked = attention(Q, K, V)
        _, unmasked_changed = attention(Q, K, changed)
        self.assertGreater(np.linalg.norm(unmasked[0] - unmasked_changed[0]), 1)
        with self.assertRaises(ValueError):
            attention(Q, K, V, np.ones_like(mask))
        with self.assertRaises(ValueError):
            attention(Q, K, V, np.zeros((1, 1), dtype=bool))

    def test_phase5_batched_heads_and_causal_isolation(self):
        values = run_guides("Phase4_Attention.md", "Phase5_Transformer_Architecture.md")
        self.assertEqual(values["multi_head_output"].shape, (2, 3, 4))
        self.assertEqual(tuple(values["logits"].shape), (1, 8, 1000))
        attention = values["multi_head_attention"]
        X = values["mh_input"]
        projections = values["mh_projections"]
        mask = values["mh_mask"]
        original = attention(X, *projections, heads=2, mask=mask)
        changed = X.copy()
        changed[:, -1] += 100
        causal_changed = attention(changed, *projections, heads=2, mask=mask)
        np.testing.assert_allclose(original[:, :-1], causal_changed[:, :-1], rtol=0, atol=1e-12)
        unmasked = attention(X, *projections, heads=2)
        unmasked_changed = attention(changed, *projections, heads=2)
        self.assertGreater(np.linalg.norm(unmasked[:, 0] - unmasked_changed[:, 0]), 1e-3)
        with self.assertRaises(ValueError):
            attention(X, *projections, heads=3)

    def test_phase6_cached_and_full_prefix_attention_match(self):
        values = run_guides("Phase4_Attention.md", "Phase6_LLM_Concepts.md")
        np.testing.assert_allclose(
            values["cached_outputs"], values["full_outputs"], rtol=0, atol=1e-12,
        )
        self.assertEqual(values["cached_keys"].shape, (3, 2))


class CapstoneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(42)
        self.model = TinyTransformer(vocab=19)

    def test_forward_shape_and_prenorm(self):
        self.assertEqual(tuple(self.model(torch.zeros((2, 8), dtype=torch.long)).shape), (2, 8, 19))
        self.assertTrue(all(layer.norm_first for layer in self.model.blocks.layers))
        self.assertEqual(tuple(self.model.norm.weight.shape), (32,))
        self.assertFalse(torch.nn.TransformerEncoderLayer(32, 4).norm_first)

    def test_future_tokens_cannot_change_prefix_logits(self):
        self.model.eval()
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
        changed = tokens.clone()
        changed[:, 3:] = torch.tensor([[10, 11, 12]])
        with torch.no_grad():
            original = self.model(tokens)
            causal_changed = self.model(changed)
            positions = torch.arange(tokens.size(1))
            unmasked = self.model.head(self.model.norm(self.model.blocks(
                self.model.embed(tokens) + self.model.pos(positions))))
            unmasked_changed = self.model.head(self.model.norm(self.model.blocks(
                self.model.embed(changed) + self.model.pos(positions))))
        torch.testing.assert_close(original[:, :3], causal_changed[:, :3], rtol=0, atol=1e-6)
        self.assertGreater((original[:, 3:] - causal_changed[:, 3:]).abs().max().item(), 1e-3)
        self.assertGreater((unmasked[:, :3] - unmasked_changed[:, :3]).abs().max().item(), 1e-3)

    def test_shifted_targets_and_loss_on_logits(self):
        inputs, targets = make_examples(torch.arange(8), 3)
        torch.testing.assert_close(inputs[0], torch.tensor([0, 1, 2]))
        torch.testing.assert_close(targets[0], torch.tensor([1, 2, 3]))
        self.assertEqual(tuple(inputs.shape), (5, 3))
        logits = torch.tensor([[[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]]])
        labels = torch.tensor([[0, 1]])
        expected = -logits.log_softmax(dim=-1).gather(-1, labels.unsqueeze(-1)).mean()
        torch.testing.assert_close(next_token_loss(logits, labels), expected)
        with self.assertRaises(ValueError):
            next_token_loss(logits, labels[:, :1])
        with self.assertRaises(ValueError):
            make_examples(torch.arange(3), 3)

    def test_default_training_reduces_heldout_loss_by_twenty_percent(self):
        vocabulary = {char: index for index, char in enumerate(sorted(set(TRAIN_TEXT)))}
        torch.manual_seed(42)
        model = TinyTransformer(vocab=len(vocabulary))
        x_train, y_train = make_examples(encode(TRAIN_TEXT, vocabulary), model.ctx)
        x_valid, y_valid = make_examples(encode(VALID_TEXT, vocabulary), model.ctx)
        initial_train = evaluate(model, x_train, y_train)
        initial_valid = evaluate(model, x_valid, y_valid)
        original_embeddings = model.embed.weight.detach().clone()
        train(model, x_train, y_train)
        self.assertLess(evaluate(model, x_train, y_train), initial_train)
        self.assertLess(evaluate(model, x_valid, y_valid), initial_valid * 0.8)
        self.assertGreater((model.embed.weight - original_embeddings).abs().max().item(), 1e-3)

    def test_generation_stop_token_budget_and_mode(self):
        with torch.no_grad():
            self.model.head.weight.zero_()
            self.model.head.bias.fill_(-10)
            self.model.head.bias[3] = 10
        prompt = torch.tensor([[1, 2]])
        output = generate(self.model, prompt, max_new_tokens=8, temperature=0, stop_token_id=3)
        torch.testing.assert_close(output, torch.tensor([[1, 2, 3]]))
        self.assertTrue(self.model.training)
        torch.testing.assert_close(generate(self.model, prompt, max_new_tokens=0), prompt)
        output = generate(self.model, prompt, max_new_tokens=4, temperature=0)
        self.assertEqual(tuple(output.shape), (1, 6))
        with self.assertRaises(ValueError):
            generate(self.model, prompt, max_new_tokens=31)
        with self.assertRaises(ValueError):
            generate(self.model, prompt, max_new_tokens=0, temperature=-1)
        with self.assertRaises(ValueError):
            generate(self.model, torch.tensor([[-1]]), max_new_tokens=0)
        with self.assertRaises(ValueError):
            self.model(torch.empty((1, 0), dtype=torch.long))
        with self.assertRaises(ValueError):
            self.model(torch.zeros((1, 33), dtype=torch.long))

    def test_sampling_greedy_top_k_and_top_p_crossing_token(self):
        logits = torch.tensor([0.6, 0.25, 0.1, 0.05]).log()
        generator = torch.Generator().manual_seed(42)
        self.assertEqual(sample_token(logits, temperature=0), 0)
        self.assertEqual(
            {sample_token(logits, top_k=1, generator=generator) for _ in range(20)}, {0},
        )
        self.assertEqual(
            {sample_token(logits, temperature=1, top_p=0.7, generator=generator)
             for _ in range(50)},
            {0, 1},
        )
        self.assertEqual(
            {sample_token(logits, temperature=1, top_p=0.5, generator=generator)
             for _ in range(20)},
            {0},
        )
        self.assertEqual(sample_token(torch.tensor([1000.0, 999.0]), temperature=1e-8), 0)
        for settings in ({"temperature": -1}, {"temperature": float("nan")},
                         {"top_k": 5}, {"top_p": 0}, {"top_p": 1.1}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                sample_token(logits, **settings)
        with self.assertRaises(ValueError):
            sample_token(torch.tensor([float("nan")]))

    def test_nonfinite_loss_is_explicit_and_evaluation_restores_mode(self):
        inputs = torch.zeros((2, 4), dtype=torch.long)
        self.assertTrue(self.model.training)
        evaluate(self.model, inputs, inputs)
        self.assertTrue(self.model.training)
        with torch.no_grad():
            self.model.head.weight.fill_(torch.nan)
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            evaluate(self.model, inputs, inputs)
        self.assertTrue(self.model.training)
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            train(self.model, inputs, inputs, steps=1)

    def test_cli_rejects_unknown_prompt_without_training(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "examples" / "tiny_transformer.py"), "--prompt", "xyz"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--prompt", result.stderr)
        self.assertNotIn("train loss:", result.stdout)


class DocumentationTests(unittest.TestCase):
    def test_python_blocks_compile_and_fences_balance(self):
        for path in ROOT.glob("*.md"):
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                self.assertEqual(len(re.findall(r"^```.*$", text, re.MULTILINE)) % 2, 0)
                for match in PYTHON_BLOCK.finditer(text):
                    compile(match.group(1), str(path), "exec")

    def test_every_phase_has_three_completion_checks(self):
        for path in ROOT.glob("Phase*.md"):
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                checks = text.split("## Before Moving On", 1)
                self.assertEqual(len(checks), 2)
                for label in ("By hand", "In code", "In your own words"):
                    self.assertIn(label, checks[1])

    def test_local_markdown_links_and_anchors_exist(self):
        for path in ROOT.glob("*.md"):
            text = path.read_text(encoding="utf-8")
            for target in re.findall(r"\[[^\]\n]+\]\(([^)\s]+)\)", text):
                if re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:", target):
                    continue
                relative, _, anchor = target.partition("#")
                destination = path.parent / relative if relative else path
                with self.subTest(path=path.name, target=target):
                    self.assertTrue(destination.is_file(), f"missing link target: {destination}")
                    if anchor and destination.suffix == ".md":
                        headings = re.findall(
                            r"^#{1,6}\s+(.+)$", destination.read_text(encoding="utf-8"),
                            re.MULTILINE,
                        )
                        slugs = {
                            re.sub(r"\s", "-", re.sub(r"[^\w -]", "", heading.lower()))
                            for heading in headings
                        }
                        self.assertIn(anchor, slugs)


if __name__ == "__main__":
    unittest.main()
