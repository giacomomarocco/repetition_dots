from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import torch

from filler.fact_eval import hf as experiment
from filler.fact_eval import model_adapter
from filler.fact_eval import protocol as legacy


class FakeTokenizer:
    chat_template = "official"

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return "<official>" + "|".join(item["content"] for item in messages)

    def __call__(self, text, **kwargs):
        self.encoded_text = text
        return {"input_ids": torch.tensor([[1, 2, 3]])}


def selected_fact() -> dict:
    return {
        "fact_id": "age:0",
        "question": "At what age did Example Person die?",
        "answer": 42,
        "category": "Age at death",
        "kind": "age",
        "source_file": "age_facts.json",
        "source_index": 0,
        "paraphrases": [f"question {index}" for index in range(5)],
    }


def trial(index: int, response: str, parsed, correct: bool) -> dict:
    return {
        "fact_id": "age:0",
        "question": selected_fact()["question"],
        "prompt_question": f"question {index - 1}",
        "expected": 42,
        "trial": index,
        "response": response,
        "parsed_answer": parsed,
        "correct": correct,
    }


class StrictParsingTests(unittest.TestCase):
    def test_only_exact_answer_integer_is_accepted(self):
        accepted = {"Answer: 42", " Answer: -7\n", "Answer:    9"}
        rejected = {
            "42",
            "Answer: 42 because...",
            "The answer is 42",
            "Answer: 4.2",
            "Answer: +42",
            "Answer: 42\nextra",
        }
        self.assertEqual([legacy.parse_answer(item) is not None for item in accepted], [True] * 3)
        self.assertEqual([legacy.parse_answer(item) for item in rejected], [None] * 6)


class ClassificationTests(unittest.TestCase):
    def test_four_of_five_is_known_and_raw_fields_are_retained(self):
        fact = selected_fact()
        rows = [trial(index, "Answer: 42", 42, index < 5) for index in range(1, 6)]
        rows[-1] = trial(5, "verbose failure", None, False)
        result = experiment.completed_results([fact], {"age:0": rows})[0]
        self.assertEqual(result["correct_count"], 4)
        self.assertEqual(result["trial_correctness"], [True, True, True, True, False])
        self.assertEqual(result["trial_responses"][-1], "verbose failure")
        self.assertEqual(result["trial_predictions"][-1], None)
        self.assertGreaterEqual(result["correct_count"], legacy.PASS_COUNT)

    def test_incomplete_fact_is_not_classified(self):
        fact = selected_fact()
        rows = [trial(index, "Answer: 42", 42, True) for index in range(1, 5)]
        self.assertEqual(experiment.completed_results([fact], {"age:0": rows}), [])


class ResumeTests(unittest.TestCase):
    def test_compatible_config_resumes(self):
        config = {
            key: value
            for key, value in {
                "model_id": "model",
                "requested_model_revision": "rev",
                "tokenizer_id": "tokenizer",
                "requested_tokenizer_revision": "tokrev",
                "selection": {"seed": 42},
                "prompt": {"history": False},
                "decoding": {"do_sample": False},
                "classification": {"known_minimum_correct": 4},
            }.items()
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run_config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            experiment.ensure_compatible_resume(path, dict(config))

    def test_incompatible_config_is_rejected(self):
        keys = {
            "model_id": "model",
            "requested_model_revision": "rev",
            "tokenizer_id": "tokenizer",
            "requested_tokenizer_revision": "tokrev",
            "selection": {"seed": 42},
            "prompt": {},
            "decoding": {},
            "classification": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run_config.json"
            path.write_text(json.dumps(keys), encoding="utf-8")
            changed = dict(keys)
            changed["selection"] = {"seed": 43}
            with self.assertRaisesRegex(ValueError, "incompatible"):
                experiment.ensure_compatible_resume(path, changed)

    def test_existing_trials_are_loaded_without_repair(self):
        fact = selected_fact()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.jsonl"
            rows = [trial(1, "Answer: 42", 42, True), trial(2, "bad", None, False)]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            loaded = legacy.load_progress(path, [fact])
            self.assertEqual(loaded["age:0"], rows)


class AdapterTests(unittest.TestCase):
    def test_official_chat_template_is_used_with_thinking_disabled(self):
        tokenizer = FakeTokenizer()
        messages = experiment.messages_for("What is the number?")
        rendered, encoded = model_adapter.render_prompt(tokenizer, messages)
        self.assertTrue(rendered.startswith("<official>"))
        self.assertEqual(encoded["input_ids"].shape[-1], 3)
        self.assertEqual(tokenizer.kwargs["enable_thinking"], False)
        self.assertEqual([item["role"] for item in tokenizer.messages], ["system", "user"])

    def test_plain_text_fallback_does_not_invent_chat_tokens(self):
        tokenizer = FakeTokenizer()
        tokenizer.chat_template = None
        rendered, _ = model_adapter.render_prompt(
            tokenizer, experiment.messages_for("What is the number?")
        )
        self.assertNotIn("<|", rendered)
        self.assertIn("Question: What is the number?", rendered)
        self.assertTrue(rendered.endswith("Answer:"))

    @mock.patch("torch.cuda.is_available", return_value=False)
    def test_explicit_cuda_fails_on_login_node(self, _available):
        with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
            model_adapter.resolve_device("cuda")


if __name__ == "__main__":
    unittest.main()
