from __future__ import annotations

import math
import json
import tempfile
import unittest
from pathlib import Path

from filler.addition import one_fact as experiment


def fake_encoder(messages, thinking_mode):
    assert thinking_mode == "chat"
    return "<bos>" + "".join(
        f"<{m['role']}>{m['content']}" for m in messages
    )


class OneFactAdditionTests(unittest.TestCase):
    def setUp(self):
        self.fact = {
            "fact_id": "facts.json:1",
            "question": "How many moons does Mars have?",
            "answer": 2,
        }

    def test_full_sum_is_not_reduced_modulo_ten(self):
        tasks = experiment.make_tasks([self.fact], 42, 1, [0, 10])
        self.assertEqual(len(tasks), 2)
        for task in tasks:
            self.assertEqual(task["target"], 2 + task["addend"])
            self.assertGreaterEqual(task["target"], 12)

    def test_pair_is_shared_across_conditions(self):
        tasks = experiment.make_tasks([self.fact], 42, 1, [0, 10, 20, 50, 100])
        self.assertEqual({row["pair_id"] for row in tasks}, {tasks[0]["pair_id"]})
        self.assertEqual({row["addend"] for row in tasks}, {tasks[0]["addend"]})
        self.assertEqual({row["target"] for row in tasks}, {tasks[0]["target"]})
        self.assertEqual(
            {row["condition"] for row in tasks},
            {"baseline", "dots_10", "dots_20", "dots_50", "dots_100"},
        )

    def test_filler_and_answer_slot_are_in_final_user_turn(self):
        tasks = experiment.make_tasks([self.fact], 42, 1, [0, 10])
        baseline = next(row for row in tasks if row["k"] == 0)
        filled = next(row for row in tasks if row["k"] == 10)
        self.assertTrue(experiment.render_prompt(fake_encoder, baseline).endswith("<user>" + experiment.render_question(baseline) + "\nAnswer:"))
        self.assertTrue(
            experiment.render_prompt(fake_encoder, filled).endswith(
                ". . . . . . . . . .\nAnswer:"
            )
        )

    def test_prompt_contains_exactly_five_demonstrations(self):
        captured = []
        def encoder(messages, thinking_mode):
            captured.extend(messages)
            return fake_encoder(messages, thinking_mode)
        task = experiment.make_tasks([self.fact], 42, 1, [5])[0]
        experiment.render_prompt(encoder, task)
        self.assertEqual(len([m for m in captured if m["role"] == "assistant"]), 5)
        self.assertEqual(captured[-1]["role"], "user")
        user_turns = [m["content"] for m in captured if m["role"] == "user"]
        self.assertEqual(len(user_turns), 6)
        self.assertTrue(all(content.endswith(". . . . .\nAnswer:") for content in user_turns))
        self.assertTrue(all("Filler:" not in content for content in user_turns))
        self.assertIn("No explanation, no words, no reasoning, just the number.", captured[0]["content"])

    def test_local_headsup_names_exact_nonzero_filler_count(self):
        captured = []
        def encoder(messages, thinking_mode):
            captured.extend(messages)
            return fake_encoder(messages, thinking_mode)
        filled = experiment.make_tasks([self.fact], 42, 1, [10])[0]
        experiment.render_prompt(encoder, filled, "local-headsup")
        self.assertIn("there will be 10 dots", captured[0]["content"])
        self.assertTrue(captured[-1]["content"].endswith(". . . . . . . . . .\nAnswer:"))

        captured.clear()
        baseline = experiment.make_tasks([self.fact], 42, 1, [0])[0]
        experiment.render_prompt(encoder, baseline, "local-headsup")
        self.assertEqual(captured[0]["content"], experiment.SYSTEM_PROMPT)

    def test_question_contains_fact_but_not_answer(self):
        task = experiment.make_tasks([self.fact], 42, 1, [0])[0]
        question = experiment.render_question(task)
        self.assertIn(self.fact["question"], question)
        self.assertNotIn(f"{self.fact['answer']} +", question)

    def test_strict_integer_completion_parser(self):
        self.assertEqual(experiment.parse_answer("42"), 42)
        self.assertEqual(experiment.parse_answer(" -3\n"), -3)
        self.assertIsNone(experiment.parse_answer("Answer: 42"))
        self.assertIsNone(experiment.parse_answer("42."))

    def test_one_token_target_validation(self):
        original = experiment.tokenize
        experiment.tokenize = lambda endpoint, text, timeout: (
            [1, 2, 68] if text.endswith("68") else [1, 2]
        )
        try:
            self.assertEqual(
                experiment.validate_one_token_target("url", "prompt", 68, 1), 68
            )
        finally:
            experiment.tokenize = original

    def test_target_score_and_rank(self):
        metadata = {
            "output_token_ids_logprobs": [[[-1.5, 68]]],
            "output_top_logprobs": [[[-0.1, 7], [-1.5, 68]]],
        }
        score = experiment.extract_target_score(metadata, 68, 2)
        self.assertEqual(score["target_top_rank"], 2)
        self.assertAlmostEqual(score["target_probability"], math.exp(-1.5))
        self.assertIsNone(score["target_rank_lower_bound"])

    def test_rank_lower_bound(self):
        metadata = {
            "output_token_ids_logprobs": [[[-9.0, 68]]],
            "output_top_logprobs": [[[-0.1, 7], [-0.2, 8]]],
        }
        score = experiment.extract_target_score(metadata, 68, 2)
        self.assertIsNone(score["target_top_rank"])
        self.assertEqual(score["target_rank_lower_bound"], 3)

    def test_progress_resume_validates_and_loads_completed_rows(self):
        prompt = experiment.make_tasks([self.fact], 42, 1, [0])[0]
        config = {key: key for key in experiment.RESUME_CONFIG_KEYS}
        result = {**prompt, "target_log_probability": -1.0}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results_progress.jsonl"
            path.write_text(json.dumps(result) + "\n")
            loaded = experiment.load_resume_results(path, [prompt], dict(config), config)
            self.assertEqual(loaded, [result])
            changed = dict(config)
            changed["seed"] = "different"
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                experiment.load_resume_results(path, [prompt], config, changed)


if __name__ == "__main__":
    unittest.main()
