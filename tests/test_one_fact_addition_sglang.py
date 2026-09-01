from __future__ import annotations

import math
import unittest

import one_fact_addition_sglang as experiment


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

    def test_filler_is_forced_before_answer_prefix(self):
        tasks = experiment.make_tasks([self.fact], 42, 1, [0, 10])
        baseline = next(row for row in tasks if row["k"] == 0)
        filled = next(row for row in tasks if row["k"] == 10)
        self.assertTrue(experiment.render_prompt(fake_encoder, baseline).endswith("Answer: "))
        self.assertTrue(
            experiment.render_prompt(fake_encoder, filled).endswith(
                ". . . . . . . . . .\nAnswer: "
            )
        )

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


if __name__ == "__main__":
    unittest.main()
