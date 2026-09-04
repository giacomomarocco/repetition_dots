from __future__ import annotations

import math
import json
import tempfile
import unittest
from pathlib import Path

from filler.addition import two_fact as experiment


def fake_encoder(messages, thinking_mode):
    assert thinking_mode == "chat"
    return "<bos>" + "".join(f"<{m['role']}>{m['content']}" for m in messages)


class TwoFactAdditionTests(unittest.TestCase):
    def setUp(self):
        self.facts = [
            {"fact_id": "facts.json:1", "question": "How many moons does Mars have?", "answer": 2},
            {"fact_id": "facts.json:2", "question": "How many continents are there?", "answer": 7},
            {"fact_id": "facts.json:3", "question": "How many legs does a spider have?", "answer": 8},
            {"fact_id": "facts.json:4", "question": "How many sides does a hexagon have?", "answer": 6},
        ]

    def test_pairing_is_deterministic_cyclic_and_seeded(self):
        first = experiment.pair_facts(self.facts, 42)
        self.assertEqual(first, experiment.pair_facts(list(reversed(self.facts)), 42))
        self.assertEqual(len(first), len(self.facts))
        self.assertTrue(all(left["fact_id"] != right["fact_id"] for left, right in first))
        occurrences = {
            fact["fact_id"]: sum(fact["fact_id"] in (left["fact_id"], right["fact_id"])
                                 for left, right in first)
            for fact in self.facts
        }
        self.assertEqual(set(occurrences.values()), {2})
        self.assertNotEqual(first, experiment.pair_facts(self.facts, 43))

    def test_pairing_requires_at_least_two_facts(self):
        self.assertEqual(experiment.pair_facts(self.facts[:1], 42), [])

    def test_target_is_sum_of_both_factual_answers(self):
        pair = [(self.facts[0], self.facts[1])]
        tasks = experiment.make_tasks(pair, [0, 10])
        self.assertEqual({row["target"] for row in tasks}, {9})
        self.assertEqual({row["pair_id"] for row in tasks}, {tasks[0]["pair_id"]})

    def test_question_contains_both_questions_but_not_answers(self):
        task = experiment.make_tasks([(self.facts[0], self.facts[1])], [0])[0]
        question = experiment.render_question(task)
        self.assertIn(self.facts[0]["question"], question)
        self.assertIn(self.facts[1]["question"], question)
        self.assertNotIn("2 + 7", question)

    def test_filler_and_answer_slot_are_in_final_user_turn(self):
        tasks = experiment.make_tasks([(self.facts[0], self.facts[1])], [0, 10])
        self.assertTrue(experiment.render_prompt(fake_encoder, tasks[0]).endswith("Answer:"))
        self.assertTrue(experiment.render_prompt(fake_encoder, tasks[1]).endswith(
            ". . . . . . . . . .\nAnswer:"
        ))

    def test_prompt_contains_exactly_five_two_fact_demonstrations(self):
        captured = []
        def encoder(messages, thinking_mode):
            captured.extend(messages)
            return fake_encoder(messages, thinking_mode)
        task = experiment.make_tasks([(self.facts[0], self.facts[1])], [5])[0]
        experiment.render_prompt(encoder, task)
        self.assertEqual(len([m for m in captured if m["role"] == "assistant"]), 5)
        self.assertEqual(captured[-1]["role"], "user")
        user_turns = [m["content"] for m in captured if m["role"] == "user"]
        self.assertEqual(len(user_turns), 6)
        self.assertTrue(all(content.endswith(". . . . .\nAnswer:") for content in user_turns))
        self.assertTrue(all("Filler:" not in content for content in user_turns))

    def test_scoring_options_match_one_fact_defaults(self):
        args = experiment.parse_args([])
        self.assertEqual(args.top_logprobs, 20)
        with self.assertRaises(SystemExit):
            experiment.parse_args(["--top-logprobs", "0"])

    def test_shared_target_score_schema(self):
        metadata = {
            "output_token_ids_logprobs": [[[-1.5, 9]]],
            "output_top_logprobs": [[[-0.1, 7], [-1.5, 9]]],
        }
        score = experiment.shared.extract_target_score(metadata, 9, 2)
        self.assertEqual(score["target_top_rank"], 2)
        self.assertAlmostEqual(score["target_probability"], math.exp(-1.5))
        self.assertEqual(score["target_log_probability"], -1.5)

    def test_progress_resume_validates_and_loads_completed_rows(self):
        prompt = experiment.make_tasks([(self.facts[0], self.facts[1])], [0])[0]
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
