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

    def test_pairing_can_generate_more_pairs_than_facts_without_duplicates(self):
        pairs = experiment.pair_facts(self.facts, 42, 10)
        ids = [(left["fact_id"], right["fact_id"]) for left, right in pairs]
        self.assertEqual(len(ids), 10)
        self.assertEqual(len(set(ids)), 10)
        self.assertTrue(all(left != right for left, right in ids))
        with self.assertRaisesRegex(ValueError, "maximum is 12"):
            experiment.pair_facts(self.facts, 42, 13)

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

    def test_upstream_prompt_matches_labeled_filler_scaffold(self):
        atomic_facts = [
            {"fact_id": "a", "question": "What is the atomic number of Thorium?", "answer": 90},
            {"fact_id": "b", "question": "What is the atomic number of Tellurium?", "answer": 52},
        ]
        captured = []
        def encoder(messages, thinking_mode):
            captured.extend(messages)
            return fake_encoder(messages, thinking_mode)
        task = experiment.make_tasks([(atomic_facts[0], atomic_facts[1])], [10])[0]
        experiment.render_prompt(encoder, task, "upstream")
        self.assertIn("some filler tokens (a sequence of dots)", captured[0]["content"])
        user_turns = [message["content"] for message in captured if message["role"] == "user"]
        self.assertEqual(len(user_turns), 6)
        self.assertEqual(
            user_turns[-1],
            "Question: What is the atomic number of Thorium plus the atomic number of Tellurium?"
            "\n\nFiller: . . . . . . . . . .\n\nAnswer:",
        )
        self.assertTrue(all("\n\nFiller: " in turn for turn in user_turns))
        self.assertEqual(
            [message["content"] for message in captured if message["role"] == "assistant"],
            ["12", "43", "95", "136", "166"],
        )

    def test_upstream_baseline_omits_filler_label_and_extra_system_instruction(self):
        atomic_facts = [
            {"fact_id": "a", "question": "What is the atomic number of Thorium?", "answer": 90},
            {"fact_id": "b", "question": "What is the atomic number of Tellurium?", "answer": 52},
        ]
        captured = []
        def encoder(messages, thinking_mode):
            captured.extend(messages)
            return fake_encoder(messages, thinking_mode)
        task = experiment.make_tasks([(atomic_facts[0], atomic_facts[1])], [0])[0]
        experiment.render_prompt(encoder, task, "upstream")
        self.assertNotIn("filler tokens", captured[0]["content"])
        self.assertTrue(all(
            "Filler:" not in message["content"]
            for message in captured if message["role"] == "user"
        ))

    def test_local_headsup_changes_only_nonzero_system_message(self):
        task_0, task_10 = experiment.make_tasks([(self.facts[0], self.facts[1])], [0, 10])

        def capture(task, variant):
            messages = []
            def encoder(rows, thinking_mode):
                messages.extend(rows)
                return fake_encoder(rows, thinking_mode)
            experiment.render_prompt(encoder, task, variant)
            return messages

        local_0 = capture(task_0, "local")
        headsup_0 = capture(task_0, "local-headsup")
        local_10 = capture(task_10, "local")
        headsup_10 = capture(task_10, "local-headsup")
        self.assertEqual(headsup_0, local_0)
        self.assertEqual(headsup_10[1:], local_10[1:])
        self.assertEqual(
            headsup_10[0]["content"],
            local_10[0]["content"]
            + " After the question, there will be 10 dots to give you extra space "
            "to process the problem before answering.",
        )
        self.assertNotIn("Filler:", "".join(row["content"] for row in headsup_10))

    def test_scoring_options_match_one_fact_defaults(self):
        args = experiment.parse_args([])
        self.assertEqual(args.top_logprobs, 20)
        self.assertIsNone(args.fact_kind)
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
