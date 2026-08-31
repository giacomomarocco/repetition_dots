from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path

import modulo10_eval as experiment


class CharacterTokenizer:
    """Small deterministic tokenizer used for model-free prompt tests."""

    is_fast = True

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        assert add_generation_prompt
        assert enable_thinking is False
        text = "".join(
            f"<{message['role']}>\n{message['content']}\n</{message['role']}>\n"
            for message in messages
        )
        text += "<assistant>\n<think>\n\n</think>\n\n"
        if tokenize:
            return self(text, add_special_tokens=False)["input_ids"]
        return text

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False):
        assert not add_special_tokens
        value = {"input_ids": [ord(character) for character in text]}
        if return_offsets_mapping:
            value["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return value


def facts(count=10):
    return [
        {
            "question": f"What is fact {index}?",
            "answer": 100 + index,
            "source_file": "facts.json",
            "source_index": index,
            "paraphrases": ["source data must remain untouched"],
            "trial_predictions": [100 + index],
        }
        for index in range(count)
    ]


def with_ids(records):
    return [{**record, "fact_id": experiment.stable_fact_id(record)} for record in records]


class ConstructionTests(unittest.TestCase):
    def test_modulo_target(self):
        self.assertEqual(experiment.target_for(37, 42), 9)
        self.assertEqual(experiment.target_for(-1, 10), 9)

    def test_exactly_five_reserved_and_excluded(self):
        records = with_ids(facts())
        few_shots, evaluation = experiment.select_few_shots(records, 42)
        self.assertEqual(len(few_shots), 5)
        self.assertEqual(len(evaluation), 5)
        self.assertTrue(
            {item["fact_id"] for item in few_shots}.isdisjoint(
                item["fact_id"] for item in evaluation
            )
        )

    def test_selection_is_seeded_and_input_order_independent(self):
        records = with_ids(facts(20))
        selected_a, _ = experiment.select_few_shots(records, 123)
        selected_b, _ = experiment.select_few_shots(list(reversed(records)), 123)
        self.assertEqual(
            [item["fact_id"] for item in selected_a],
            [item["fact_id"] for item in selected_b],
        )

    def test_addends_are_two_digit_and_deterministic(self):
        values = [experiment.deterministic_addend(42, f"fact:{index}") for index in range(500)]
        self.assertTrue(all(10 <= value <= 99 for value in values))
        self.assertEqual(values, [experiment.deterministic_addend(42, f"fact:{index}") for index in range(500)])

    def test_factual_and_numeric_share_pair_values(self):
        pair = experiment.make_pair(with_ids(facts(1))[0], 42)
        factual = experiment.render_question(pair, "factual")
        numeric = experiment.render_question(pair, "numeric")
        self.assertIn(str(pair["addend"]), factual)
        self.assertIn(str(pair["addend"]), numeric)
        self.assertIn(str(pair["answer_value"]), numeric)
        rows = experiment.prompt_task_rows([pair], "run", "tokenizer")
        self.assertEqual({row["pair_id"] for row in rows}, {pair["pair_id"]})
        self.assertEqual({row["target"] for row in rows}, {pair["target"]})
        self.assertEqual({row["addend"] for row in rows}, {pair["addend"]})
        self.assertEqual({row["answer_value"] for row in rows}, {pair["answer_value"]})

    def test_pair_reused_across_four_conditions(self):
        pair = experiment.make_pair(with_ids(facts(1))[0], 42)
        rows = experiment.prompt_task_rows([pair], "run", "tokenizer")
        for question_type in experiment.QUESTION_TYPES:
            selected = [row for row in rows if row["question_type"] == question_type]
            self.assertEqual(len(selected), 4)
            self.assertEqual({row["pair_id"] for row in selected}, {pair["pair_id"]})
            self.assertEqual({row["addend"] for row in selected}, {pair["addend"]})
            self.assertEqual({row["target"] for row in selected}, {pair["target"]})

    def test_dot_fillers_and_baseline(self):
        for k in (10, 25, 50):
            value = experiment.filler(k)
            self.assertEqual(value.count("."), k)
            self.assertEqual(value.split(" "), ["."] * k)
            self.assertNotIn("…", value)
        self.assertEqual(experiment.filler(0), "")
        self.assertEqual(experiment.assistant_content(3, 0), "Answer: 3")

    def test_all_demos_use_condition_filler(self):
        records = with_ids(facts(7))
        demos = [experiment.make_pair(record, 42) for record in records[:5]]
        pair = experiment.make_pair(records[5], 42)
        condition = experiment.CONDITIONS[2]
        messages = experiment.make_messages(pair, "factual", condition, demos)
        assistants = [message["content"] for message in messages if message["role"] == "assistant"]
        self.assertEqual(len(assistants), 5)
        for content in assistants:
            line = content.splitlines()[0]
            self.assertEqual(line, experiment.filler(condition.k))

    def test_numeric_demos_use_same_underlying_five(self):
        records = with_ids(facts(7))
        demos = [experiment.make_pair(record, 42) for record in records[:5]]
        pair = experiment.make_pair(records[5], 42)
        factual = experiment.make_messages(pair, "factual", experiment.CONDITIONS[1], demos)
        numeric = experiment.make_messages(pair, "numeric", experiment.CONDITIONS[1], demos)
        factual_answers = [m["content"] for m in factual if m["role"] == "assistant"]
        numeric_answers = [m["content"] for m in numeric if m["role"] == "assistant"]
        self.assertEqual(factual_answers, numeric_answers)
        numeric_users = [m["content"] for m in numeric if m["role"] == "user"][:5]
        self.assertEqual(
            numeric_users,
            [experiment.render_question(demo, "numeric") for demo in demos],
        )

    def test_each_digit_is_one_appended_token(self):
        tokenizer = CharacterTokenizer()
        mapping = experiment.validate_digit_continuations(tokenizer, "Answer: ")
        self.assertEqual(mapping, {digit: ord(digit) for digit in "0123456789"})

    def test_spans_match_rendered_prompt_and_no_think(self):
        records = with_ids(facts(7))
        demos = [experiment.make_pair(record, 42) for record in records[:5]]
        pair = experiment.make_pair(records[5], 42)
        prompt = experiment.render_prompt(
            CharacterTokenizer(), "fake", pair, "factual", experiment.CONDITIONS[2], demos
        )
        text = prompt["rendered_prompt"]
        self.assertNotIn("<think>", text)
        fs, fe = prompt["filler_char_span"]
        self.assertEqual(text[fs:fe], experiment.filler(25))
        self.assertEqual(len(prompt["filler_item_token_spans"]), 25)
        for char_span, token_span in zip(
            prompt["filler_item_char_spans"], prompt["filler_item_token_spans"]
        ):
            self.assertEqual(text[slice(*char_span)], ".")
            self.assertEqual(token_span[1] - token_span[0], 1)
        aps, ape = prompt["answer_prefix_char_span"]
        self.assertEqual(text[aps:ape], "Answer: ")
        self.assertEqual(prompt["next_token_position"], len(prompt["input_ids"]))

    def test_default_result_count(self):
        records = with_ids(facts(17))
        few_shots, evaluation = experiment.select_few_shots(records, 42)
        self.assertEqual(len(few_shots), 5)
        pairs = experiment.make_pairs(evaluation, 42, 1)
        rows = experiment.prompt_task_rows(pairs, "run", "tokenizer")
        self.assertEqual(len(rows), 8 * (len(records) - 5))

    def test_source_data_not_modified(self):
        original = facts(8)
        snapshot = copy.deepcopy(original)
        records = with_ids(original)
        experiment.select_few_shots(records, 42)
        experiment.make_pairs(records, 42, 1)
        self.assertEqual(original, snapshot)

    def test_fixed_seed_prompts_and_ids_identical(self):
        records = with_ids(facts(8))
        demos = [experiment.make_pair(record, 42) for record in records[:5]]
        pair_a = experiment.make_pair(records[5], 42)
        pair_b = experiment.make_pair(records[5], 42)
        prompt_a = experiment.render_prompt(
            CharacterTokenizer(), "fake", pair_a, "numeric", experiment.CONDITIONS[3], demos
        )
        prompt_b = experiment.render_prompt(
            CharacterTokenizer(), "fake", pair_b, "numeric", experiment.CONDITIONS[3], demos
        )
        self.assertEqual(prompt_a, prompt_b)

    def test_malformed_integer_answers_are_reported_and_file_unchanged(self):
        records = facts(6)
        records[2]["answer"] = 3.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "facts.json"
            original = json.dumps(records)
            path.write_text(original, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"\[2\]\.answer"):
                experiment.load_facts(path)
            self.assertEqual(path.read_text(encoding="utf-8"), original)

    def test_paired_analysis_transitions_and_four_way_counts(self):
        results = []
        # Baseline factual=[wrong, right], numeric=[right, wrong]. dots_10
        # changes factual to [right, wrong] while numeric becomes [right, right].
        patterns = {
            ("factual", "baseline"): [False, True],
            ("numeric", "baseline"): [True, False],
            ("factual", "dots_10"): [True, False],
            ("numeric", "dots_10"): [True, True],
        }
        for question_type in experiment.QUESTION_TYPES:
            for condition in experiment.CONDITIONS:
                values = patterns.get((question_type, condition.name), [False, False])
                for index, correct in enumerate(values):
                    results.append(
                        {
                            "pair_id": f"pair-{index}",
                            "question_type": question_type,
                            "condition": condition.name,
                            "correct": correct,
                            "digit_constrained_correct": correct,
                        }
                    )
        summary, csv_rows = experiment.summarize_results(results, 42, 100)
        dots = summary["question_types"]["factual"]["dots_10"]
        self.assertEqual(dots["unconstrained_wrong_to_right"], 1)
        self.assertEqual(dots["unconstrained_right_to_wrong"], 1)
        self.assertEqual(dots["unconstrained_absolute_change_from_baseline"], 0.0)
        paired = summary["paired_factual_numeric"]["dots_10"]
        self.assertEqual(paired["numeric_correct_pair_count"], 2)
        self.assertEqual(paired["factual_accuracy_when_numeric_correct"], 0.5)
        self.assertEqual(
            paired["outcome_counts"],
            {
                "factual_correct_numeric_correct": 1,
                "factual_correct_numeric_wrong": 0,
                "factual_wrong_numeric_correct": 1,
                "factual_wrong_numeric_wrong": 0,
            },
        )
        self.assertEqual(len(csv_rows), 8)


class ActualQwenTokenizerTests(unittest.TestCase):
    def test_actual_prefix_digits_and_template_have_no_think(self):
        try:
            from transformers import AutoTokenizer
        except ImportError:
            self.skipTest("Transformers is not installed")
        cache = Path(__file__).resolve().parents[1] / ".hf-cache"
        previous = os.environ.get("HF_HOME")
        os.environ["HF_HOME"] = str(cache)
        try:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    experiment.DEFAULT_MODEL, local_files_only=True
                )
            except OSError as error:
                self.skipTest(f"local Qwen tokenizer is unavailable: {error}")
            records = with_ids(facts(7))
            demos = [experiment.make_pair(record, 42) for record in records[:5]]
            pair = experiment.make_pair(records[5], 42)
            for condition in experiment.CONDITIONS:
                prompt = experiment.render_prompt(
                    tokenizer,
                    experiment.DEFAULT_MODEL,
                    pair,
                    "numeric",
                    condition,
                    demos,
                )
                self.assertNotIn("<think>", prompt["rendered_prompt"])
                self.assertEqual(len(set(prompt["digit_token_ids"].values())), 10)
        finally:
            if previous is None:
                os.environ.pop("HF_HOME", None)
            else:
                os.environ["HF_HOME"] = previous


if __name__ == "__main__":
    unittest.main()
