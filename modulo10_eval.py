#!/usr/bin/env python3
"""Paired dot-filler evaluation for modulo-10 one-fact addition."""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import importlib.metadata
import io
import json
import math
import os
import random
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_SEED = 42
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
QUESTION_TYPES = ("factual", "numeric")
FILLER_RULE = '" ".join(["."] * k)'


@dataclass(frozen=True)
class Condition:
    name: str
    k: int
    system_prompt: str


def system_prompt(k: int) -> str:
    if k == 0:
        return (
            'Solve each modulo-10 problem. Give no explanation. Respond with '
            '"Answer: " followed by exactly one digit from 0 to 9.'
        )
    return (
        f'Solve each modulo-10 problem. Before "Answer:", include exactly {k} '
        'periods separated by single spaces as filler. Give no other explanation. '
        'End with "Answer: " followed by exactly one digit from 0 to 9.'
    )


CONDITIONS = tuple(
    Condition(name, k, system_prompt(k))
    for name, k in (("baseline", 0), ("dots_10", 10), ("dots_25", 25), ("dots_50", 50))
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", required=True, type=Path, help="filtered fact JSON array")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", default=None, help="defaults to --model")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--addends-per-fact", type=int, default=1)
    parser.add_argument(
        "--prompt-only",
        action="store_true",
        help="construct and validate prompts without loading model weights",
    )
    parser.add_argument(
        "--inspect-fact-id",
        help="construct only this evaluation fact and print its eight prompts",
    )
    parser.add_argument(
        "--max-eval-facts",
        type=int,
        help="limit evaluation facts after stable sorting (for smoke tests only)",
    )
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument(
        "--device",
        choices=("auto", "mps", "cpu"),
        default="auto",
        help="model device (default: auto)",
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="never download model/tokenizer files (default: true)",
    )
    args = parser.parse_args(argv)
    if args.addends_per_fact < 1:
        parser.error("--addends-per-fact must be at least 1")
    if args.max_eval_facts is not None and args.max_eval_facts < 1:
        parser.error("--max-eval-facts must be at least 1")
    if args.top_n < 1:
        parser.error("--top-n must be at least 1")
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be at least 1")
    if args.inspect_fact_id and not args.prompt_only:
        parser.error("--inspect-fact-id requires --prompt-only")
    return args


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def seeded_digest(seed: int, namespace: str, value: str) -> bytes:
    text = f"{seed}\0{namespace}\0{value}".encode("utf-8")
    return hashlib.sha256(text).digest()


def stable_fact_id(record: dict[str, Any]) -> str:
    source_file = record.get("source_file")
    source_index = record.get("source_index")
    if isinstance(source_file, str) and source_file and source_index is not None:
        if isinstance(source_index, bool) or not isinstance(source_index, (int, str)):
            raise ValueError("source_index must be an integer or string when source_file is present")
        return f"{source_file}:{source_index}"
    question = record.get("question")
    if not isinstance(question, str) or not question:
        raise ValueError("cannot construct fallback fact ID without a non-empty question")
    return f"sha256:{sha256_bytes(question.encode('utf-8'))}"


def load_facts(path: Path) -> tuple[list[dict[str, Any]], str]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON: {error}") from error
    if not isinstance(value, list):
        raise ValueError(f"{path}: top-level value must be a JSON array")
    facts: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    errors: list[str] = []
    for index, record in enumerate(value):
        label = f"{path}[{index}]"
        if not isinstance(record, dict):
            errors.append(f"{label}: record must be an object")
            continue
        question = record.get("question")
        answer = record.get("answer")
        if not isinstance(question, str) or not question.strip():
            errors.append(f"{label}.question: expected a non-empty string")
        if isinstance(answer, bool) or not isinstance(answer, int):
            errors.append(f"{label}.answer: expected an integer, got {answer!r}")
        try:
            fact_id = stable_fact_id(record)
        except ValueError as error:
            errors.append(f"{label}: {error}")
            continue
        if fact_id in seen:
            errors.append(f"{label}: duplicate fact ID {fact_id!r} (first at index {seen[fact_id]})")
        else:
            seen[fact_id] = index
        copied = copy.deepcopy(record)
        copied["fact_id"] = fact_id
        facts.append(copied)
    if errors:
        joined = "\n  - ".join(errors)
        raise ValueError(f"malformed fact records:\n  - {joined}")
    if len(facts) < 6:
        raise ValueError(f"{path}: need at least 6 valid facts; found {len(facts)}")
    return facts, sha256_bytes(raw)


def select_few_shots(
    facts: Sequence[dict[str, Any]], seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ranked = sorted(
        facts,
        key=lambda fact: (seeded_digest(seed, "few-shot", fact["fact_id"]), fact["fact_id"]),
    )
    few_shots = list(ranked[:5])
    reserved = {fact["fact_id"] for fact in few_shots}
    evaluation = sorted(
        (fact for fact in facts if fact["fact_id"] not in reserved),
        key=lambda fact: fact["fact_id"],
    )
    return few_shots, evaluation


def deterministic_addend(seed: int, fact_id: str, addend_index: int = 0) -> int:
    digest = seeded_digest(seed, "addend", f"{fact_id}\0{addend_index}")
    return 10 + int.from_bytes(digest[:8], "big") % 90


def target_for(answer: int, addend: int) -> int:
    return (answer + addend) % 10


def make_pair(fact: dict[str, Any], seed: int, addend_index: int = 0) -> dict[str, Any]:
    answer = fact["answer"]
    addend = deterministic_addend(seed, fact["fact_id"], addend_index)
    identity = f"{seed}\0{fact['fact_id']}\0{addend_index}\0{addend}"
    return {
        "pair_id": f"pair-{sha256_bytes(identity.encode('utf-8'))[:20]}",
        "fact_id": fact["fact_id"],
        "question": fact["question"],
        "answer_value": answer,
        "answer_mod_10": answer % 10,
        "addend_index": addend_index,
        "addend": addend,
        "addend_mod_10": addend % 10,
        "target": target_for(answer, addend),
    }


def make_pairs(
    facts: Sequence[dict[str, Any]], seed: int, addends_per_fact: int
) -> list[dict[str, Any]]:
    return [
        make_pair(fact, seed, addend_index)
        for fact in facts
        for addend_index in range(addends_per_fact)
    ]


def filler(k: int) -> str:
    if k < 0:
        raise ValueError("filler length cannot be negative")
    return " ".join(["."] * k)


def render_question(pair: dict[str, Any], question_type: str) -> str:
    if question_type == "factual":
        return (
            "What is (the numeric answer to the fact question below + "
            f"{pair['addend']}) modulo 10?\nFact question: {pair['question']}"
        )
    if question_type == "numeric":
        return f"What is ({pair['answer_value']} + {pair['addend']}) modulo 10?"
    raise ValueError(f"unknown question type: {question_type}")


def assistant_content(target: int, k: int, trailing_space: bool = False) -> str:
    answer = f"Answer: {'' if trailing_space else target}"
    return f"{filler(k)}\n{answer}" if k else answer


def make_messages(
    pair: dict[str, Any],
    question_type: str,
    condition: Condition,
    demo_pairs: Sequence[dict[str, Any]],
) -> list[dict[str, str]]:
    if len(demo_pairs) != 5:
        raise ValueError(f"expected exactly five demonstration pairs, got {len(demo_pairs)}")
    messages: list[dict[str, str]] = [
        {"role": "system", "content": condition.system_prompt}
    ]
    for demo in demo_pairs:
        messages.extend(
            [
                {"role": "user", "content": render_question(demo, question_type)},
                {
                    "role": "assistant",
                    "content": assistant_content(demo["target"], condition.k),
                },
            ]
        )
    messages.append({"role": "user", "content": render_question(pair, question_type)})
    return messages


def apply_chat_template(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError as error:
        raise RuntimeError(
            "tokenizer chat template does not accept enable_thinking=False; "
            "Qwen thinking must be explicitly disabled"
        ) from error
    if not isinstance(rendered, str):
        raise TypeError("chat template did not return text")
    # As of Transformers 5.16.1, Qwen3.5's official template emits an empty
    # thinking block even with thinking disabled. Remove that exact, empty
    # generation suffix so the supplied assistant prefix immediately follows
    # the official assistant role marker. Never remove a non-empty block.
    empty_thinking_suffix = "<think>\n\n</think>\n\n"
    if rendered.endswith(empty_thinking_suffix):
        rendered = rendered[: -len(empty_thinking_suffix)]
    tail = rendered[len(rendered) // 2 :].lower()
    if "<think>" in tail or "</think>" in tail:
        raise RuntimeError("chat template inserted a non-empty or unrecognized <think> section")
    return rendered


def encode_with_offsets(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = encoded["input_ids"]
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        raise RuntimeError("tokenizer must be a fast tokenizer with offset mappings")
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
        offsets = offsets[0]
    ids = [int(value) for value in input_ids]
    spans = [(int(start), int(end)) for start, end in offsets]
    if len(ids) != len(spans):
        raise RuntimeError("token IDs and offset mapping have different lengths")
    if any(start < 0 or end < start or end > len(text) for start, end in spans):
        raise RuntimeError("tokenizer returned an invalid character offset")
    return ids, spans


def char_span_to_token_span(
    offsets: Sequence[tuple[int, int]], char_span: Sequence[int]
) -> list[int]:
    start, end = char_span
    if start == end:
        position = next((i for i, (left, _) in enumerate(offsets) if left >= start), len(offsets))
        return [position, position]
    indices = [
        index
        for index, (left, right) in enumerate(offsets)
        if right > start and left < end
    ]
    if not indices:
        raise RuntimeError(f"character span {list(char_span)} does not overlap any token")
    return [indices[0], indices[-1] + 1]


def validate_span(
    text: str,
    offsets: Sequence[tuple[int, int]],
    char_span: Sequence[int],
    token_span: Sequence[int],
    expected: str,
    label: str,
) -> None:
    start, end = char_span
    if text[start:end] != expected:
        raise RuntimeError(
            f"{label} character span mismatch: expected {expected!r}, got {text[start:end]!r}"
        )
    calculated = char_span_to_token_span(offsets, char_span)
    if list(token_span) != calculated:
        raise RuntimeError(
            f"{label} token span mismatch: stored {list(token_span)}, calculated {calculated}"
        )


def validate_digit_continuations(
    tokenizer: Any, rendered_prompt: str, input_ids: Sequence[int] | None = None
) -> dict[str, int]:
    if input_ids is None:
        input_ids, _ = encode_with_offsets(tokenizer, rendered_prompt)
    prefix = list(input_ids)
    digit_ids: dict[str, int] = {}
    for digit in "0123456789":
        combined, _ = encode_with_offsets(tokenizer, rendered_prompt + digit)
        if combined[: len(prefix)] != prefix or len(combined) != len(prefix) + 1:
            common = 0
            for old, new in zip(prefix, combined):
                if old != new:
                    break
                common += 1
            raise RuntimeError(
                f"digit {digit!r} is not one appended continuation token after the actual "
                f"assistant prefix (prefix tokens={len(prefix)}, combined tokens={len(combined)}, "
                f"unchanged prefix tokens={common})"
            )
        digit_ids[digit] = combined[-1]
    if len(set(digit_ids.values())) != 10:
        raise RuntimeError(f"digit continuation token IDs are not distinct: {digit_ids}")
    return digit_ids


def make_prompt_id(
    pair_id: str, question_type: str, condition: Condition, tokenizer_id: str
) -> str:
    identity = canonical_json(
        {
            "pair_id": pair_id,
            "question_type": question_type,
            "condition": condition.name,
            "k": condition.k,
            "tokenizer": tokenizer_id,
        }
    )
    return f"prompt-{sha256_bytes(identity.encode('utf-8'))[:24]}"


def render_prompt(
    tokenizer: Any,
    tokenizer_id: str,
    pair: dict[str, Any],
    question_type: str,
    condition: Condition,
    demo_pairs: Sequence[dict[str, Any]],
    known_digit_ids: dict[str, int] | None = None,
) -> dict[str, Any]:
    messages = make_messages(pair, question_type, condition, demo_pairs)
    question = messages[-1]["content"]
    chat_prefix = apply_chat_template(tokenizer, messages)
    supplied_prefix = assistant_content(pair["target"], condition.k, trailing_space=True)
    rendered = chat_prefix + supplied_prefix
    input_ids, offsets = encode_with_offsets(tokenizer, rendered)

    chat_ids, _ = encode_with_offsets(tokenizer, chat_prefix)
    if input_ids[: len(chat_ids)] != chat_ids:
        raise RuntimeError("assistant prefill retokenized the official chat-template prefix")

    q_start = chat_prefix.rfind(question)
    if q_start < 0:
        raise RuntimeError("could not locate the current test question in the rendered prompt")
    q_span = [q_start, q_start + len(question)]
    role_span = [q_span[1], len(chat_prefix)]
    assistant_span = [len(chat_prefix), len(rendered)]
    filler_text = filler(condition.k)
    if condition.k:
        filler_span = [assistant_span[0], assistant_span[0] + len(filler_text)]
        answer_start = filler_span[1] + 1
    else:
        filler_span = [assistant_span[0], assistant_span[0]]
        answer_start = assistant_span[0]
    answer_text = "Answer: "
    answer_span = [answer_start, answer_start + len(answer_text)]
    dot_char_spans = [
        [filler_span[0] + 2 * index, filler_span[0] + 2 * index + 1]
        for index in range(condition.k)
    ]

    q_token_span = char_span_to_token_span(offsets, q_span)
    role_token_span = char_span_to_token_span(offsets, role_span)
    assistant_token_span = char_span_to_token_span(offsets, assistant_span)
    filler_token_span = char_span_to_token_span(offsets, filler_span)
    answer_token_span = char_span_to_token_span(offsets, answer_span)
    dot_token_spans = [char_span_to_token_span(offsets, span) for span in dot_char_spans]

    validate_span(rendered, offsets, q_span, q_token_span, question, "question")
    validate_span(rendered, offsets, filler_span, filler_token_span, filler_text, "filler")
    validate_span(rendered, offsets, answer_span, answer_token_span, answer_text, "answer prefix")
    for index, (char_span, token_span) in enumerate(zip(dot_char_spans, dot_token_spans)):
        validate_span(rendered, offsets, char_span, token_span, ".", f"filler dot {index}")
    if rendered[assistant_span[0] : assistant_span[1]] != supplied_prefix:
        raise RuntimeError("assistant prefix span does not match the supplied prefix")
    if rendered[filler_span[1] : answer_span[0]] != ("\n" if condition.k else ""):
        raise RuntimeError("unexpected separator between filler and Answer: prefix")
    if answer_span[1] != len(rendered):
        raise RuntimeError("rendered prompt has content after the supplied Answer: prefix")

    actual_digit_ids = validate_digit_continuations(tokenizer, rendered, input_ids)
    if known_digit_ids is not None and actual_digit_ids != known_digit_ids:
        raise RuntimeError(
            f"digit token IDs changed across prompts: expected {known_digit_ids}, got {actual_digit_ids}"
        )
    prompt_id = make_prompt_id(pair["pair_id"], question_type, condition, tokenizer_id)
    return {
        "prompt_id": prompt_id,
        "pair_id": pair["pair_id"],
        "fact_id": pair["fact_id"],
        "question_type": question_type,
        "condition": condition.name,
        "k": condition.k,
        "rendered_prompt": rendered,
        "input_ids": input_ids,
        "token_offsets": [list(span) for span in offsets],
        "question_char_span": q_span,
        "question_token_span": q_token_span,
        "question_end_char_position": q_span[1],
        "question_end_token_position": q_token_span[1],
        "assistant_role_prefix_char_span": role_span,
        "assistant_role_prefix_token_span": role_token_span,
        "assistant_prefix_char_span": assistant_span,
        "assistant_prefix_token_span": assistant_token_span,
        "filler_char_span": filler_span,
        "filler_token_span": filler_token_span,
        "filler_item_char_spans": dot_char_spans,
        "filler_item_token_spans": dot_token_spans,
        "filler_item_end_char_positions": [span[1] for span in dot_char_spans],
        "filler_item_end_token_positions": [span[1] for span in dot_token_spans],
        "filler_item_last_token_indices": [span[1] - 1 for span in dot_token_spans],
        "answer_prefix_char_span": answer_span,
        "answer_prefix_token_span": answer_token_span,
        "prediction_source_token_index": len(input_ids) - 1,
        "next_token_position": len(input_ids),
        "answer_append_char_position": len(rendered),
        "answer_append_token_position": len(input_ids),
        "target_token_id": actual_digit_ids[str(pair["target"])],
        "digit_token_ids": actual_digit_ids,
    }


def prompt_task_rows(
    pairs: Sequence[dict[str, Any]],
    run_id: str,
    tokenizer_id: str,
) -> list[dict[str, Any]]:
    rows = []
    for pair in pairs:
        for question_type in QUESTION_TYPES:
            for condition in CONDITIONS:
                rows.append(
                    {
                        "run_id": run_id,
                        "prompt_id": make_prompt_id(
                            pair["pair_id"], question_type, condition, tokenizer_id
                        ),
                        "pair_id": pair["pair_id"],
                        "fact_id": pair["fact_id"],
                        "question_type": question_type,
                        "condition": condition.name,
                        "k": condition.k,
                        "answer_value": pair["answer_value"],
                        "answer_mod_10": pair["answer_mod_10"],
                        "addend_index": pair["addend_index"],
                        "addend": pair["addend"],
                        "addend_mod_10": pair["addend_mod_10"],
                        "target": pair["target"],
                    }
                )
    return rows


def build_prompts(
    tokenizer: Any,
    tokenizer_id: str,
    pairs: Sequence[dict[str, Any]],
    demo_pairs: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    digit_ids_by_condition: dict[str, dict[str, int]] = {}
    for pair in pairs:
        for question_type in QUESTION_TYPES:
            for condition in CONDITIONS:
                prompt = render_prompt(
                    tokenizer,
                    tokenizer_id,
                    pair,
                    question_type,
                    condition,
                    demo_pairs,
                    digit_ids_by_condition.get(condition.name),
                )
                digit_ids_by_condition.setdefault(condition.name, prompt["digit_token_ids"])
                prompts.append(prompt)
    expected = len(pairs) * len(QUESTION_TYPES) * len(CONDITIONS)
    if len(prompts) != expected or len({row["prompt_id"] for row in prompts}) != expected:
        raise RuntimeError("prompt construction produced a missing or duplicate prompt ID")
    return prompts


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": sys.version.split()[0]}
    for package in ("torch", "transformers", "accelerate", "huggingface-hub", "tokenizers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def resolved_revision(value: Any) -> str | None:
    for container in (getattr(value, "init_kwargs", None), getattr(value, "config", None), value):
        if isinstance(container, dict):
            revision = container.get("_commit_hash") or container.get("commit_hash")
        else:
            revision = getattr(container, "_commit_hash", None) if container is not None else None
        if revision:
            return str(revision)
    return None


def resolve_hf_revision(
    value: Any, identifier: str, requested_revision: str | None, filename: str
) -> str | None:
    revision = resolved_revision(value)
    if revision:
        return revision
    path = Path(identifier)
    if path.exists():
        return None
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(
            identifier, filename, revision=requested_revision or "main"
        )
    except (OSError, ValueError):
        return None
    if not isinstance(cached, str):
        return None
    parts = Path(cached).parts
    try:
        return parts[parts.index("snapshots") + 1]
    except (ValueError, IndexError):
        return None


def make_run_id(
    source_sha: str,
    seed: int,
    model_id: str,
    tokenizer_id: str,
    addends_per_fact: int,
) -> str:
    identity = canonical_json(
        {
            "source_sha256": source_sha,
            "seed": seed,
            "model": model_id,
            "tokenizer": tokenizer_id,
            "addends_per_fact": addends_per_fact,
            "conditions": [(condition.name, condition.k) for condition in CONDITIONS],
        }
    )
    return f"run-{sha256_bytes(identity.encode('utf-8'))[:20]}"


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def write_prompts(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def lightweight_fact(fact: dict[str, Any], role: str) -> dict[str, Any]:
    return {
        "fact_id": fact["fact_id"],
        "question": fact["question"],
        "answer": fact["answer"],
        "category": fact.get("category"),
        "name": fact.get("name"),
        "kind": fact.get("kind"),
        "source_file": fact.get("source_file"),
        "source_index": fact.get("source_index"),
        "role": role,
    }


def exact_mcnemar(baseline: Sequence[bool], treatment: Sequence[bool]) -> dict[str, Any]:
    wrong_to_right = sum(not old and new for old, new in zip(baseline, treatment))
    right_to_wrong = sum(old and not new for old, new in zip(baseline, treatment))
    discordant = wrong_to_right + right_to_wrong
    if discordant == 0:
        p_value = 1.0
    else:
        tail = min(wrong_to_right, right_to_wrong)
        probability = sum(math.comb(discordant, i) for i in range(tail + 1)) / (2**discordant)
        p_value = min(1.0, 2 * probability)
    return {
        "wrong_to_right": wrong_to_right,
        "right_to_wrong": right_to_wrong,
        "discordant": discordant,
        "exact_two_sided_p": p_value,
    }


def percentile(sorted_values: Sequence[float], probability: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def bootstrap_accuracy_change(
    baseline: Sequence[bool],
    treatment: Sequence[bool],
    seed: int,
    namespace: str,
    samples: int,
) -> list[float]:
    if len(baseline) != len(treatment) or not baseline:
        raise ValueError("paired bootstrap needs equally sized non-empty samples")
    differences = [int(new) - int(old) for old, new in zip(baseline, treatment)]
    rng_seed = int.from_bytes(seeded_digest(seed, "bootstrap", namespace)[:8], "big")
    rng = random.Random(rng_seed)
    count = len(differences)
    values = sorted(
        sum(differences[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(samples)
    )
    return [percentile(values, 0.025), percentile(values, 0.975)]


def summarize_results(
    results: Sequence[dict[str, Any]], seed: int, bootstrap_samples: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_key = {(row["pair_id"], row["question_type"], row["condition"]): row for row in results}
    pair_ids = sorted({row["pair_id"] for row in results})
    expected = len(pair_ids) * len(QUESTION_TYPES) * len(CONDITIONS)
    if len(results) != expected or len(by_key) != expected:
        raise ValueError("results are incomplete or contain duplicate paired cells")

    summary: dict[str, Any] = {
        "primary_metric": "unconstrained greedy next-token exact-digit accuracy",
        "bootstrap_samples": bootstrap_samples,
        "resampling_unit": "fact/addend pair_id",
        "question_types": {},
        "paired_factual_numeric": {},
    }
    csv_rows: list[dict[str, Any]] = []
    for question_type in QUESTION_TYPES:
        type_summary: dict[str, Any] = {}
        baseline_rows = [by_key[(pair_id, question_type, "baseline")] for pair_id in pair_ids]
        baseline_correct = [bool(row["correct"]) for row in baseline_rows]
        baseline_constrained = [bool(row["digit_constrained_correct"]) for row in baseline_rows]
        for condition in CONDITIONS:
            rows = [by_key[(pair_id, question_type, condition.name)] for pair_id in pair_ids]
            correct = [bool(row["correct"]) for row in rows]
            constrained = [bool(row["digit_constrained_correct"]) for row in rows]
            total = len(rows)
            accuracy = sum(correct) / total
            constrained_accuracy = sum(constrained) / total
            transitions = exact_mcnemar(baseline_correct, correct)
            constrained_transitions = exact_mcnemar(baseline_constrained, constrained)
            if condition.k == 0:
                ci = [0.0, 0.0]
                constrained_ci = [0.0, 0.0]
            else:
                ci = bootstrap_accuracy_change(
                    baseline_correct,
                    correct,
                    seed,
                    f"{question_type}:{condition.name}:unconstrained",
                    bootstrap_samples,
                )
                constrained_ci = bootstrap_accuracy_change(
                    baseline_constrained,
                    constrained,
                    seed,
                    f"{question_type}:{condition.name}:constrained",
                    bootstrap_samples,
                )
            item = {
                "k": condition.k,
                "total": total,
                "unconstrained_accuracy": accuracy,
                "digit_constrained_accuracy": constrained_accuracy,
                "unconstrained_absolute_change_from_baseline": accuracy
                - sum(baseline_correct) / total,
                "digit_constrained_absolute_change_from_baseline": constrained_accuracy
                - sum(baseline_constrained) / total,
                "unconstrained_wrong_to_right": transitions["wrong_to_right"],
                "unconstrained_right_to_wrong": transitions["right_to_wrong"],
                "digit_constrained_wrong_to_right": constrained_transitions["wrong_to_right"],
                "digit_constrained_right_to_wrong": constrained_transitions["right_to_wrong"],
                "mcnemar_unconstrained": transitions,
                "mcnemar_digit_constrained": constrained_transitions,
                "unconstrained_change_bootstrap_95_ci": ci,
                "digit_constrained_change_bootstrap_95_ci": constrained_ci,
            }
            type_summary[condition.name] = item
            csv_rows.append({"question_type": question_type, "condition": condition.name, **item})
        summary["question_types"][question_type] = type_summary

    for condition in CONDITIONS:
        outcomes = {
            "factual_correct_numeric_correct": 0,
            "factual_correct_numeric_wrong": 0,
            "factual_wrong_numeric_correct": 0,
            "factual_wrong_numeric_wrong": 0,
        }
        numeric_correct_pairs = []
        factual_correct_when_numeric_correct = 0
        for pair_id in pair_ids:
            factual_ok = bool(by_key[(pair_id, "factual", condition.name)]["correct"])
            numeric_ok = bool(by_key[(pair_id, "numeric", condition.name)]["correct"])
            key = (
                ("factual_correct" if factual_ok else "factual_wrong")
                + "_"
                + ("numeric_correct" if numeric_ok else "numeric_wrong")
            )
            outcomes[key] += 1
            if numeric_ok:
                numeric_correct_pairs.append(pair_id)
                factual_correct_when_numeric_correct += int(factual_ok)
        denominator = len(numeric_correct_pairs)
        summary["paired_factual_numeric"][condition.name] = {
            "k": condition.k,
            "outcome_counts": outcomes,
            "numeric_correct_pair_count": denominator,
            "factual_accuracy_when_numeric_correct": (
                factual_correct_when_numeric_correct / denominator if denominator else None
            ),
        }
        for row in csv_rows:
            if row["question_type"] == "factual" and row["condition"] == condition.name:
                row.update(summary["paired_factual_numeric"][condition.name])
                row.pop("outcome_counts", None)
                row.update(outcomes)
    return summary, csv_rows


def write_summary_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    flattened = []
    for row in rows:
        flat: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, (dict, list)):
                flat[key] = canonical_json(value)
            else:
                flat[key] = value
        flattened.append(flat)
    fieldnames: list[str] = []
    for row in flattened:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flattened)
    temporary.replace(path)


def load_tokenizer(args: argparse.Namespace) -> Any:
    from transformers import AutoTokenizer

    tokenizer_id = args.tokenizer or args.model
    kwargs: dict[str, Any] = {"local_files_only": args.local_files_only}
    revision = args.tokenizer_revision or args.revision
    if revision:
        kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, **kwargs)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("a fast tokenizer is required for exact offset mappings")
    return tokenizer


def load_model(args: argparse.Namespace) -> tuple[Any, str]:
    import torch
    from transformers import Qwen3_5ForCausalLM

    mps_available = torch.backends.mps.is_available()
    if args.device == "mps" and not mps_available:
        raise RuntimeError(
            "--device mps was requested, but torch.backends.mps.is_available() is false; "
            "on macOS this can mean the process is running in a sandbox that hides Metal"
        )
    use_mps = mps_available if args.device == "auto" else args.device == "mps"
    device = "mps" if use_mps else "cpu"
    dtype = torch.bfloat16 if device == "mps" else torch.float32
    kwargs: dict[str, Any] = {
        "dtype": dtype,
        "device_map": device,
        "local_files_only": args.local_files_only,
    }
    if args.revision:
        kwargs["revision"] = args.revision
    print(f"Loading {args.model} on {device} ({dtype})...", flush=True)
    model = Qwen3_5ForCausalLM.from_pretrained(args.model, **kwargs)
    model.eval()
    return model, device


def evaluate(
    model: Any,
    tokenizer: Any,
    device: str,
    task_rows: Sequence[dict[str, Any]],
    prompts: Sequence[dict[str, Any]],
    top_n: int,
) -> list[dict[str, Any]]:
    import torch

    prompt_by_id = {prompt["prompt_id"]: prompt for prompt in prompts}
    results: list[dict[str, Any]] = []
    for index, task in enumerate(task_rows, start=1):
        prompt = prompt_by_id[task["prompt_id"]]
        input_ids = torch.tensor([prompt["input_ids"]], dtype=torch.long, device=device)
        with torch.inference_mode():
            logits = model(input_ids=input_ids, use_cache=False).logits[0, -1].float()
        log_probs = torch.log_softmax(logits, dim=-1)
        prediction_token_id = int(torch.argmax(logits).item())
        prediction_text = tokenizer.decode(
            [prediction_token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        prediction_digit = int(prediction_text) if len(prediction_text) == 1 and prediction_text.isdigit() else None
        digit_token_ids = prompt["digit_token_ids"]
        digit_values = torch.tensor(
            [digit_token_ids[str(digit)] for digit in range(10)],
            dtype=torch.long,
            device=logits.device,
        )
        digit_argmax = int(torch.argmax(logits[digit_values]).item())
        top_values, top_indices = torch.topk(logits, min(top_n, logits.numel()))
        top_log_probs = log_probs[top_indices]
        result = {
            **task,
            "prediction_token_id": prediction_token_id,
            "prediction_text": prediction_text,
            "prediction_digit": prediction_digit,
            "correct": prediction_text == str(task["target"]),
            "digit_logits": {
                str(digit): float(logits[token_id].item())
                for digit, token_id in ((d, digit_token_ids[str(d)]) for d in range(10))
            },
            "digit_log_probabilities": {
                str(digit): float(log_probs[token_id].item())
                for digit, token_id in ((d, digit_token_ids[str(d)]) for d in range(10))
            },
            "digit_constrained_prediction": digit_argmax,
            "digit_constrained_correct": digit_argmax == task["target"],
            "top_predictions": [
                {
                    "token_id": int(token_id.item()),
                    "text": tokenizer.decode(
                        [int(token_id.item())],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    ),
                    "logit": float(value.item()),
                    "log_probability": float(log_probability.item()),
                }
                for value, token_id, log_probability in zip(top_values, top_indices, top_log_probs)
            ],
        }
        results.append(result)
        print(
            f"[{index}/{len(task_rows)}] {task['question_type']} {task['condition']} "
            f"target={task['target']} prediction={prediction_text!r}",
            flush=True,
        )
    return results


def print_inspection(prompts: Sequence[dict[str, Any]]) -> None:
    for prompt in prompts:
        print(
            f"\n===== {prompt['question_type']} / {prompt['condition']} "
            f"({prompt['prompt_id']}) =====\n"
        )
        print(prompt["rendered_prompt"])
        print(
            f"\n[input tokens={len(prompt['input_ids'])}; "
            f"filler token span={prompt['filler_token_span']}; "
            f"next token position={prompt['next_token_position']}]"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    # Match the repository's existing local-cache and MPS safety conventions.
    os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    facts, source_sha = load_facts(args.facts)
    few_shots, evaluation_facts = select_few_shots(facts, args.seed)
    full_default_row_count = (len(facts) - 5) * args.addends_per_fact * 8
    if args.inspect_fact_id:
        matches = [fact for fact in evaluation_facts if fact["fact_id"] == args.inspect_fact_id]
        if not matches:
            reserved = {fact["fact_id"] for fact in few_shots}
            detail = " (it is reserved as a few-shot)" if args.inspect_fact_id in reserved else ""
            raise ValueError(f"evaluation fact ID {args.inspect_fact_id!r} not found{detail}")
        evaluation_facts = matches
    elif args.max_eval_facts is not None:
        evaluation_facts = evaluation_facts[: args.max_eval_facts]

    demo_pairs = [make_pair(fact, args.seed) for fact in few_shots]
    pairs = make_pairs(evaluation_facts, args.seed, args.addends_per_fact)
    tokenizer_id = args.tokenizer or args.model
    run_id = make_run_id(source_sha, args.seed, args.model, tokenizer_id, args.addends_per_fact)
    task_rows = prompt_task_rows(pairs, run_id, tokenizer_id)

    print(f"Loading tokenizer {tokenizer_id}...", flush=True)
    tokenizer = load_tokenizer(args)
    prompts = build_prompts(tokenizer, tokenizer_id, pairs, demo_pairs)
    if len(task_rows) != len(prompts):
        raise RuntimeError("task and prompt counts differ")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_manifest = {
        "input_path": str(args.facts.resolve()),
        "sha256": source_sha,
        "record_count": len(facts),
    }
    atomic_write_json(args.output_dir / "source_manifest.json", source_manifest)
    atomic_write_json(
        args.output_dir / "conditions.json",
        [
            {"condition": condition.name, "k": condition.k, "system_prompt": condition.system_prompt}
            for condition in CONDITIONS
        ],
    )
    atomic_write_json(args.output_dir / "few_shots.json", demo_pairs)
    write_jsonl(
        args.output_dir / "facts_used.jsonl",
        [lightweight_fact(fact, "few_shot") for fact in few_shots]
        + [lightweight_fact(fact, "evaluation") for fact in evaluation_facts],
    )
    write_prompts(args.output_dir / "prompts.jsonl.gz", prompts)

    model = None
    device = None
    if not args.prompt_only:
        model, device = load_model(args)
    config = {
        "run_id": run_id,
        "mode": "prompt_only" if args.prompt_only else "evaluation",
        "model_identifier": args.model,
        "model_requested_revision": args.revision,
        "model_resolved_revision": (
            resolve_hf_revision(model, args.model, args.revision, "config.json")
            if model is not None
            else resolve_hf_revision(None, args.model, args.revision, "config.json")
        ),
        "tokenizer_identifier": tokenizer_id,
        "tokenizer_requested_revision": args.tokenizer_revision or args.revision,
        "tokenizer_resolved_revision": resolve_hf_revision(
            tokenizer,
            tokenizer_id,
            args.tokenizer_revision or args.revision,
            "tokenizer_config.json",
        ),
        "seed": args.seed,
        "addends_per_fact": args.addends_per_fact,
        "generation_settings": {"do_sample": False, "temperature": 0, "next_token_only": True},
        "system_prompts": {condition.name: condition.system_prompt for condition in CONDITIONS},
        "filler_construction_rule": FILLER_RULE,
        "chat_template_adjustment": (
            "render with the official Qwen template and enable_thinking=False; "
            "remove only its exact empty <think>\\n\\n</think>\\n\\n generation suffix"
        ),
        "source_fact_count": len(facts),
        "few_shot_fact_ids": [fact["fact_id"] for fact in few_shots],
        "evaluation_fact_count": len(evaluation_facts),
        "constructed_prompt_count": len(prompts),
        "full_default_result_count": full_default_row_count,
        "package_versions": package_versions(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": git_commit(),
        "local_files_only": args.local_files_only,
        "device": device,
    }
    atomic_write_json(args.output_dir / "run_config.json", config)

    if args.prompt_only:
        write_jsonl(args.output_dir / "results.jsonl", [])
        construction_summary = {
            "mode": "prompt_only",
            "source_fact_count": len(facts),
            "few_shot_count": len(few_shots),
            "evaluation_fact_count": len(evaluation_facts),
            "fact_addend_pair_count": len(pairs),
            "constructed_prompt_count": len(prompts),
            "full_default_result_count": full_default_row_count,
            "predictions_run": 0,
        }
        atomic_write_json(args.output_dir / "summary.json", construction_summary)
        write_summary_csv(args.output_dir / "summary.csv", [construction_summary])
        if args.inspect_fact_id:
            print_inspection(prompts)
        print(
            f"Constructed and validated {len(prompts)} prompts in {args.output_dir}; "
            "model weights were not loaded."
        )
        return 0

    assert model is not None and device is not None
    results = evaluate(model, tokenizer, device, task_rows, prompts, args.top_n)
    write_jsonl(args.output_dir / "results.jsonl", results)
    summary, csv_rows = summarize_results(results, args.seed, args.bootstrap_samples)
    summary.update(
        {
            "run_id": run_id,
            "source_fact_count": len(facts),
            "few_shot_count": len(few_shots),
            "evaluation_fact_count": len(evaluation_facts),
            "result_count": len(results),
        }
    )
    atomic_write_json(args.output_dir / "summary.json", summary)
    write_summary_csv(args.output_dir / "summary.csv", csv_rows)
    print(f"Wrote {len(results)} evaluation rows to {args.output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
