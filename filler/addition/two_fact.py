#!/usr/bin/env python3
"""Baseline-versus-filler two-fact addition through an SGLang endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from filler.addition import one_fact as shared


ROOT = Path(__file__).resolve().parents[2]
RESUME_CONFIG_KEYS = (
    "model_id", "encoder", "endpoint", "source", "seed", "pairing",
    "filler_lengths", "filler_construction", "prompt_protocol", "demonstrations",
    "system_prompt", "decoding",
    "scoring", "strict_completion_pattern",
)
TWO_FACT_DEMONSTRATIONS = (
    ("How many days are in a week?", "How many sides does a triangle have?", 10),
    ("How many planets are in the Solar System?", "How many sides does a hexagon have?", 14),
    ("How many months are in a year?", "How many legs does a spider have?", 20),
    ("How many letters are in the English alphabet?", "How many fingers does a typical person have?", 36),
    ("How many hours are in a day?", "How many cards are in a standard deck?", 76),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", type=Path, default=shared.DEFAULT_FACTS)
    parser.add_argument("--model", type=Path, default=shared.DEFAULT_MODEL)
    parser.add_argument("--encoder", type=Path, default=shared.DEFAULT_ENCODER)
    parser.add_argument("--endpoint", default=shared.DEFAULT_ENDPOINT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs" / "deepseek-v4-flash" / "two-fact-addition-smoke",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--filler-lengths", type=int, nargs="+", default=[0, 10, 20, 50, 100],
        help="dot counts; include 0 for the baseline (default: 0 10 20 50 100)",
    )
    parser.add_argument("--max-pairs", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--top-logprobs",
        type=int,
        default=20,
        help="number of highest-ranked next tokens to record (default: 20)",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--prompt-only", action="store_true")
    args = parser.parse_args(argv)
    if args.max_pairs is not None and args.max_pairs < 1:
        parser.error("--max-pairs must be positive")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if args.top_logprobs < 1:
        parser.error("--top-logprobs must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if not args.filler_lengths or any(k < 0 for k in args.filler_lengths):
        parser.error("--filler-lengths must contain nonnegative integers")
    if len(set(args.filler_lengths)) != len(args.filler_lengths):
        parser.error("--filler-lengths must not contain duplicates")
    return args


def pair_facts(facts: Sequence[dict[str, Any]], seed: int) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Deterministically order facts and pair each with its cyclic successor.

    This produces one pair per fact.  Every fact occurs in two pairs (once in
    each position), and no fact is paired with itself when at least two facts
    are available.
    """
    ordered = sorted(
        facts,
        key=lambda fact: (shared.digest(seed, "two-fact-order", fact["fact_id"]), fact["fact_id"]),
    )
    if len(ordered) < 2:
        return []
    return list(zip(ordered, ordered[1:] + ordered[:1]))


def make_tasks(
    pairs: Sequence[tuple[dict[str, Any], dict[str, Any]]], filler_lengths: Sequence[int]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fact_1, fact_2 in pairs:
        pair_key = f"{fact_1['fact_id']}\0{fact_2['fact_id']}"
        pair_id = "pair-" + hashlib.sha256(pair_key.encode()).hexdigest()[:20]
        target = fact_1["answer"] + fact_2["answer"]
        for k in filler_lengths:
            condition = "baseline" if k == 0 else f"dots_{k}"
            prompt_key = f"{pair_id}\0{condition}"
            rows.append({
                "prompt_id": "prompt-" + hashlib.sha256(prompt_key.encode()).hexdigest()[:24],
                "pair_id": pair_id,
                "fact_1_id": fact_1["fact_id"],
                "fact_1_question": fact_1["question"],
                "fact_1_answer": fact_1["answer"],
                "fact_2_id": fact_2["fact_id"],
                "fact_2_question": fact_2["question"],
                "fact_2_answer": fact_2["answer"],
                "target": target,
                "condition": condition,
                "k": k,
            })
    return rows


def render_question(task: dict[str, Any]) -> str:
    return two_fact_question(task["fact_1_question"], task["fact_2_question"])


def two_fact_question(question_1: str, question_2: str) -> str:
    return (
        "What is the sum of the numeric answers to the two fact questions below?\n"
        f"Fact question 1: {question_1}\n"
        f"Fact question 2: {question_2}"
    )


def demonstration_messages(k: int) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for question_1, question_2, answer in TWO_FACT_DEMONSTRATIONS:
        messages.extend([
            {
                "role": "user",
                "content": two_fact_question(question_1, question_2) + "\n" + shared.answer_slot(k),
            },
            {"role": "assistant", "content": str(answer)},
        ])
    return messages


def render_prompt(encode_messages, task: dict[str, Any]) -> str:
    messages = [
        {"role": "system", "content": shared.SYSTEM_PROMPT},
        *demonstration_messages(task["k"]),
        {"role": "user", "content": render_question(task) + "\n" + shared.answer_slot(task["k"])},
    ]
    rendered = encode_messages(messages, thinking_mode="chat")
    if shared.answer_slot(task["k"]) not in rendered:
        raise RuntimeError("DeepSeek encoder did not preserve the target user-turn answer slot")
    return rendered


def load_resume_results(
    progress_path: Path,
    prompts: Sequence[dict[str, Any]],
    previous_config: dict[str, Any] | None,
    current_config: dict[str, Any],
) -> list[dict[str, Any]]:
    if not progress_path.exists():
        return []
    if previous_config is None:
        raise ValueError(f"{progress_path} exists without run_config.json")
    for key in RESUME_CONFIG_KEYS:
        if previous_config.get(key) != current_config.get(key):
            raise ValueError(f"cannot resume: run configuration changed at {key!r}")
    expected = {row["prompt_id"]: row for row in prompts}
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    with progress_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("prompt_id"), str):
                raise ValueError(f"{progress_path}:{line_number}: malformed result")
            prompt_id = row["prompt_id"]
            if prompt_id in seen:
                raise ValueError(f"{progress_path}:{line_number}: duplicate {prompt_id}")
            if prompt_id not in expected:
                raise ValueError(f"{progress_path}:{line_number}: unexpected {prompt_id}")
            prompt = expected[prompt_id]
            for key in ("pair_id", "condition", "target"):
                if row.get(key) != prompt[key]:
                    raise ValueError(
                        f"{progress_path}:{line_number}: {key} does not match prompt"
                    )
            seen.add(prompt_id)
            results.append(row)
    return results


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    facts, source_sha = shared.load_facts(args.facts)
    pairs = pair_facts(facts, args.seed)
    if args.max_pairs is not None:
        pairs = pairs[:args.max_pairs]
    if not pairs:
        raise ValueError("at least two facts are required")
    tasks = make_tasks(pairs, args.filler_lengths)
    encoder = shared.load_encoder(args.encoder)
    prompts = [{**task, "rendered_prompt": render_prompt(encoder, task)} for task in tasks]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "run_config.json"
    previous_config = None
    if config_path.exists():
        previous_config = json.loads(config_path.read_text())
        if not isinstance(previous_config, dict):
            raise ValueError(f"{config_path}: expected a JSON object")
    config = {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_revision": shared.git_revision(ROOT),
        "mode": "prompt_only" if args.prompt_only else "generation",
        "model_id": str(args.model.resolve()),
        "encoder": str(args.encoder.resolve()),
        "endpoint": args.endpoint,
        "source": {"path": str(args.facts.resolve()), "sha256": source_sha, "selected_pairs": len(pairs)},
        "seed": args.seed,
        "pairing": (
            "seed-keyed ordering; each fact paired with its cyclic successor, "
            "yielding one pair per fact"
        ),
        "filler_lengths": args.filler_lengths,
        "prompt_protocol": (
            "five fixed user/assistant demonstrations; identical k fillers before Answer: "
            "in every demonstration and target user turn"
        ),
        "demonstrations": [
            {"question_1": q1, "question_2": q2, "answer": answer}
            for q1, q2, answer in TWO_FACT_DEMONSTRATIONS
        ],
        "filler_construction": (
            "space-separated periods before Answer: in all five demonstration user turns "
            "and the target user turn"
        ),
        "system_prompt": shared.SYSTEM_PROMPT,
        "decoding": {"temperature": 0, "max_new_tokens": args.max_new_tokens},
        "scoring": {
            "position": "first generated token after the Answer: prefix",
            "one_token_target_required": True,
            "top_logprobs_requested": args.top_logprobs,
            "rank_semantics": "exact within returned top tokens; otherwise a lower bound",
        },
        "strict_completion_pattern": shared.ANSWER_RE.pattern,
    }
    progress_path = args.output_dir / "results_progress.jsonl"
    results = load_resume_results(progress_path, prompts, previous_config, config)
    shared.atomic_write_json(config_path, config)
    shared.atomic_write_json(args.output_dir / "prompts.json", prompts)
    if args.prompt_only:
        shared.atomic_write_json(args.output_dir / "summary.json", {
            "mode": "prompt_only", "selected_pairs": len(pairs), "prompt_count": len(prompts)
        })
        print(f"Constructed {len(prompts)} prompts in {args.output_dir}.")
        return 0

    completed = {row["prompt_id"] for row in results}
    if completed:
        print(f"Resuming with {len(completed)}/{len(prompts)} prompts complete.", flush=True)
    with progress_path.open("a", encoding="utf-8") as progress_handle:
        for index, row in enumerate(prompts, 1):
            if row["prompt_id"] in completed:
                continue
            started = time.perf_counter()
            target_token_id = shared.validate_one_token_target(
                args.endpoint, row["rendered_prompt"], row["target"], args.timeout
            )
            response, metadata = shared.request_generation_with_limit(
                args.endpoint,
                row["rendered_prompt"],
                args.timeout,
                args.max_new_tokens,
                target_token_id,
                args.top_logprobs,
            )
            parsed = shared.parse_answer(response)
            score = shared.extract_target_score(metadata, target_token_id, args.top_logprobs)
            result = {
                **row,
                **score,
                "response": response,
                "parsed_answer": parsed,
                "correct": parsed == row["target"],
                "generation_seconds": time.perf_counter() - started,
                "server_metadata": metadata,
            }
            progress_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            progress_handle.flush()
            os.fsync(progress_handle.fileno())
            results.append(result)
            rank = score["target_top_rank"] or f">={score['target_rank_lower_bound']}"
            print(
                f"[{index}/{len(prompts)}] {row['condition']} target={row['target']} "
                f"response={response!r} logp={score['target_log_probability']:.6f} "
                f"rank={rank}",
                flush=True,
            )
    shared.atomic_write_json(args.output_dir / "results.json", results)
    shared.atomic_write_json(args.output_dir / "summary.json", shared.summarize(results))
    print(f"Wrote {len(results)} results to {args.output_dir}.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=__import__("sys").stderr)
        raise SystemExit(2)
