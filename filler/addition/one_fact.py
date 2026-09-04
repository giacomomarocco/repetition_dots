#!/usr/bin/env python3
"""Baseline-versus-filler one-fact addition through an SGLang endpoint."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "model" / "DeepSeek-V4-Flash-0731-MoE-MXFP4-BF16"
DEFAULT_ENCODER = ROOT / "model" / "DeepSeek-V4-Flash-0731" / "encoding" / "encoding_dsv4.py"
DEFAULT_ENDPOINT = "http://127.0.0.1:30002/generate"
DEFAULT_FACTS = ROOT / "runs" / "deepseek-v4-flash" / "fact-knowledge" / "known_facts.json"
ANSWER_RE = re.compile(r"^[+-]?\d+$")
SYSTEM_PROMPT = (
    "Solve each addition problem. After 'Answer:' respond with only the integer answer. "
    "No explanation, no words, no reasoning, just the number."
)
ONE_FACT_DEMONSTRATIONS = (
    ("How many sides does a triangle have?", 17, 20),
    ("How many days are in a week?", 24, 31),
    ("How many legs does a spider have?", 35, 43),
    ("How many letters are in the English alphabet?", 46, 72),
    ("How many planets are in the Solar System?", 61, 69),
)
RESUME_CONFIG_KEYS = (
    "model_id", "encoder", "endpoint", "source", "seed", "addends_per_fact",
    "filler_lengths", "filler_construction", "prompt_protocol",
    "demonstrations", "system_prompt", "decoding", "scoring",
    "strict_completion_pattern",
)


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def git_revision(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", type=Path, default=DEFAULT_FACTS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--encoder", type=Path, default=DEFAULT_ENCODER)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs" / "deepseek-v4-flash" / "one-fact-addition-smoke",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--addends-per-fact", type=int, default=1)
    parser.add_argument(
        "--filler-lengths",
        type=int,
        nargs="+",
        default=[0, 10, 20, 50, 100],
        help="dot counts; include 0 for the baseline (default: 0 10 20 50 100)",
    )
    parser.add_argument("--max-facts", type=int, default=1)
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
    if args.addends_per_fact < 1:
        parser.error("--addends-per-fact must be positive")
    if args.max_facts is not None and args.max_facts < 1:
        parser.error("--max-facts must be positive")
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


def digest(seed: int, namespace: str, value: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{namespace}\0{value}".encode()).digest()


def stable_fact_id(fact: dict[str, Any]) -> str:
    if isinstance(fact.get("source_file"), str) and fact.get("source_index") is not None:
        return f"{fact['source_file']}:{fact['source_index']}"
    question = fact.get("question")
    if not isinstance(question, str) or not question:
        raise ValueError("fact lacks a stable source identity and nonempty question")
    return "sha256:" + hashlib.sha256(question.encode()).hexdigest()


def load_facts(path: Path) -> tuple[list[dict[str, Any]], str]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, list):
        raise ValueError(f"{path}: expected a JSON array")
    facts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{path}[{index}]: expected an object")
        answer = item.get("answer")
        if isinstance(answer, bool) or not isinstance(answer, int):
            raise ValueError(f"{path}[{index}].answer: expected an integer")
        question = item.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{path}[{index}].question: expected a nonempty string")
        fact = dict(item)
        fact["fact_id"] = stable_fact_id(fact)
        if fact["fact_id"] in seen:
            raise ValueError(f"{path}[{index}]: duplicate fact ID {fact['fact_id']!r}")
        seen.add(fact["fact_id"])
        facts.append(fact)
    return sorted(facts, key=lambda row: row["fact_id"]), hashlib.sha256(raw).hexdigest()


def deterministic_addend(seed: int, fact_id: str, addend_index: int) -> int:
    return 10 + int.from_bytes(digest(seed, "addition-addend", f"{fact_id}\0{addend_index}")[:8], "big") % 90


def make_tasks(
    facts: Sequence[dict[str, Any]], seed: int, addends_per_fact: int, filler_lengths: Sequence[int]
) -> list[dict[str, Any]]:
    rows = []
    for fact in facts:
        for addend_index in range(addends_per_fact):
            addend = deterministic_addend(seed, fact["fact_id"], addend_index)
            target = fact["answer"] + addend
            pair_key = f"{seed}\0{fact['fact_id']}\0{addend_index}\0{addend}"
            pair_id = "pair-" + hashlib.sha256(pair_key.encode()).hexdigest()[:20]
            for k in filler_lengths:
                condition = "baseline" if k == 0 else f"dots_{k}"
                prompt_key = f"{pair_id}\0{condition}"
                rows.append({
                        "prompt_id": "prompt-" + hashlib.sha256(prompt_key.encode()).hexdigest()[:24],
                        "pair_id": pair_id,
                        "fact_id": fact["fact_id"],
                        "question": fact["question"],
                        "answer_value": fact["answer"],
                        "addend_index": addend_index,
                        "addend": addend,
                        "target": target,
                        "condition": condition,
                        "k": k,
                    })
    return rows


def render_question(task: dict[str, Any]) -> str:
    return one_fact_question(task["question"], task["addend"])


def one_fact_question(question: str, addend: int) -> str:
    return (
        "What is the numeric answer to the fact question below, plus "
        f"{addend}?\nFact question: {question}"
    )


def filler(k: int) -> str:
    return " ".join(["."] * k)


def answer_slot(k: int) -> str:
    """Return the target user-turn suffix immediately before assistant generation."""
    return (filler(k) + "\n" if k else "") + "Answer:"


def demonstration_messages(
    k: int,
    demonstrations: Sequence[tuple[str, int, int]] = ONE_FACT_DEMONSTRATIONS,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for question, addend, answer in demonstrations:
        messages.extend([
            {
                "role": "user",
                "content": one_fact_question(question, addend) + "\n" + answer_slot(k),
            },
            {"role": "assistant", "content": str(answer)},
        ])
    return messages


def load_encoder(path: Path) -> Callable[..., str]:
    spec = importlib.util.spec_from_file_location("deepseek_v4_addition_encoding", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import DeepSeek encoder: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.encode_messages


def render_prompt(encode_messages: Callable[..., str], task: dict[str, Any]) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *demonstration_messages(task["k"]),
        {"role": "user", "content": render_question(task) + "\n" + answer_slot(task["k"])},
    ]
    rendered = encode_messages(messages, thinking_mode="chat")
    if answer_slot(task["k"]) not in rendered:
        raise RuntimeError("DeepSeek encoder did not preserve the target user-turn answer slot")
    return rendered


def load_resume_results(
    progress_path: Path,
    prompts: Sequence[dict[str, Any]],
    previous_config: dict[str, Any] | None,
    current_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Load completed prompts only when the prior run is fully compatible."""
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


def split_target_prompt(prompt: str, k: int) -> tuple[str, str, str, str]:
    """Split a rendered prompt around target filler, Answer:, and assistant transition."""
    filler_text = filler(k) + "\n" if k else ""
    slot = filler_text + "Answer:"
    start = prompt.rfind(slot)
    if start < 0:
        raise ValueError("rendered prompt lacks the target user-turn answer slot")
    answer_start = start + len(filler_text)
    answer_end = answer_start + len("Answer:")
    return prompt[:start], filler_text, prompt[answer_start:answer_end], prompt[answer_end:]


def parse_answer(text: str) -> int | None:
    stripped = text.strip()
    return int(stripped) if ANSWER_RE.fullmatch(stripped) else None


def summarize(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        grouped[row["condition"]].append(row)
    conditions = {}
    for condition, rows in sorted(grouped.items()):
        conditions[condition] = {
            "count": len(rows),
            "correct": sum(bool(row["correct"]) for row in rows),
            "accuracy": sum(bool(row["correct"]) for row in rows) / len(rows),
            "mean_target_log_probability": sum(
                row["target_log_probability"] for row in rows
            ) / len(rows),
        }
    by_pair: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in results:
        by_pair[row["pair_id"]][row["condition"]] = row
    changes = {}
    for condition in sorted({row["condition"] for row in results} - {"baseline"}):
        pairs = [
            (values["baseline"], values[condition])
            for values in by_pair.values()
            if "baseline" in values and condition in values
        ]
        changes[condition] = {
            "count": len(pairs),
            "mean_target_log_probability_change_from_baseline": (
                sum(
                    new["target_log_probability"] - old["target_log_probability"]
                    for old, new in pairs
                )
                / len(pairs)
                if pairs
                else None
            ),
            "target_probability_increased_count": sum(
                new["target_log_probability"] > old["target_log_probability"]
                for old, new in pairs
            ),
            "target_top_rank_improved_count": sum(
                new["target_top_rank"] is not None
                and (
                    old["target_top_rank"] is None
                    or new["target_top_rank"] < old["target_top_rank"]
                )
                for old, new in pairs
            ),
        }
    return {
        "result_count": len(results),
        "conditions": conditions,
        "paired_changes_from_baseline": changes,
    }


def endpoint_url(generate_endpoint: str, path: str) -> str:
    return generate_endpoint.rsplit("/", 1)[0] + path


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"SGLang returned HTTP {error.code}: {detail}") from error
    if not isinstance(body, dict):
        raise ValueError(f"unexpected JSON response from {url}: {body!r}")
    return body


def tokenize(endpoint: str, text: str, timeout: float) -> list[int]:
    body = post_json(
        endpoint_url(endpoint, "/tokenize"),
        {"prompt": text, "add_special_tokens": False},
        timeout,
    )
    tokens = body.get("tokens")
    if not isinstance(tokens, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in tokens
    ):
        raise ValueError(f"unexpected SGLang tokenize response: {body!r}")
    return tokens


def validate_one_token_target(
    endpoint: str, prompt: str, target: int, timeout: float
) -> int:
    prompt_ids = tokenize(endpoint, prompt, timeout)
    combined_ids = tokenize(endpoint, prompt + str(target), timeout)
    if (
        combined_ids[: len(prompt_ids)] != prompt_ids
        or len(combined_ids) != len(prompt_ids) + 1
    ):
        raise RuntimeError(
            f"target {target} is not exactly one continuation token "
            f"(prompt tokens={len(prompt_ids)}, combined tokens={len(combined_ids)})"
        )
    return combined_ids[-1]


def logprob_tuple(item: Any) -> tuple[float, int]:
    if not isinstance(item, (list, tuple)) or len(item) < 2:
        raise ValueError(f"malformed SGLang logprob item: {item!r}")
    value, token_id = item[:2]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or isinstance(token_id, bool)
        or not isinstance(token_id, int)
    ):
        raise ValueError(f"malformed SGLang logprob item: {item!r}")
    return float(value), token_id


def extract_target_score(
    metadata: dict[str, Any], target_token_id: int, requested_top_n: int
) -> dict[str, Any]:
    requested = metadata.get("output_token_ids_logprobs")
    if not isinstance(requested, list) or not requested or not isinstance(requested[0], list):
        raise ValueError("SGLang response lacks first-position output_token_ids_logprobs")
    target_values = [
        value
        for value, token_id in map(logprob_tuple, requested[0])
        if token_id == target_token_id
    ]
    if len(target_values) != 1:
        raise ValueError(f"expected one requested logprob for target token {target_token_id}")
    target_logprob = target_values[0]

    top = metadata.get("output_top_logprobs")
    if not isinstance(top, list) or not top or not isinstance(top[0], list):
        raise ValueError("SGLang response lacks first-position output_top_logprobs")
    top_items = [logprob_tuple(item) for item in top[0]]
    top_rank = next(
        (
            index
            for index, (_, token_id) in enumerate(top_items, 1)
            if token_id == target_token_id
        ),
        None,
    )
    return {
        "target_token_id": target_token_id,
        "target_log_probability": target_logprob,
        "target_probability": math.exp(target_logprob),
        "target_top_rank": top_rank,
        "target_rank_lower_bound": None if top_rank is not None else requested_top_n + 1,
        "top_logprobs_returned": len(top_items),
        "top_logprobs": [
            {"rank": index, "log_probability": value, "token_id": token_id}
            for index, (value, token_id) in enumerate(top_items, 1)
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    facts, source_sha = load_facts(args.facts)
    if args.max_facts is not None:
        facts = facts[: args.max_facts]
    tasks = make_tasks(facts, args.seed, args.addends_per_fact, args.filler_lengths)
    encoder = load_encoder(args.encoder)
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
        "repository_revision": git_revision(ROOT),
        "mode": "prompt_only" if args.prompt_only else "generation",
        "model_id": str(args.model.resolve()),
        "encoder": str(args.encoder.resolve()),
        "endpoint": args.endpoint,
        "source": {"path": str(args.facts.resolve()), "sha256": source_sha, "selected_facts": len(facts)},
        "seed": args.seed,
        "addends_per_fact": args.addends_per_fact,
        "filler_lengths": args.filler_lengths,
        "prompt_protocol": (
            "five fixed user/assistant demonstrations; identical k fillers before Answer: "
            "in every demonstration and target user turn"
        ),
        "demonstrations": [
            {"question": question, "addend": addend, "answer": answer}
            for question, addend, answer in ONE_FACT_DEMONSTRATIONS
        ],
        "filler_construction": (
            "space-separated periods before Answer: in all five demonstration user turns "
            "and the target user turn"
        ),
        "system_prompt": SYSTEM_PROMPT,
        "decoding": {"temperature": 0, "max_new_tokens": args.max_new_tokens},
        "scoring": {
            "position": "first generated token after the Answer: prefix",
            "one_token_target_required": True,
            "top_logprobs_requested": args.top_logprobs,
            "rank_semantics": "exact within returned top tokens; otherwise a lower bound",
        },
        "strict_completion_pattern": ANSWER_RE.pattern,
    }
    progress_path = args.output_dir / "results_progress.jsonl"
    results = load_resume_results(progress_path, prompts, previous_config, config)
    atomic_write_json(config_path, config)
    atomic_write_json(args.output_dir / "prompts.json", prompts)
    if args.prompt_only:
        atomic_write_json(args.output_dir / "summary.json", {
            "mode": "prompt_only", "selected_facts": len(facts), "prompt_count": len(prompts)
        })
        print(f"Constructed {len(prompts)} prompts in {args.output_dir}.")
        return 0

    completed = {row["prompt_id"] for row in results}
    if completed:
        print(f"Resuming with {len(completed)}/{len(prompts)} prompts complete.", flush=True)
    interrupted = False
    with progress_path.open("a", encoding="utf-8") as progress_handle:
        try:
            for index, row in enumerate(prompts, 1):
                if row["prompt_id"] in completed:
                    continue
                started = time.perf_counter()
                target_token_id = validate_one_token_target(
                    args.endpoint, row["rendered_prompt"], row["target"], args.timeout
                )
                response, metadata = request_generation_with_limit(
                    args.endpoint,
                    row["rendered_prompt"],
                    args.timeout,
                    args.max_new_tokens,
                    target_token_id,
                    args.top_logprobs,
                )
                parsed = parse_answer(response)
                score = extract_target_score(metadata, target_token_id, args.top_logprobs)
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
        except KeyboardInterrupt:
            interrupted = True
    atomic_write_json(args.output_dir / "results.json", results)
    if results:
        summary = summarize(results)
    else:
        summary = {"result_count": 0, "conditions": {}, "paired_changes_from_baseline": {}}
    summary["complete"] = not interrupted and len(results) == len(prompts)
    summary["prompt_count"] = len(prompts)
    atomic_write_json(args.output_dir / "summary.json", summary)
    if interrupted:
        print(
            f"Interrupted cleanly after {len(results)}/{len(prompts)} prompts; "
            "rerun the same command to resume.",
            flush=True,
        )
        return 130
    print(f"Wrote {len(results)} results to {args.output_dir}.")
    return 0


def request_generation_with_limit(
    endpoint: str,
    prompt: str,
    timeout: float,
    max_new_tokens: int,
    target_token_id: int,
    top_logprobs_num: int,
) -> tuple[str, dict[str, Any]]:
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
        },
        "return_logprob": True,
        "top_logprobs_num": top_logprobs_num,
        "token_ids_logprob": [target_token_id],
        "return_text_in_logprobs": False,
    }
    body = post_json(endpoint, payload, timeout)
    if not isinstance(body.get("text"), str):
        raise ValueError(f"unexpected SGLang response: {body!r}")
    metadata = body.get("meta_info", {})
    return body["text"].strip(), metadata if isinstance(metadata, dict) else {}


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=__import__("sys").stderr)
        raise SystemExit(2)
