#!/usr/bin/env python3
"""Baseline-versus-filler one-fact addition through an SGLang endpoint."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import evaluate_facts as fact_protocol
import evaluate_facts_hf as fact_outputs
from evaluate_facts_sglang import DEFAULT_ENCODER, DEFAULT_ENDPOINT, DEFAULT_MODEL


ROOT = Path(__file__).resolve().parent
DEFAULT_FACTS = ROOT / "runs" / "deepseek-v4-flash" / "fact-knowledge" / "known_facts.json"
ANSWER_RE = re.compile(r"^[+-]?\d+$")
SYSTEM_PROMPT = (
    "Solve the addition problem. The assistant response has already been started and ends "
    "with 'Answer: '. Continue it with only the integer answer, with no other text."
)


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
    return (
        "What is the numeric answer to the fact question below, plus "
        f"{task['addend']}?\nFact question: {task['question']}"
    )


def filler(k: int) -> str:
    return " ".join(["."] * k)


def assistant_prefix(k: int) -> str:
    return (filler(k) + "\n" if k else "") + "Answer: "


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
        {"role": "user", "content": render_question(task)},
        {"role": "assistant", "content": assistant_prefix(task["k"]), "wo_eos": True},
    ]
    rendered = encode_messages(messages, thinking_mode="chat")
    prefix = assistant_prefix(task["k"])
    if not rendered.endswith(prefix):
        raise RuntimeError("DeepSeek encoder did not preserve the assistant prefix at prompt end")
    return rendered


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
    config = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_revision": fact_outputs.git_revision(ROOT),
        "mode": "prompt_only" if args.prompt_only else "generation",
        "model_id": str(args.model.resolve()),
        "encoder": str(args.encoder.resolve()),
        "endpoint": args.endpoint,
        "source": {"path": str(args.facts.resolve()), "sha256": source_sha, "selected_facts": len(facts)},
        "seed": args.seed,
        "addends_per_fact": args.addends_per_fact,
        "filler_lengths": args.filler_lengths,
        "filler_construction": "space-separated periods in a forced assistant prefix",
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
    fact_protocol.atomic_write_json(args.output_dir / "run_config.json", config)
    fact_protocol.atomic_write_json(args.output_dir / "prompts.json", prompts)
    if args.prompt_only:
        fact_protocol.atomic_write_json(args.output_dir / "summary.json", {
            "mode": "prompt_only", "selected_facts": len(facts), "prompt_count": len(prompts)
        })
        print(f"Constructed {len(prompts)} prompts in {args.output_dir}.")
        return 0

    results = []
    for index, row in enumerate(prompts, 1):
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
        results.append(result)
        rank = score["target_top_rank"] or f">={score['target_rank_lower_bound']}"
        print(
            f"[{index}/{len(prompts)}] {row['condition']} target={row['target']} "
            f"response={response!r} logp={score['target_log_probability']:.6f} "
            f"rank={rank}",
            flush=True,
        )
    fact_protocol.atomic_write_json(args.output_dir / "results.json", results)
    fact_protocol.atomic_write_json(args.output_dir / "summary.json", summarize(results))
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
