#!/usr/bin/env python3
"""Evaluate numeric fact knowledge through an SGLang /generate endpoint."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from filler.fact_eval import hf as outputs
from filler.fact_eval import protocol


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "model" / "DeepSeek-V4-Flash-0731-MoE-MXFP4-BF16"
DEFAULT_ENCODER = ROOT / "model" / "DeepSeek-V4-Flash-0731" / "encoding" / "encoding_dsv4.py"
DEFAULT_ENDPOINT = "http://127.0.0.1:30002/generate"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=protocol.DEFAULT_SOURCES)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--encoder", type=Path, default=DEFAULT_ENCODER)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "runs" / "deepseek-v4-flash" / "fact-knowledge"
    )
    parser.add_argument("--per-source", type=int, default=99)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-facts", type=int)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prompt-only", action="store_true")
    args = parser.parse_args(argv)
    if args.per_source < 1:
        parser.error("--per-source must be positive")
    if args.max_facts is not None and args.max_facts < 1:
        parser.error("--max-facts must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return args


def load_encoder(path: Path) -> Callable[..., str]:
    if not path.is_file():
        raise FileNotFoundError(f"DeepSeek encoder not found: {path}")
    spec = importlib.util.spec_from_file_location("deepseek_v4_encoding", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import DeepSeek encoder: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.encode_messages


def messages_for(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": protocol.DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def render_prompt(encode_messages: Callable[..., str], question: str) -> str:
    return encode_messages(messages_for(question), thinking_mode="chat")


def request_generation(endpoint: str, prompt: str, timeout: float) -> tuple[str, dict[str, Any]]:
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": protocol.MAX_NEW_TOKENS,
        },
    }
    request = urllib.request.Request(
        endpoint,
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
    if not isinstance(body, dict) or not isinstance(body.get("text"), str):
        raise ValueError(f"unexpected SGLang response: {body!r}")
    metadata = body.get("meta_info", {})
    return body["text"].strip(), metadata if isinstance(metadata, dict) else {}


def base_run_config(args: argparse.Namespace, selection: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_revision": outputs.git_revision(ROOT),
        "model_id": str(args.model.resolve()),
        "requested_model_revision": outputs.revision(str(args.model)),
        "tokenizer_id": str(args.model.resolve()),
        "requested_tokenizer_revision": outputs.revision(str(args.model)),
        "endpoint": args.endpoint,
        "backend": "sglang_generate_http",
        "selection": {
            "seed": args.seed,
            "per_source": args.per_source,
            "max_facts": args.max_facts,
            "selected_facts": len(selection),
            "source_counts": dict(Counter(item["kind"] for item in selection)),
        },
        "prompt": {
            "system_message": protocol.DEFAULT_SYSTEM_PROMPT,
            "messages_per_trial": ["system", "user"],
            "encoder": str(args.encoder.resolve()),
            "format": "deepseek_v4_official",
            "thinking_mode": "chat",
            "history": False,
            "retrieval_context": False,
            "few_shot_examples": 0,
        },
        "decoding": {
            "do_sample": False,
            "temperature": 0.0,
            "max_new_tokens": protocol.MAX_NEW_TOKENS,
        },
        "classification": {
            "trials_per_fact": protocol.TRIALS_PER_FACT,
            "known_minimum_correct": protocol.PASS_COUNT,
            "strict_pattern": protocol.ANSWER_RE.pattern,
        },
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    selection = protocol.load_selection(args.sources, args.per_source, args.seed)
    if args.max_facts is not None:
        selection = selection[: args.max_facts]
    total_trials = len(selection) * protocol.TRIALS_PER_FACT
    print(f"Selected {len(selection)} facts and {total_trials} trials.", flush=True)
    if args.dry_run:
        for fact in selection:
            print(f"{fact['fact_id']}: {fact['paraphrases']}")
        return

    encode_messages = load_encoder(args.encoder)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = base_run_config(args, selection)
    config_path = args.output_dir / "run_config.json"
    outputs.ensure_compatible_resume(config_path, config)
    protocol.atomic_write_json(args.output_dir / "source_manifest.json", outputs.source_manifest(args.sources))

    if args.prompt_only:
        prompts = [
            {
                "fact_id": fact["fact_id"],
                "trial": trial,
                "prompt_question": question,
                "rendered_prompt": render_prompt(encode_messages, question),
            }
            for fact in selection
            for trial, question in enumerate(fact["paraphrases"], 1)
        ]
        config["mode"] = "prompt_only"
        protocol.atomic_write_json(args.output_dir / "rendered_prompts.json", prompts)
        protocol.atomic_write_json(config_path, config)
        print(f"Wrote {len(prompts)} rendered prompts to {args.output_dir}.")
        return

    progress_path = args.output_dir / "fact_eval_progress.jsonl"
    progress = protocol.load_progress(progress_path, selection)
    checkpoint_path = args.output_dir / "fact_eval_checkpoint.json"
    previous_runtime = 0.0
    if checkpoint_path.exists():
        previous_runtime = json.loads(checkpoint_path.read_text()).get("total_runtime_seconds", 0.0)
    config["mode"] = "generation"
    protocol.atomic_write_json(config_path, config)
    started = time.perf_counter()
    outputs.save_outputs(args.output_dir, selection, progress, started, previous_runtime, args.seed)

    with progress_path.open("a", encoding="utf-8") as handle:
        for fact in selection:
            completed = {item["trial"] for item in progress[fact["fact_id"]]}
            for trial, question in enumerate(fact["paraphrases"], 1):
                if trial in completed:
                    continue
                rendered = render_prompt(encode_messages, question)
                request_started = time.perf_counter()
                response, metadata = request_generation(args.endpoint, rendered, args.timeout)
                elapsed = time.perf_counter() - request_started
                parsed = protocol.parse_answer(response)
                record = {
                    "fact_id": fact["fact_id"],
                    "question": fact["question"],
                    "prompt_question": question,
                    "rendered_prompt": rendered,
                    "expected": fact["answer"],
                    "trial": trial,
                    "decoding": "greedy",
                    "do_sample": False,
                    "temperature": 0.0,
                    "response": response,
                    "parsed_answer": parsed,
                    "correct": parsed == fact["answer"],
                    "generation_seconds": elapsed,
                    "server_metadata": metadata,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                progress[fact["fact_id"]].append(record)
                checkpoint = outputs.save_outputs(
                    args.output_dir, selection, progress, started, previous_runtime, args.seed
                )
                print(
                    f"{checkpoint['completed_trials']}/{checkpoint['total_trials']} "
                    f"{fact['fact_id']} trial {trial}: {response!r}",
                    flush=True,
                )

    outputs.validate_complete(selection, progress)
    checkpoint = outputs.save_outputs(
        args.output_dir, selection, progress, started, previous_runtime, args.seed
    )
    print(json.dumps(checkpoint, indent=2), flush=True)


if __name__ == "__main__":
    main()
