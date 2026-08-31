#!/usr/bin/env python3
"""Evaluate numeric fact knowledge with a local Hugging Face model."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

import evaluate_facts as protocol
from model_adapter import (
    configured_revision,
    load_model,
    load_tokenizer,
    local_revision,
    render_prompt,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "Qwen3.6-27B"
DEFAULT_CACHE = ROOT / ".hf-cache"


def sanitize_model_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", Path(value).name.lower()).strip("-")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=protocol.DEFAULT_SOURCES)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--tokenizer")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    parser.add_argument("--per-source", type=int, default=99)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-facts", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prompt-only", action="store_true")
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args(argv)
    if args.per_source < 1:
        parser.error("--per-source must be positive")
    if args.max_facts is not None and args.max_facts < 1:
        parser.error("--max-facts must be positive")
    if args.output_dir is None:
        args.output_dir = (
            ROOT / "runs" / sanitize_model_name(args.model) / "fact-knowledge"
        )
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {"python": platform.python_version()}
    for package in ("torch", "transformers", "accelerate", "safetensors"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def source_manifest(sources: Path) -> dict[str, Any]:
    entries = []
    for filename in protocol.SOURCE_FILES:
        path = (sources / filename).resolve()
        records = json.loads(path.read_text(encoding="utf-8"))
        entries.append(
            {
                "source_file": filename,
                "resolved_path": str(path),
                "record_count": len(records),
                "sha256": sha256(path),
            }
        )
    return {"sources": entries}


def revision(value: str) -> str | None:
    return local_revision(value) or configured_revision(value)


def base_run_config(args: argparse.Namespace, selected: list[dict[str, Any]]) -> dict[str, Any]:
    tokenizer = args.tokenizer or args.model
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_revision": git_revision(ROOT),
        "model_id": args.model,
        "requested_model_revision": revision(args.model),
        "tokenizer_id": tokenizer,
        "requested_tokenizer_revision": revision(tokenizer),
        "device_requested": args.device,
        "dtype_requested": args.dtype,
        "cache_dir": str(args.cache_dir.resolve()),
        "local_files_only": args.local_files_only,
        "package_versions": package_versions(),
        "selection": {
            "seed": args.seed,
            "per_source": args.per_source,
            "max_facts": args.max_facts,
            "selected_facts": len(selected),
            "source_counts": dict(Counter(item["kind"] for item in selected)),
        },
        "prompt": {
            "system_message": protocol.DEFAULT_SYSTEM_PROMPT,
            "messages_per_trial": ["system", "user"],
            "official_chat_template_when_available": True,
            "base_model_fallback": (
                "{system_message}\\n\\nQuestion: {question}\\nAnswer:"
            ),
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


def resume_signature(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: config[key]
        for key in (
            "model_id",
            "requested_model_revision",
            "tokenizer_id",
            "requested_tokenizer_revision",
            "selection",
            "prompt",
            "decoding",
            "classification",
        )
    }


def ensure_compatible_resume(path: Path, config: dict[str, Any]) -> None:
    if not path.exists():
        return
    existing = json.loads(path.read_text(encoding="utf-8"))
    if resume_signature(existing) != resume_signature(config):
        raise ValueError(
            f"existing {path} is incompatible with this run; use a new output directory"
        )


def completed_results(
    selection: list[dict[str, Any]],
    progress: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    results = []
    for fact in selection:
        trials = progress[fact["fact_id"]]
        if len(trials) != protocol.TRIALS_PER_FACT:
            continue
        correct = [bool(record["correct"]) for record in trials]
        record = {key: value for key, value in fact.items() if key != "fact_id"}
        record.update(
            {
                "trial_responses": [item["response"] for item in trials],
                "trial_predictions": [item["parsed_answer"] for item in trials],
                "trial_correctness": correct,
                "correct_count": sum(correct),
                "correct_fraction": sum(correct) / protocol.TRIALS_PER_FACT,
                "trial_count": protocol.TRIALS_PER_FACT,
            }
        )
        results.append(record)
    return results


def save_outputs(
    output_dir: Path,
    selection: list[dict[str, Any]],
    progress: dict[str, list[dict[str, Any]]],
    started: float,
    previous_runtime: float,
    selection_seed: int,
) -> dict[str, Any]:
    results = completed_results(selection, progress)
    known = [item for item in results if item["correct_count"] >= protocol.PASS_COUNT]
    unknown = [item for item in results if item["correct_count"] < protocol.PASS_COUNT]
    completed_trials = sum(len(items) for items in progress.values())
    total_trials = len(selection) * protocol.TRIALS_PER_FACT
    runtime = previous_runtime + time.perf_counter() - started
    malformed = sum(
        item.get("parsed_answer") is None
        for items in progress.values()
        for item in items
    )
    checkpoint = {
        "status": "complete" if completed_trials == total_trials else "running",
        "selected_facts": len(selection),
        "completed_facts": len(results),
        "trials_per_fact": protocol.TRIALS_PER_FACT,
        "completed_trials": completed_trials,
        "total_trials": total_trials,
        "known_so_far": len(known),
        "unknown_so_far": len(unknown),
        "malformed_responses": malformed,
        "total_runtime_seconds": runtime,
        "selection_seed": selection_seed,
    }
    protocol.atomic_write_json(output_dir / "known_facts.json", known)
    protocol.atomic_write_json(output_dir / "unknown_facts.json", unknown)
    protocol.atomic_write_json(output_dir / "fact_eval_checkpoint.json", checkpoint)
    return checkpoint


def messages_for(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": protocol.DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def generate(model_bundle: Any, question: str) -> tuple[str, str, float]:
    rendered, inputs = render_prompt(model_bundle.tokenizer, messages_for(question))
    inputs = inputs.to(model_bundle.input_device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model_bundle.model.generate(
            **inputs,
            max_new_tokens=protocol.MAX_NEW_TOKENS,
            do_sample=False,
        )
    if model_bundle.metadata["device"] == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    input_length = inputs["input_ids"].shape[1]
    response = model_bundle.tokenizer.decode(
        output[0, input_length:], skip_special_tokens=True
    ).strip()
    return rendered, response, elapsed


def validate_complete(
    selection: list[dict[str, Any]], progress: dict[str, list[dict[str, Any]]]
) -> None:
    if any(len(progress[item["fact_id"]]) != protocol.TRIALS_PER_FACT for item in selection):
        raise RuntimeError("final validation failed: a selected fact does not have five trials")
    results = completed_results(selection, progress)
    known_ids = {
        (item["source_file"], item["source_index"])
        for item in results
        if item["correct_count"] >= protocol.PASS_COUNT
    }
    unknown_ids = {
        (item["source_file"], item["source_index"])
        for item in results
        if item["correct_count"] < protocol.PASS_COUNT
    }
    if known_ids & unknown_ids or len(known_ids | unknown_ids) != len(selection):
        raise RuntimeError("final validation failed: classifications are not exclusive")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    selection = protocol.load_selection(args.sources, args.per_source, args.seed)
    if args.max_facts is not None:
        selection = selection[: args.max_facts]
    print(f"Selected {len(selection)} facts and {len(selection) * 5} trials.", flush=True)
    if args.dry_run:
        for fact in selection:
            print(f"{fact['fact_id']}: {fact['paraphrases']}")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = source_manifest(args.sources)
    config = base_run_config(args, selection)
    config_path = args.output_dir / "run_config.json"
    ensure_compatible_resume(config_path, config)
    protocol.atomic_write_json(args.output_dir / "source_manifest.json", manifest)

    if args.prompt_only:
        tokenizer = load_tokenizer(
            args.model, args.tokenizer, args.cache_dir, args.local_files_only
        )
        prompts = []
        for fact in selection:
            for trial, question in enumerate(fact["paraphrases"], 1):
                rendered, encoded = render_prompt(tokenizer, messages_for(question))
                prompts.append(
                    {
                        "fact_id": fact["fact_id"],
                        "trial": trial,
                        "prompt_question": question,
                        "rendered_prompt": rendered,
                        "input_tokens": encoded["input_ids"].shape[-1],
                    }
                )
        protocol.atomic_write_json(args.output_dir / "rendered_prompts.json", prompts)
        config["mode"] = "prompt_only"
        protocol.atomic_write_json(config_path, config)
        print(f"Wrote {len(prompts)} rendered prompts to {args.output_dir}.")
        return

    progress_path = args.output_dir / "fact_eval_progress.jsonl"
    progress = protocol.load_progress(progress_path, selection)
    checkpoint_path = args.output_dir / "fact_eval_checkpoint.json"
    previous_runtime = 0.0
    if checkpoint_path.exists():
        previous_runtime = json.loads(checkpoint_path.read_text()).get(
            "total_runtime_seconds", 0.0
        )
    bundle = load_model(
        args.model,
        args.tokenizer,
        args.device,
        args.dtype,
        args.cache_dir,
        args.local_files_only,
    )
    config.update(bundle.metadata)
    config["mode"] = "generation"
    protocol.atomic_write_json(config_path, config)
    started = time.perf_counter()
    save_outputs(args.output_dir, selection, progress, started, previous_runtime, args.seed)

    with progress_path.open("a", encoding="utf-8") as handle:
        for fact in selection:
            completed = {item["trial"] for item in progress[fact["fact_id"]]}
            for trial, question in enumerate(fact["paraphrases"], 1):
                if trial in completed:
                    continue
                rendered, response, elapsed = generate(bundle, question)
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
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                progress[fact["fact_id"]].append(record)
                checkpoint = save_outputs(
                    args.output_dir, selection, progress, started, previous_runtime, args.seed
                )
                print(
                    f"{checkpoint['completed_trials']}/{checkpoint['total_trials']} "
                    f"{fact['fact_id']} trial {trial}: {response!r}",
                    flush=True,
                )

    validate_complete(selection, progress)
    checkpoint = save_outputs(
        args.output_dir, selection, progress, started, previous_runtime, args.seed
    )
    print(json.dumps(checkpoint, indent=2), flush=True)


if __name__ == "__main__":
    main()
