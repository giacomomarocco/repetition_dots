#!/usr/bin/env python3
"""Evaluate a time-bounded, balanced subset of integer fact questions."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

# The checkpoint is complete, so keep this run fully offline. This also avoids
# Hugging Face network retries when the model is loaded from the local cache.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch

from run_qwen import DEFAULT_SYSTEM_PROMPT, load_model


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCES = ROOT / "compose_facts"
SOURCE_FILES = ("age_facts.json", "atomic_facts.json", "static_facts.json")
TRIALS_PER_FACT = 5
MAX_NEW_TOKENS = 8
PASS_COUNT = 4
ANSWER_RE = re.compile(r"Answer:\s*(-?\d+)\s*")

STATIC_PARAPHRASES: dict[int, list[str]] = {
    1: [
        "What is the number of cantos in Dante's Inferno?",
        "How many cantos are in Dante's Inferno?",
        "How many cantos does Dante's Inferno contain?",
        "Give the total number of cantos in Dante's Inferno.",
        "What is the canto count for Dante's Inferno?",
    ],
    3: [
        "What is the number of essays in The Federalist Papers?",
        "How many essays are in The Federalist Papers?",
        "How many essays does The Federalist Papers contain?",
        "Give the total number of essays in The Federalist Papers.",
        "What is the essay count for The Federalist Papers?",
    ],
    12: [
        "What is the number of amino acids used in protein synthesis?",
        "How many amino acids are used in protein synthesis?",
        "Protein synthesis uses how many amino acids?",
        "Give the standard number of amino acids used to synthesize proteins.",
        "What is the count of amino acids used in making proteins?",
    ],
    14: [
        "What is the number of fables attributed to Aesop?",
        "How many fables are attributed to Aesop?",
        "Aesop is traditionally credited with how many fables?",
        "Give the total count of fables attributed to Aesop.",
        "What is the number of fables traditionally credited to Aesop?",
    ],
    20: [
        "What is the number of moons Saturn has?",
        "How many moons does Saturn have?",
        "What is the total count of Saturn's moons?",
        "Give the number of natural satellites orbiting Saturn.",
        "How many natural satellites belong to Saturn?",
    ],
    22: [
        "What is the number of piano sonatas Schubert composed?",
        "How many piano sonatas did Schubert compose?",
        "What is the total count of Schubert's piano sonatas?",
        "Give the number of piano sonatas composed by Schubert.",
        "Schubert composed how many piano sonatas?",
    ],
    28: [
        "What is the number of fugues in Bach's Well-Tempered Clavier?",
        "How many fugues are in Bach's Well-Tempered Clavier?",
        "How many fugues does Bach's Well-Tempered Clavier contain?",
        "Give the total number of fugues in The Well-Tempered Clavier by Bach.",
        "What is the fugue count in Bach's Well-Tempered Clavier?",
    ],
    37: [
        "What is the number of Goldberg Variations Bach composed?",
        "How many Goldberg Variations did Bach compose?",
        "What is the total number of Bach's Goldberg Variations?",
        "Give the count of Goldberg Variations composed by Bach.",
        "Bach composed how many Goldberg Variations?",
    ],
    38: [
        "What is the number of storeys in the Leaning Tower of Pisa?",
        "How many storeys does the Leaning Tower of Pisa have?",
        "What is the storey count of the Leaning Tower of Pisa?",
        "Give the number of storeys in the Leaning Tower of Pisa.",
        "The Leaning Tower of Pisa consists of how many storeys?",
    ],
    48: [
        "What is the number of floors in the Empire State Building?",
        "How many floors does the Empire State Building have?",
        "What is the floor count of the Empire State Building?",
        "Give the number of floors in the Empire State Building.",
        "The Empire State Building has how many floors?",
    ],
    52: [
        "What is the number of nocturnes Chopin composed?",
        "How many nocturnes did Chopin compose?",
        "What is the total count of Chopin's nocturnes?",
        "Give the number of nocturnes composed by Chopin.",
        "Chopin composed how many nocturnes?",
    ],
    65: [
        "What is the number of books in Homer's Iliad?",
        "How many books are in Homer's Iliad?",
        "How many books does Homer's Iliad contain?",
        "Give the total number of books in the Iliad.",
        "What is the book count of Homer's Iliad?",
    ],
    66: [
        "What is the number of books in Homer's Odyssey?",
        "How many books are in Homer's Odyssey?",
        "How many books does Homer's Odyssey contain?",
        "Give the total number of books in the Odyssey.",
        "What is the book count of Homer's Odyssey?",
    ],
    69: [
        "What is the number of novels Jane Austen completed?",
        "How many novels did Jane Austen complete?",
        "What is the total count of novels completed by Jane Austen?",
        "Give the number of novels Jane Austen finished.",
        "Jane Austen completed how many novels?",
    ],
    72: [
        "What is the number of dots on a standard die?",
        "How many dots are on a standard die in total?",
        "What is the total dot count across all faces of a standard die?",
        "Give the number of pips on all sides of a standard die combined.",
        "A standard six-sided die has how many dots altogether?",
    ],
    88: [
        "What is the number of voivodeships in Poland?",
        "How many voivodeships are in Poland?",
        "Into how many voivodeships is Poland divided?",
        "Give the total number of Polish voivodeships.",
        "What is Poland's count of voivodeships?",
    ],
    89: [
        "What is the number of departments in France?",
        "How many departments are in France?",
        "Into how many departments is France divided?",
        "Give the total number of French departments.",
        "What is France's count of departments?",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument(
        "--per-source",
        type=int,
        default=99,
        help="questions sampled from each input file (default: 99; 297 total)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate inputs and print the selection without loading Qwen",
    )
    args = parser.parse_args()
    if args.per_source < 1:
        parser.error("--per-source must be positive")
    return args


def make_paraphrases(fact: dict[str, Any]) -> list[str]:
    question = fact["question"]
    kind = fact["kind"]
    if kind == "age":
        match = re.fullmatch(r"At what age did (.+) die\?", question)
        if not match:
            raise ValueError(f"cannot paraphrase age question: {question}")
        person = match.group(1)
        variants = [
            question,
            f"How old was {person} when they died?",
            f"What was {person}'s age at death?",
            f"How many years old was {person} at the time of their death?",
            f"At the time of death, what age was {person}?",
        ]
    elif kind == "atomic":
        match = re.fullmatch(r"What is the atomic number of (.+)\?", question)
        if not match:
            raise ValueError(f"cannot paraphrase atomic question: {question}")
        element = match.group(1)
        variants = [
            question,
            f"Which atomic number does {element} have?",
            f"What number on the periodic table corresponds to {element}?",
            f"How many protons are in an atom of {element}?",
            f"Give the atomic number for {element}.",
        ]
    else:
        variants = STATIC_PARAPHRASES.get(fact["source_index"])
        if variants is None:
            match = re.fullmatch(r"What is the number of (.+)\?", question)
            if not match:
                raise ValueError(f"cannot paraphrase static question: {question}")
            subject = match.group(1)
            variants = [
                question,
                f"How many {subject}?",
                f"Give the number of {subject}.",
                f"State the total count of {subject}.",
                f"What is the total number of {subject}?",
            ]

    if len(variants) != TRIALS_PER_FACT or len(set(variants)) != TRIALS_PER_FACT:
        raise ValueError(f"expected five unique paraphrases for: {question}")
    return variants


def load_selection(sources: Path, per_source: int, seed: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for source_number, filename in enumerate(SOURCE_FILES):
        path = sources / filename
        facts = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(facts, list):
            raise ValueError(f"{path} must contain a JSON list")
        if per_source > len(facts):
            raise ValueError(
                f"requested {per_source} questions from {path}, which has {len(facts)}"
            )

        # Give every source its own deterministic random stream so changing the
        # size of one source cannot perturb the samples from the other sources.
        rng = random.Random(seed + source_number)
        indices = sorted(rng.sample(range(len(facts)), per_source))
        kind = filename.removesuffix("_facts.json")
        for source_index in indices:
            fact = facts[source_index]
            if not isinstance(fact, dict):
                raise ValueError(f"{path}[{source_index}] must be an object")
            question = fact.get("question")
            answer = fact.get("answer")
            if not isinstance(question, str):
                raise ValueError(f"{path}[{source_index}].question must be a string")
            if not isinstance(answer, int) or isinstance(answer, bool):
                raise ValueError(f"{path}[{source_index}].answer must be an integer")
            selected_fact = {
                **fact,
                "kind": kind,
                "source_file": filename,
                "source_index": source_index,
                "fact_id": f"{kind}:{source_index}",
            }
            selected_fact["paraphrases"] = make_paraphrases(selected_fact)
            selected.append(selected_fact)
    return selected


def parse_answer(response: str) -> int | None:
    match = ANSWER_RE.fullmatch(response.strip())
    return int(match.group(1)) if match else None


def load_progress(path: Path, selection: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    expected = {fact["fact_id"]: fact for fact in selection}
    progress: dict[str, list[dict[str, Any]]] = {fact_id: [] for fact_id in expected}
    if not path.exists():
        return progress

    seen: set[tuple[str, int]] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            fact_id = record.get("fact_id")
            trial = record.get("trial")
            key = (fact_id, trial)
            if fact_id not in expected:
                continue
            if not isinstance(trial, int) or not 1 <= trial <= TRIALS_PER_FACT:
                raise ValueError(f"invalid trial in {path}:{line_number}")
            if key in seen:
                raise ValueError(f"duplicate {fact_id} trial {trial} in {path}")
            fact = expected[fact_id]
            if (
                record.get("question") != fact["question"]
                or record.get("prompt_question") != fact["paraphrases"][trial - 1]
                or record.get("expected") != fact["answer"]
            ):
                raise ValueError(f"fact mismatch for {fact_id} in {path}:{line_number}")
            seen.add(key)
            progress[fact_id].append(record)

    for records in progress.values():
        records.sort(key=lambda item: item["trial"])
    return progress


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def completed_results(
    selection: list[dict[str, Any]],
    progress: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    results = []
    for fact in selection:
        trials = progress[fact["fact_id"]]
        if len(trials) != TRIALS_PER_FACT:
            continue
        correct_count = sum(record["correct"] for record in trials)
        result = {
            key: value for key, value in fact.items() if key != "fact_id"
        }
        result.update(
            {
                "correct_fraction": correct_count / TRIALS_PER_FACT,
                "correct_count": correct_count,
                "trial_count": TRIALS_PER_FACT,
                "trial_predictions": [record["parsed_answer"] for record in trials],
                "trial_responses": [record["response"] for record in trials],
            }
        )
        results.append(result)
    return results


def save_outputs(
    output_dir: Path,
    selection: list[dict[str, Any]],
    progress: dict[str, list[dict[str, Any]]],
    run_started: float,
) -> dict[str, Any]:
    results = completed_results(selection, progress)
    known = [item for item in results if item["correct_count"] >= PASS_COUNT]
    unknown = [item for item in results if item["correct_count"] < PASS_COUNT]
    completed_trials = sum(len(records) for records in progress.values())
    total_trials = len(selection) * TRIALS_PER_FACT
    elapsed = time.perf_counter() - run_started
    seconds_per_trial = elapsed / completed_trials if completed_trials else None
    remaining = total_trials - completed_trials
    eta_seconds = seconds_per_trial * remaining if seconds_per_trial else None
    checkpoint = {
        "status": "complete" if completed_trials == total_trials else "running",
        "selected_facts": len(selection),
        "completed_facts": len(results),
        "trials_per_fact": TRIALS_PER_FACT,
        "completed_trials": completed_trials,
        "total_trials": total_trials,
        "progress_fraction": completed_trials / total_trials,
        "known_so_far": len(known),
        "unknown_so_far": len(unknown),
        "elapsed_seconds_this_run": elapsed,
        "seconds_per_trial_this_run": seconds_per_trial,
        "eta_seconds": eta_seconds,
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "thinking_enabled": False,
        "filler_tokens": 0,
        "history": False,
        "decoding": "greedy for all five trials",
        "temperature": 0.0,
        "paraphrases_per_fact": TRIALS_PER_FACT,
        "pass_count": PASS_COUNT,
        "pass_fraction": PASS_COUNT / TRIALS_PER_FACT,
        "selection_seed": 42,
        "source_counts": dict(Counter(fact["kind"] for fact in selection)),
    }
    atomic_write_json(output_dir / "known_facts.json", known)
    atomic_write_json(output_dir / "unknown_facts.json", unknown)
    atomic_write_json(output_dir / "fact_eval_checkpoint.json", checkpoint)
    return checkpoint


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "measuring"
    minutes, second = divmod(max(0, round(seconds)), 60)
    hours, minute = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minute:02d}m {second:02d}s"
    return f"{minute}m {second:02d}s"


def generate_response(
    model: Any,
    tokenizer: Any,
    device: str,
    question: str,
) -> tuple[str, float]:
    # Exactly two messages each time: the fixed system prompt and one question.
    # No prior user or assistant messages and no filler are ever included.
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(device)
    generation_args: dict[str, Any] = {
        **inputs,
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
    }

    if device == "mps":
        torch.mps.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**generation_args)
    if device == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - started
    input_length = inputs["input_ids"].shape[1]
    response = tokenizer.decode(output[0, input_length:], skip_special_tokens=True)
    return response.strip(), elapsed


def main() -> None:
    args = parse_args()
    selection = load_selection(args.sources, args.per_source, args.seed)
    source_counts = Counter(fact["kind"] for fact in selection)
    print(
        f"Selected {len(selection)} facts: "
        + ", ".join(f"{kind}={count}" for kind, count in source_counts.items()),
        flush=True,
    )
    print(
        f"Plan: {TRIALS_PER_FACT} trials/fact, {len(selection) * TRIALS_PER_FACT} "
        "independent paraphrases; all five use greedy temperature-0 decoding; "
        f"known threshold {PASS_COUNT}/{TRIALS_PER_FACT}.",
        flush=True,
    )
    if args.dry_run:
        for fact in selection:
            print(f"{fact['fact_id']}: {fact['question']} -> {fact['answer']}")
            for trial, paraphrase in enumerate(fact["paraphrases"], start=1):
                print(f"  {trial}. {paraphrase}")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / "fact_eval_progress.jsonl"
    progress = load_progress(progress_path, selection)
    existing_trials = sum(len(records) for records in progress.values())
    if existing_trials:
        print(f"Resuming from {existing_trials} completed trials in {progress_path}.", flush=True)

    model, tokenizer, device = load_model()
    if device != "mps":
        raise RuntimeError("this time-bounded evaluation requires MPS, but MPS is unavailable")

    run_started = time.perf_counter()
    checkpoint = save_outputs(args.output_dir, selection, progress, run_started)
    print(
        f"CHECKPOINT {checkpoint['completed_trials']}/{checkpoint['total_trials']} trials "
        f"({checkpoint['progress_fraction']:.1%}); ETA {format_duration(checkpoint['eta_seconds'])}",
        flush=True,
    )

    with progress_path.open("a", encoding="utf-8") as progress_file:
        for fact_number, fact in enumerate(selection, start=1):
            records = progress[fact["fact_id"]]
            completed_trial_numbers = {record["trial"] for record in records}
            for trial in range(1, TRIALS_PER_FACT + 1):
                if trial in completed_trial_numbers:
                    continue
                prompt_question = fact["paraphrases"][trial - 1]
                response, generation_seconds = generate_response(
                    model,
                    tokenizer,
                    device,
                    prompt_question,
                )
                parsed = parse_answer(response)
                record = {
                    "fact_id": fact["fact_id"],
                    "question": fact["question"],
                    "prompt_question": prompt_question,
                    "expected": fact["answer"],
                    "trial": trial,
                    "decoding": "greedy",
                    "temperature": 0.0,
                    "response": response,
                    "parsed_answer": parsed,
                    "correct": parsed == fact["answer"],
                    "generation_seconds": generation_seconds,
                }
                progress_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                progress_file.flush()
                records.append(record)

                checkpoint = save_outputs(args.output_dir, selection, progress, run_started)
                print(
                    f"TRIAL {checkpoint['completed_trials']:>3}/{checkpoint['total_trials']} "
                    f"fact {fact_number:>2}/{len(selection)} {fact['fact_id']} "
                    f"run {trial}/{TRIALS_PER_FACT}: {response!r} "
                    f"({'correct' if record['correct'] else 'wrong'}; {generation_seconds:.1f}s)",
                    flush=True,
                )

            checkpoint = save_outputs(args.output_dir, selection, progress, run_started)
            print(
                f"CHECKPOINT {checkpoint['completed_facts']}/{checkpoint['selected_facts']} facts, "
                f"{checkpoint['completed_trials']}/{checkpoint['total_trials']} trials "
                f"({checkpoint['progress_fraction']:.1%}); "
                f"known={checkpoint['known_so_far']}, unknown={checkpoint['unknown_so_far']}; "
                f"elapsed {format_duration(checkpoint['elapsed_seconds_this_run'])}, "
                f"ETA {format_duration(checkpoint['eta_seconds'])}",
                flush=True,
            )

    checkpoint = save_outputs(args.output_dir, selection, progress, run_started)
    print(
        f"COMPLETE: known={checkpoint['known_so_far']}, "
        f"unknown={checkpoint['unknown_so_far']}, "
        f"elapsed {format_duration(checkpoint['elapsed_seconds_this_run'])}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
