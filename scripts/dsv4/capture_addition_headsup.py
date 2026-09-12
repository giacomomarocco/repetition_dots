#!/usr/bin/env python3
"""Capture reproducible residual trajectories for heads-up addition prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer

from filler.addition.one_fact import split_target_prompt


def post(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def continuation_id(tokenizer, prefix: str, value: int) -> int:
    before = tokenizer.encode(prefix, add_special_tokens=False)
    after = tokenizer.encode(prefix + str(value), add_special_tokens=False)
    if after[: len(before)] != before or len(after) != len(before) + 1:
        raise RuntimeError(f"{value} is not one continuation token")
    return after[-1]


def selected_pair_ids(rows: list[dict], count: int, seed: int) -> list[str]:
    pair_ids = {row["pair_id"] for row in rows}
    return sorted(
        pair_ids,
        key=lambda value: hashlib.sha256(f"{seed}\0{value}".encode()).digest(),
    )[:count]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, nargs="+", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pairs-per-experiment", type=int, default=16)
    parser.add_argument("--trajectory-stride", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--url", default="http://127.0.0.1:30002/generate")
    args = parser.parse_args()
    if args.pairs_per_experiment < 1 or args.trajectory_stride < 1:
        parser.error("capture count and trajectory stride must be positive")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    trigger = args.capture_root / "CAPTURE_NEXT"
    base_mtime_ns = time.time_ns() + 1_000_000_000
    request_index = 0
    args.output_root.mkdir(parents=True, exist_ok=True)

    for prompts_path in args.prompts:
        rows = json.loads(prompts_path.read_text())
        experiment = "two_fact" if rows and "fact_1_id" in rows[0] else "one_fact"
        chosen = set(selected_pair_ids(rows, args.pairs_per_experiment, args.seed))
        for row in rows:
            if row["pair_id"] not in chosen:
                continue
            output_dir = args.output_root / experiment / row["condition"] / row["pair_id"]
            manifest_path = output_dir / "manifest.json"
            if manifest_path.is_file():
                print(f"skip completed {experiment} {row['condition']} {row['pair_id']}", flush=True)
                continue
            q, filler, answer, generation = split_target_prompt(row["rendered_prompt"], row["k"])
            q_ids = tokenizer.encode(q, add_special_tokens=False)
            qf_ids = tokenizer.encode(q + filler, add_special_tokens=False)
            qfa_ids = tokenizer.encode(q + filler + answer, add_special_tokens=False)
            full_ids = tokenizer.encode(row["rendered_prompt"], add_special_tokens=False)
            if full_ids[:len(q_ids)] != q_ids or full_ids[:len(qf_ids)] != qf_ids or full_ids[:len(qfa_ids)] != qfa_ids:
                raise RuntimeError(f"token boundary merge in {row['prompt_id']}")
            filler_positions = list(range(len(q_ids), len(qf_ids)))
            kept_fillers = [
                (i, position) for i, position in enumerate(filler_positions)
                if i == 0 or i == len(filler_positions) - 1 or i % args.trajectory_stride == 0
            ]
            positions = [("last_question", len(q_ids) - 1)]
            positions += [(f"filler_{i}", position) for i, position in kept_fillers]
            positions.append(("answer_prompt", len(qfa_ids) - 1))
            values = (
                {"A": row["answer_value"], "X": row["addend"], "A+X": row["target"]}
                if experiment == "one_fact" else
                {"A1": row["fact_1_answer"], "A2": row["fact_2_answer"], "A1+A2": row["target"]}
            )
            target_ids = {label: continuation_id(tokenizer, q + filler + answer + generation, value)
                          for label, value in values.items()}
            output_dir.mkdir(parents=True, exist_ok=True)
            captures = []
            for label, position in positions:
                before = {p.name for p in (args.capture_root / "rank0").glob("pass*.pt")}
                trigger.write_text(f"{experiment} {row['prompt_id']} {label}\n")
                stamp = base_mtime_ns + request_index * 1_000_000_000
                os.utime(trigger, ns=(stamp, stamp))
                response = post(args.url, {"input_ids": full_ids[:position + 1],
                    "sampling_params": {"temperature": 0, "max_new_tokens": 1},
                    "return_logprob": True, "top_logprobs_num": 10})
                created = {p.name for p in (args.capture_root / "rank0").glob("pass*.pt")} - before
                if len(created) != 1:
                    raise RuntimeError(f"{row['prompt_id']} {label}: captures={sorted(created)}")
                name = created.pop()
                if any(not (args.capture_root / f"rank{rank}" / name).is_file() for rank in range(1, 4)):
                    raise RuntimeError(f"incomplete TP capture {name}")
                pass_id = int(name.removeprefix("pass").removesuffix(".pt"))
                response_path = output_dir / f"{pass_id:05d}-{label}.json"
                response_path.write_text(json.dumps(response, indent=2, sort_keys=True) + "\n")
                captures.append({"label": label, "absolute_position": position,
                    "filler_index": int(label.split("_")[1]) if label.startswith("filler_") else None,
                    "token_id": full_ids[position], "token": tokenizer.convert_ids_to_tokens(full_ids[position]),
                    "pass_id": pass_id, "generated_token_id": response["output_ids"][0],
                    "response": str(response_path)})
                request_index += 1
            manifest = {"schema_version": 1, "source_prompts": str(prompts_path),
                "experiment": experiment, "prompt_id": row["prompt_id"], "pair_id": row["pair_id"],
                "condition": row["condition"], "filler_length": row["k"],
                "trajectory_stride": args.trajectory_stride, "values": values,
                "target_token_ids": target_ids, "prompt_token_count": len(full_ids),
                "capture_root": str(args.capture_root), "captures": captures}
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            print(f"completed {experiment} {row['condition']} {row['pair_id']} requests={request_index}", flush=True)


if __name__ == "__main__":
    main()
