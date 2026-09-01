#!/usr/bin/env python3
"""Capture a documented one-fact filler trajectory from a live DSV4 server."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer


def post(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-results", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:30002/generate")
    args = parser.parse_args()

    raw = json.loads(args.source_results.read_text())
    rows = raw if isinstance(raw, list) else raw["results"]
    row = next(item for item in rows if item.get("condition") == "dots_10")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ids = tokenizer.encode(row["rendered_prompt"], add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(ids)

    assistant = ids.index(128804)
    think_end = ids.index(128822, assistant)
    answer = next(i for i in range(think_end + 1, len(ids)) if tokens[i] == "Answer")
    filler_positions = list(range(think_end + 1, answer))
    if len(filler_positions) != 10 or tokens[answer : answer + 3] != ["Answer", ":", "Ġ"]:
        raise RuntimeError("documented dots_10 prompt has an unexpected token layout")
    positions = [("last_question", assistant - 1)]
    positions.extend((f"filler_{i}", pos) for i, pos in enumerate(filler_positions))
    positions.append(("answer_prompt", answer + 2))

    target_values = {"A": int(row["answer_value"]), "X": int(row["addend"]), "A+X": int(row["target"])}
    target_ids = {}
    for label, value in target_values.items():
        encoded = tokenizer.encode(str(value), add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(f"target {label}={value} is not one token: {encoded}")
        target_ids[label] = encoded[0]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    captures = []
    trigger = args.capture_root / "CAPTURE_NEXT"
    base_mtime_ns = time.time_ns() + 1_000_000_000
    for offset, (label, position) in enumerate(positions):
        before = {path.name for path in (args.capture_root / "rank0").glob("pass*.pt")}
        trigger.write_text(f"{label}\n")
        unique_mtime_ns = base_mtime_ns + offset * 1_000_000_000
        os.utime(trigger, ns=(unique_mtime_ns, unique_mtime_ns))
        result = post(
            args.url,
            {
                "input_ids": ids[: position + 1],
                "sampling_params": {"temperature": 0, "max_new_tokens": 1},
                "return_logprob": True,
                "top_logprobs_num": 10,
            },
        )
        after = {path.name for path in (args.capture_root / "rank0").glob("pass*.pt")}
        created = after - before
        if len(created) != 1:
            raise RuntimeError(f"{label}: expected one new rank0 capture, got {sorted(created)}")
        capture_name = created.pop()
        for rank in range(1, 4):
            if not (args.capture_root / f"rank{rank}" / capture_name).is_file():
                raise RuntimeError(f"{label}: {capture_name} missing on rank {rank}")
        pass_id = int(capture_name.removeprefix("pass").removesuffix(".pt"))
        response_path = args.output_dir / f"{pass_id:05d}-{label}.json"
        response_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        captures.append(
            {
                "label": label,
                "absolute_position": position,
                "token_id": ids[position],
                "token": tokens[position],
                "pass_id": pass_id,
                "prompt_tokens": len(ids[: position + 1]),
                "generated_token_id": result["output_ids"][0],
                "response": str(response_path),
            }
        )
        print(f"captured {label} position={position} pass={pass_id}", flush=True)

    manifest = {
        "source_results": str(args.source_results),
        "source_condition": "dots_10",
        "fact_id": row["fact_id"],
        "question": row["question"],
        "target_values": target_values,
        "target_token_ids": target_ids,
        "full_prompt_token_count": len(ids),
        "captures": captures,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
