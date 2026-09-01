#!/usr/bin/env python3
"""Capture every aligned residual position for one rendered factorial panel."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer


def post(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rendered", type=Path, required=True)
    parser.add_argument("--panel-id")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:30002/generate")
    args = parser.parse_args()

    rendered = json.loads(args.rendered.read_text())
    panels = rendered["eligible_panels"]
    panel = next(
        row for row in panels
        if args.panel_id is None or row["panel_id"] == args.panel_id
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    positions = [("last_question", panel["positions"]["last_question"])]
    positions.extend((f"filler_{i}", pos) for i, pos in enumerate(panel["positions"]["fillers"]))
    positions.append(("answer_prompt", panel["positions"]["answer_prompt"]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trigger = args.capture_root / "CAPTURE_NEXT"
    base_mtime_ns = time.time_ns() + 1_000_000_000
    records = []
    request_index = 0
    for cell in panel["cells"]:
        ids = cell["input_ids"]
        tokens = tokenizer.convert_ids_to_tokens(ids)
        numeric_ids = {}
        for label, value in {
            "A": cell["left_value"], "X": cell["right_value"], "A+X": cell["target"]
        }.items():
            encoded = tokenizer.encode(str(value), add_special_tokens=False)
            if len(encoded) != 1:
                raise RuntimeError(f"{cell['cell_id']} {label}={value} is not one token")
            numeric_ids[label] = encoded[0]

        captures = []
        for label, position in positions:
            before = {path.name for path in (args.capture_root / "rank0").glob("pass*.pt")}
            trigger.write_text(f"{cell['cell_id']} {label}\n")
            unique_mtime_ns = base_mtime_ns + request_index * 1_000_000_000
            os.utime(trigger, ns=(unique_mtime_ns, unique_mtime_ns))
            response = post(args.url, {
                "input_ids": ids[: position + 1],
                "sampling_params": {"temperature": 0, "max_new_tokens": 1},
                "return_logprob": True, "top_logprobs_num": 10,
            })
            after = {path.name for path in (args.capture_root / "rank0").glob("pass*.pt")}
            created = after - before
            if len(created) != 1:
                raise RuntimeError(
                    f"{cell['cell_id']} {label}: expected one new capture, got {sorted(created)}"
                )
            capture_name = created.pop()
            for rank in range(1, 4):
                if not (args.capture_root / f"rank{rank}" / capture_name).is_file():
                    raise RuntimeError(f"{capture_name} missing on rank {rank}")
            pass_id = int(capture_name.removeprefix("pass").removesuffix(".pt"))
            response_file = args.output_dir / f"{pass_id:05d}-{cell['row']}{cell['col']}-{label}.json"
            response_file.write_text(json.dumps(response, indent=2, sort_keys=True) + "\n")
            captures.append({
                "label": label, "absolute_position": position, "token_id": ids[position],
                "token": tokens[position], "pass_id": pass_id,
                "generated_token_id": response["output_ids"][0], "response": str(response_file),
            })
            request_index += 1
            print(f"captured {cell['cell_id']} {label} pass={pass_id}", flush=True)

        answer_capture = captures[-1]
        records.append({
            **{key: value for key, value in cell.items() if key not in {"input_ids", "rendered_prompt"}},
            "target_token_ids": numeric_ids, "captures": captures,
            "clean_generated_token_id": answer_capture["generated_token_id"],
            "clean_correct": answer_capture["generated_token_id"] == numeric_ids["A+X"],
        })

    manifest = {
        "rendered_manifest": str(args.rendered), "panel_id": panel["panel_id"],
        "split": "discovery", "filler_length": rendered["filler_length"],
        "positions": panel["positions"], "prompt_token_count": panel["prompt_token_count"],
        "capture_root": str(args.capture_root), "cells": records,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
