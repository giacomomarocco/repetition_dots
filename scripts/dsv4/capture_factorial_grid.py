#!/usr/bin/env python3
"""Capture complete token trajectories for a rendered multi-length panel grid."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer


def post(url: str, payload: dict) -> dict:
    request = urllib.request.Request(url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rendered", type=Path, nargs="+", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frozen-final-sites", action="store_true",
        help="capture final filler (when present) and answer prompt only")
    parser.add_argument("--url", default="http://127.0.0.1:30002/generate")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    trigger = args.capture_root / "CAPTURE_NEXT"
    base_mtime_ns = time.time_ns() + 1_000_000_000
    request_index = 0

    for rendered_path in args.rendered:
        rendered = json.loads(rendered_path.read_text())
        k = int(rendered["filler_length"])
        for panel in rendered["eligible_panels"]:
            output_dir = args.output_root / f"k{k}" / panel["panel_id"]
            manifest_path = output_dir / "manifest.json"
            if manifest_path.is_file():
                print(f"skip completed k={k} {panel['panel_id']}", flush=True)
                continue
            output_dir.mkdir(parents=True, exist_ok=True)
            positions = [("last_question", panel["positions"]["last_question"])]
            positions.extend((f"filler_{i}", p) for i, p in enumerate(panel["positions"]["fillers"]))
            positions.append(("answer_prompt", panel["positions"]["answer_prompt"]))
            if args.frozen_final_sites:
                positions = ([positions[-2]] if panel["positions"]["fillers"] else []) + [positions[-1]]
            cell_records = []
            for cell in panel["cells"]:
                ids = cell["input_ids"]
                tokens = tokenizer.convert_ids_to_tokens(ids)
                target_ids = {}
                for label, value in {"A": cell["left_value"], "X": cell["right_value"],
                                     "A+X": cell["target"]}.items():
                    encoded = tokenizer.encode(str(value), add_special_tokens=False)
                    if len(encoded) != 1:
                        raise RuntimeError(f"non-single-token target {cell['cell_id']} {label}")
                    target_ids[label] = encoded[0]
                captures = []
                for label, position in positions:
                    before = {p.name for p in (args.capture_root / "rank0").glob("pass*.pt")}
                    trigger.write_text(f"k={k} {cell['cell_id']} {label}\n")
                    stamp = base_mtime_ns + request_index * 1_000_000_000
                    os.utime(trigger, ns=(stamp, stamp))
                    response = post(args.url, {"input_ids": ids[:position + 1],
                        "sampling_params": {"temperature": 0, "max_new_tokens": 1},
                        "return_logprob": True, "top_logprobs_num": 10})
                    created = {p.name for p in (args.capture_root / "rank0").glob("pass*.pt")} - before
                    if len(created) != 1:
                        raise RuntimeError(f"k={k} {cell['cell_id']} {label}: captures={sorted(created)}")
                    name = created.pop()
                    if any(not (args.capture_root / f"rank{rank}" / name).is_file() for rank in range(1, 4)):
                        raise RuntimeError(f"incomplete TP capture {name}")
                    pass_id = int(name.removeprefix("pass").removesuffix(".pt"))
                    response_path = output_dir / f"{pass_id:05d}-{cell['row']}{cell['col']}-{label}.json"
                    response_path.write_text(json.dumps(response, indent=2, sort_keys=True) + "\n")
                    captures.append({"label": label, "absolute_position": position,
                        "token_id": ids[position], "token": tokens[position], "pass_id": pass_id,
                        "generated_token_id": response["output_ids"][0], "response": str(response_path)})
                    request_index += 1
                cell_records.append({
                    **{key: value for key, value in cell.items() if key not in {"input_ids", "rendered_prompt"}},
                    "target_token_ids": target_ids, "captures": captures,
                    "clean_generated_token_id": captures[-1]["generated_token_id"],
                    "clean_correct": captures[-1]["generated_token_id"] == target_ids["A+X"],
                })
            manifest = {"rendered_manifest": str(rendered_path), "panel_id": panel["panel_id"],
                "split": cell_records[0]["split"], "filler_length": k, "positions": panel["positions"],
                "prompt_token_count": panel["prompt_token_count"],
                "capture_root": str(args.capture_root), "cells": cell_records}
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            print(f"completed k={k} {panel['panel_id']} requests={request_index}", flush=True)


if __name__ == "__main__":
    main()
