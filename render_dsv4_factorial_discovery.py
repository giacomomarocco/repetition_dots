#!/usr/bin/env python3
"""Render and validate one-fact discovery panels for live residual capture."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from transformers import AutoTokenizer

from dsv4_factorial import FactorialCell, assert_aligned_and_single_token, locate_positions
from evaluate_facts_sglang import DEFAULT_ENCODER, DEFAULT_MODEL
from one_fact_addition_sglang import assistant_prefix, filler, load_encoder, load_facts, render_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--encoder", type=Path, default=DEFAULT_ENCODER)
    parser.add_argument("--filler-length", type=int, default=10)
    parser.add_argument("--split", choices=["discovery", "confirmation"], default="discovery")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    design = json.loads(args.design.read_text())
    facts, facts_sha256 = load_facts(args.facts)
    facts_by_id = {row["fact_id"]: row for row in facts}
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    encode_messages = load_encoder(args.encoder)

    panels: dict[str, list[dict]] = defaultdict(list)
    for raw in design["cells"]:
        if raw["split"] == args.split and raw["kind"] == "one_fact":
            panels[raw["panel_id"]].append(raw)

    accepted = []
    rejected = []
    for panel_id, raw_cells in panels.items():
        records = []
        try:
            for raw in sorted(raw_cells, key=lambda row: (row["row"], row["col"])):
                cell = FactorialCell(**{key: raw[key] for key in FactorialCell.__dataclass_fields__})
                fact = facts_by_id[cell.left_id]
                task = {
                    "question": fact["question"], "answer_value": cell.left_value,
                    "addend": cell.right_value, "target": cell.target, "k": args.filler_length,
                }
                prompt = render_prompt(encode_messages, task)
                suffix = assistant_prefix(args.filler_length)
                question_prefix = prompt[: -len(suffix)]
                filler_text = (
                    filler(args.filler_length) + "\n" if args.filler_length else ""
                )
                input_ids, positions = locate_positions(
                    tokenizer, prompt, question_prefix=question_prefix,
                    filler_text=filler_text, answer_prefix="Answer: ",
                )
                records.append({
                    "cell": cell, "positions": positions, "input_ids": input_ids,
                    "rendered_prompt": prompt, "question": fact["question"],
                })
            assert_aligned_and_single_token(records, tokenizer)
        except Exception as exc:
            rejected.append({"panel_id": panel_id, "reason": f"{type(exc).__name__}: {exc}"})
            continue
        accepted.append({
            "panel_id": panel_id,
            "positions": asdict(records[0]["positions"]),
            "prompt_token_count": len(records[0]["input_ids"]),
            "cells": [
                {
                    **asdict(record["cell"]),
                    "cell_id": record["cell"].cell_id,
                    "question": record["question"],
                    "rendered_prompt": record["rendered_prompt"],
                    "input_ids": record["input_ids"],
                }
                for record in records
            ],
        })

    output = {
        "design": str(args.design), "facts": str(args.facts), "facts_sha256": facts_sha256,
        "model": str(args.model), "encoder": str(args.encoder),
        "filler_length": args.filler_length, "eligible_panel_count": len(accepted),
        "split": args.split,
        "rejected_panel_count": len(rejected), "eligible_panels": accepted,
        "rejected_panels": rejected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "eligible_panel_count": len(accepted), "rejected_panel_count": len(rejected),
        "first_eligible_panel": accepted[0]["panel_id"] if accepted else None,
    }, indent=2))


if __name__ == "__main__":
    main()
