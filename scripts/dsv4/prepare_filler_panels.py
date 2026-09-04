#!/usr/bin/env python3
"""Prepare many fact-disjoint, token-aligned one-fact filler panels."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from transformers import AutoTokenizer

from filler.addition.one_fact import load_encoder, load_facts, render_prompt, split_target_prompt
from filler.dsv4.factorial import FactorialCell, assert_aligned_and_single_token, donor_roles, locate_positions
from filler.fact_eval.sglang import DEFAULT_ENCODER, DEFAULT_MODEL


def stable_panel_id(split: str, fact_ids: list[str], xs: tuple[int, int]) -> str:
    value = "\0".join([split, *fact_ids, *(str(x) for x in xs)])
    return f"{split}-filler-" + hashlib.sha256(value.encode()).hexdigest()[:20]


def render_record(tokenizer, encoder, fact, x: int, k: int, cell: FactorialCell):
    task = {"question": fact["question"], "answer_value": fact["answer"],
            "addend": x, "target": fact["answer"] + x, "k": k}
    prompt = render_prompt(encoder, task)
    qprefix, filler_text, answer_prefix, generation_prefix = split_target_prompt(prompt, k)
    ids, positions = locate_positions(
        tokenizer, prompt, question_prefix=qprefix, filler_text=filler_text,
        answer_prefix=answer_prefix, generation_prefix=generation_prefix,
    )
    return {"cell": cell, "input_ids": ids, "positions": positions}


def curate_split(facts, split, count, tokenizer, encoder, lengths):
    # Pair only facts whose rendered question endpoint has the same absolute
    # position. Facts are consumed at most once within and across splits.
    groups = defaultdict(list)
    for fact in facts:
        if len(tokenizer.encode(str(fact["answer"]), add_special_tokens=False)) != 1:
            continue
        dummy = FactorialCell("dummy", split, 0, 0, fact["fact_id"], "x-10",
                              fact["answer"], 10, fact["answer"] + 10, "one_fact")
        try:
            record = render_record(tokenizer, encoder, fact, 10, 0, dummy)
        except ValueError:
            continue
        groups[record["positions"].last_question].append(fact)

    pairs = []
    for _, items in sorted(groups.items()):
        for index in range(0, len(items) - 1, 2):
            pairs.append((items[index], items[index + 1]))

    panels, cells = [], []
    x_candidates = [(x, x + 1) for x in range(10, 99, 2)]
    for pair_index, (left0, left1) in enumerate(pairs):
        if len(panels) >= count:
            break
        for shift in range(len(x_candidates)):
            xs = x_candidates[(pair_index + shift) % len(x_candidates)]
            sums = {left0["answer"] + xs[0], left0["answer"] + xs[1],
                    left1["answer"] + xs[0], left1["answer"] + xs[1]}
            if len(sums) != 4:
                continue
            pid = stable_panel_id(split, [left0["fact_id"], left1["fact_id"]], xs)
            panel_cells = [
                FactorialCell(pid, split, row, col, fact["fact_id"], f"x-{x}",
                              int(fact["answer"]), x, int(fact["answer"]) + x, "one_fact")
                for row, fact in enumerate((left0, left1))
                for col, x in enumerate(xs)
            ]
            try:
                for k in lengths:
                    records = [
                        render_record(tokenizer, encoder, (left0, left1)[cell.row],
                                      cell.right_value, k, cell)
                        for cell in panel_cells
                    ]
                    assert_aligned_and_single_token(records, tokenizer)
                    for cell in panel_cells:
                        for value in (cell.left_value, cell.right_value):
                            if len(tokenizer.encode(str(value), add_special_tokens=False)) != 1:
                                raise ValueError("non-single-token operand")
            except ValueError:
                continue
            panels.append({"panel_id": pid, "fact_ids": [left0["fact_id"], left1["fact_id"]],
                           "filler_lengths": lengths})
            cells.extend(panel_cells)
            break
    if len(panels) < count:
        raise RuntimeError(f"only curated {len(panels)} of {count} requested {split} panels")
    return panels, cells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--encoder", type=Path, default=DEFAULT_ENCODER)
    parser.add_argument("--filler-lengths", type=int, nargs="+", default=[0, 5, 10, 20])
    parser.add_argument("--discovery-panels", type=int, default=24)
    parser.add_argument("--confirmation-panels", type=int, default=60)
    args = parser.parse_args()
    facts, source_sha = load_facts(args.facts)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    encoder = load_encoder(args.encoder)

    # Allocate facts before curation so no confirmation fact can enter discovery.
    ordered = sorted(facts, key=lambda row: hashlib.sha256(row["fact_id"].encode()).digest())
    discovery_pool = ordered[: len(ordered) // 3]
    confirmation_pool = ordered[len(ordered) // 3 :]
    dpanels, dcells = curate_split(discovery_pool, "discovery", args.discovery_panels,
                                    tokenizer, encoder, args.filler_lengths)
    cpanels, ccells = curate_split(confirmation_pool, "confirmation", args.confirmation_panels,
                                    tokenizer, encoder, args.filler_lengths)
    cells = dcells + ccells
    payload = {
        "schema_version": 2,
        "source": {"path": str(args.facts.resolve()), "sha256": source_sha,
                   "knowledge_status": "pre-filtered by user"},
        "design": {"filler_lengths": args.filler_lengths,
                   "prompt_protocol": (
                       "five-shot; identical k fillers before Answer: in every "
                       "demonstration and target user turn"
                   ),
                   "independent_observation": "fact-disjoint 2x2 panel",
                   "discovery_panel_count": len(dpanels),
                   "confirmation_panel_count": len(cpanels)},
        "panels": dpanels + cpanels,
        "cells": [{**asdict(cell), "cell_id": cell.cell_id} for cell in cells],
        "donors": donor_roles(cells),
        "curation": {"fact_disjoint_across_all_panels": True,
                     "aligned_at_all_filler_lengths": True,
                     "single_token_operands_and_sums": True,
                     "clean_correctness": "record, never silently filter"},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["design"], indent=2))


if __name__ == "__main__":
    main()
