#!/usr/bin/env python3
"""Prepare disjoint DeepSeek-V4 one- and two-fact factorial manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from filler.addition.one_fact import load_facts
from filler.dsv4.factorial import crossed_panels, donor_roles, write_manifest


def _axis(facts, n, offset=0):
    """Take distinct numerical values while preserving a disjoint fact slice."""
    chosen, seen = [], set()
    cursor = offset
    while cursor < len(facts) and len(chosen) < n:
        f = facts[cursor]
        cursor += 1
        value = int(f["answer"])
        if value not in seen:
            chosen.append((f["fact_id"], value))
            seen.add(value)
    if len(chosen) != n:
        raise ValueError(f"need {n} distinct-valued verified facts after offset {offset}")
    return chosen, cursor


def _pair_deltas(axis):
    return {abs(axis[i][1] - axis[i + 1][1]) for i in range(0, len(axis), 2)}


def _pair_avoiding(values, n, forbidden):
    """Form pairs whose deltas cannot create repeated 2x2 sums."""
    pool = list(values)
    if len(pool) < n or n % 2:
        raise ValueError("pairing requires at least an even number of requested values")

    def allowed(a, b):
        return a[1] != b[1] and abs(a[1] - b[1]) not in forbidden

    def match(items):
        if not items:
            return []
        # Branch first on the most constrained item. This finds a perfect
        # matching without the avoidable dead ends of the former greedy pass.
        i = min(range(len(items)), key=lambda k: sum(allowed(items[k], x) for j, x in enumerate(items) if j != k))
        first = items[i]
        rest = items[:i] + items[i + 1:]
        for j, other in enumerate(rest):
            if allowed(first, other):
                tail = match(rest[:j] + rest[j + 1:])
                if tail is not None:
                    return [first, other, *tail]
        return None

    result = match(pool[:n])
    if result is None:
        raise ValueError("could not curate distinct-sum factorial pairs from available values")
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--facts", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--discovery-size", type=int, default=18)
    p.add_argument("--confirmation-size", type=int, default=26)
    args = p.parse_args()
    if args.discovery_size % 2 or args.confirmation_size % 2:
        p.error("grid sizes must be even")
    facts, source_sha = load_facts(args.facts)
    d, c = args.discovery_size, args.confirmation_size
    cells = []
    # One-fact task: factual A values and explicit numerical X values are each
    # disjoint across splits. Two-fact task uses a separate set of facts.
    cursor = 0
    one_d_a, cursor = _axis(facts, d, cursor)
    one_c_a, cursor = _axis(facts, c, cursor)
    numeric_pool = [(f"x-{x}", x) for x in range(10, 100)]
    one_d_x = _pair_avoiding(numeric_pool, d, _pair_deltas(one_d_a))
    used_x = {x[0] for x in one_d_x}
    one_c_x = _pair_avoiding([x for x in numeric_pool if x[0] not in used_x], c,
                             _pair_deltas(one_c_a))
    two_d_u, cursor = _axis(facts, d, cursor)
    raw_two_d_v, cursor = _axis(facts, d, cursor)
    two_d_v = _pair_avoiding(raw_two_d_v, d, _pair_deltas(two_d_u))
    two_c_u, cursor = _axis(facts, c, cursor)
    raw_two_c_v, cursor = _axis(facts, c, cursor)
    two_c_v = _pair_avoiding(raw_two_c_v, c, _pair_deltas(two_c_u))
    for kind, split, left, right in (
        ("one_fact", "discovery", one_d_a, one_d_x),
        ("one_fact", "confirmation", one_c_a, one_c_x),
        ("two_fact", "discovery", two_d_u, two_d_v),
        ("two_fact", "confirmation", two_c_u, two_c_v),
    ):
        cells.extend(crossed_panels(left, right, split=split, kind=kind))
    payload = {
        "schema_version": 1,
        "source": {"path": str(args.facts.resolve()), "sha256": source_sha,
                   "knowledge_status": "input must contain independently verified known facts"},
        "design": {"discovery_grid": [d, d], "confirmation_grid": [c, c],
                   "split_unit": "fact value", "independent_observation": "factorial panel",
                   "rotation": "all four cells used as target"},
        "cells": [{**x.__dict__, "cell_id": x.cell_id} for x in cells],
        "donors": donor_roles(cells),
        "required_controls": ["same_sum_different_decomposition", "operand_order",
                              "same_value_different_factual_cue"],
        "target_labels": {"one_fact": ["A", "X", "A+X"],
                          "two_fact": ["A1", "A2", "A1+A2"]},
        "curation": {"all_four_sums_distinct": True, "single_token_sum": "validate after rendering",
                     "absolute_token_alignment": "validate after rendering",
                     "balance": ["carry", "no_carry", "answer_magnitude"],
                     "clean_correctness": "record every cell; never silently filter"},
    }
    write_manifest(args.output, payload)
    print(f"Wrote {len(cells)} cells and {len(payload['donors'])} rotated donor directions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
