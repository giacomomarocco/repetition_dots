#!/usr/bin/env python3
"""Plot the 2x3 top-token/top-numerical-token object heatmap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from filler.dsv4.top_object_heatmap import aggregate_rows, plot_stats, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True,
                        help="JSONL with top_token_id, top_numeric_token_id, and targets")
    parser.add_argument("--filler-length", type=int, default=20)
    parser.add_argument("--cohort", choices=("all", "correct", "wrong"), default="all")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    args = parser.parse_args()
    correct = None if args.cohort == "all" else args.cohort == "correct"
    stats = aggregate_rows(
        read_jsonl(args.rows), filler_length=args.filler_length, correct=correct
    )
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    plot_stats(stats, args.output)


if __name__ == "__main__":
    main()
