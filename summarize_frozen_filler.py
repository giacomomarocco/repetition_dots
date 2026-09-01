#!/usr/bin/env python3
"""Panel-clustered summaries for a frozen final-filler Logit Lens rule."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def bootstrap_mean(values, draws=10000, seed=20260831):
    rng = random.Random(seed)
    n = len(values)
    sims = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(draws))
    return [sims[int(0.025 * draws)], sims[int(0.975 * draws)]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    grouped = defaultdict(lambda: defaultdict(list))
    rank1 = defaultdict(lambda: defaultdict(list))
    clean = defaultdict(dict)
    with args.rows.open() as source:
        for line in source:
            row = json.loads(line)
            k = int(row["filler_length"])
            clean[k][row["cell_id"]] = bool(row["clean_correct"])
            wanted = row["layer"] == 41 and (
                row["position_label"] == "answer_prompt"
                or (k > 0 and row["position_label"] == f"filler_{k - 1}")
            )
            if not wanted:
                continue
            site = "answer_prompt" if row["position_label"] == "answer_prompt" else "final_filler"
            key = (k, site)
            metric = row["targets"]["A+X"]
            grouped[key][row["panel_id"]].append(metric["log_odds_vs_rest"])
            rank1[key][row["panel_id"]].append(metric["rank"] == 1)
    summaries = []
    panel_values = {}
    for key, panels in sorted(grouped.items()):
        values = {panel: sum(xs) / len(xs) for panel, xs in panels.items()}
        rates = {panel: sum(xs) / len(xs) for panel, xs in rank1[key].items()}
        panel_values[key] = values
        vector = list(values.values())
        rate_vector = list(rates.values())
        summaries.append({"filler_length": key[0], "site": key[1], "panel_count": len(vector),
            "mean_log_odds": sum(vector) / len(vector), "log_odds_panel_bootstrap_95ci": bootstrap_mean(vector),
            "mean_rank1_rate": sum(rate_vector) / len(rate_vector),
            "rank1_panel_bootstrap_95ci": bootstrap_mean(rate_vector, seed=20260832),
            "clean_accuracy": sum(clean[key[0]].values()) / len(clean[key[0]])})
    changes = []
    for a, b in ((5, 10), (10, 20), (20, 50), (50, 100)):
        ka, kb = (a, "final_filler"), (b, "final_filler")
        if ka not in panel_values or kb not in panel_values:
            continue
        common = sorted(set(panel_values[ka]) & set(panel_values[kb]))
        diffs = [panel_values[kb][p] - panel_values[ka][p] for p in common]
        changes.append({"from": a, "to": b, "panel_count": len(common),
            "mean_paired_log_odds_change": sum(diffs) / len(diffs),
            "panel_bootstrap_95ci": bootstrap_mean(diffs, seed=20260833 + a)})
    output = {"analysis_unit": "fact-disjoint panel", "frozen_layer": 41,
              "summaries": summaries, "paired_changes": changes}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
