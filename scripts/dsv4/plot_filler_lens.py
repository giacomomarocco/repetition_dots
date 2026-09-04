#!/usr/bin/env python3
"""Plot target-rank heatmaps for a DSV4 filler Logit Lens pilot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.results.read_text())["rows"]
    manifest = json.loads(args.manifest.read_text())
    positions = [item["label"] for item in manifest["captures"]]
    labels = ["A", "X", "A+X"]

    fig, axes = plt.subplots(3, 1, figsize=(13, 8), constrained_layout=True)
    csv_rows = []
    for axis, target in zip(axes, labels):
        matrix = np.array(
            [
                [
                    next(
                        row["targets"][target]["rank"]
                        for row in rows
                        if row["position_label"] == position and row["layer"] == layer
                    )
                    for layer in range(43)
                ]
                for position in positions
            ]
        )
        image = axis.imshow(
            np.log10(matrix), aspect="auto", origin="lower", cmap="viridis_r", vmin=0, vmax=5.2
        )
        axis.set_title(f"{target} token rank (log10; yellow is better)")
        axis.set_ylabel("prompt position")
        axis.set_yticks(range(len(positions)), positions)
        axis.set_xlabel("decoder layer")
        fig.colorbar(image, ax=axis, label="log10(rank)")
        for pos_index, position in enumerate(positions):
            best_layer = int(matrix[pos_index].argmin())
            csv_rows.append(
                {
                    "position": position,
                    "target": target,
                    "best_layer": best_layer,
                    "best_rank": int(matrix[pos_index, best_layer]),
                    "layer_42_rank": int(matrix[pos_index, 42]),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)


if __name__ == "__main__":
    main()
