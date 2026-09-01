#!/usr/bin/env python3
"""Project all residuals from a captured factorial panel through the native lens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from deepseek_v4_logit_lens import load_checkpoint_readout, project_logits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())

    rows, states = [], []
    for cell in manifest["cells"]:
        for capture in cell["captures"]:
            path = args.capture_root / "rank0" / f"pass{capture['pass_id']:05d}.pt"
            item = torch.load(path, map_location="cpu", weights_only=True)
            if list(item["states"]) != list(range(43)):
                raise AssertionError(f"incomplete capture: {path}")
            for layer in range(43):
                rows.append({
                    "cell_id": cell["cell_id"], "panel_id": manifest["panel_id"],
                    "split": manifest["split"], "row": cell["row"], "col": cell["col"],
                    "clean_correct": cell["clean_correct"],
                    "position_label": capture["label"],
                    "absolute_position": capture["absolute_position"], "layer": layer,
                })
                states.append(item["states"][layer])

    weights = load_checkpoint_readout(args.checkpoint, device=args.device)
    logits = project_logits(torch.stack(states).to(args.device), weights)
    cells = {cell["cell_id"]: cell for cell in manifest["cells"]}
    for index, row in enumerate(rows):
        vector = logits[index]
        metrics = {}
        for label, token_id in cells[row["cell_id"]]["target_token_ids"].items():
            value = vector[token_id]
            other = torch.cat((vector[:token_id], vector[token_id + 1 :]))
            metrics[label] = {
                "token_id": token_id, "logit": float(value),
                "rank": int((vector > value).sum()) + 1,
                "log_odds_vs_rest": float(value - torch.logsumexp(other, 0)),
            }
        row["targets"] = metrics
        row["top_token_id"] = int(vector.argmax())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"manifest": str(args.manifest), "rows": rows}, indent=2) + "\n")
    for cell in manifest["cells"]:
        selected = [row for row in rows if row["cell_id"] == cell["cell_id"]]
        best = min(selected, key=lambda row: row["targets"]["A+X"]["rank"])
        print(
            f"{cell['cell_id']} correct={cell['clean_correct']} best_sum_rank="
            f"{best['targets']['A+X']['rank']} at {best['position_label']} L{best['layer']}"
        )


if __name__ == "__main__":
    main()
