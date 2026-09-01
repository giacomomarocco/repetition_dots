#!/usr/bin/env python3
"""Batch-project layer residuals from the DSV4 filler pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from deepseek_v4_logit_lens import load_checkpoint_readout, project_logits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())

    rows, states = [], []
    for capture in manifest["captures"]:
        path = args.capture_root / "rank0" / f"pass{capture['pass_id']:05d}.pt"
        item = torch.load(path, map_location="cpu", weights_only=True)
        if list(item["states"]) != list(range(43)):
            raise AssertionError(f"incomplete capture: {path}")
        for layer in range(43):
            rows.append({"position_label": capture["label"], "layer": layer})
            states.append(item["states"][layer])

    weights = load_checkpoint_readout(args.checkpoint, device=args.device)
    logits = project_logits(torch.stack(states).to(args.device), weights)
    target_ids = manifest["target_token_ids"]
    for index, row in enumerate(rows):
        vector = logits[index]
        metrics = {}
        for label, token_id in target_ids.items():
            value = vector[token_id]
            other = torch.cat((vector[:token_id], vector[token_id + 1 :]))
            metrics[label] = {
                "token_id": token_id,
                "logit": float(value),
                "rank": int((vector > value).sum()) + 1,
                "log_odds_vs_rest": float(value - torch.logsumexp(other, 0)),
            }
        row["targets"] = metrics
        row["top_token_id"] = int(vector.argmax())

    result = {"manifest": str(args.manifest), "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for position in manifest["captures"]:
        selected = [r for r in rows if r["position_label"] == position["label"]]
        best = min(selected, key=lambda r: r["targets"]["A+X"]["rank"])
        print(
            f"{position['label']}: best sum rank={best['targets']['A+X']['rank']} "
            f"at layer={best['layer']} log_odds={best['targets']['A+X']['log_odds_vs_rest']:.3f}"
        )


if __name__ == "__main__":
    main()
