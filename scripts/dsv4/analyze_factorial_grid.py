#!/usr/bin/env python3
"""Stream native Logit Lens scores from a multi-length factorial capture grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from filler.dsv4.lens import load_checkpoint_readout, project_logits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-root", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    manifests = sorted(args.grid_root.glob("k*/*/manifest.json"))
    if not manifests:
        raise RuntimeError(f"no completed manifests below {args.grid_root}")
    weights = load_checkpoint_readout(args.checkpoint, device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with args.output.open("w") as sink:
        for manifest_path in manifests:
            manifest = json.loads(manifest_path.read_text())
            for cell in manifest["cells"]:
                for capture in cell["captures"]:
                    path = args.capture_root / "rank0" / f"pass{capture['pass_id']:05d}.pt"
                    item = torch.load(path, map_location="cpu", weights_only=True)
                    if list(item["states"]) != list(range(43)):
                        raise RuntimeError(f"incomplete layer capture {path}")
                    logits = project_logits(
                        torch.stack([item["states"][layer] for layer in range(43)]).to(args.device),
                        weights,
                    )
                    for layer, vector in enumerate(logits):
                        metrics = {}
                        for label, token_id in cell["target_token_ids"].items():
                            value = vector[token_id]
                            other_lse = torch.logsumexp(
                                torch.cat((vector[:token_id], vector[token_id + 1 :])), 0
                            )
                            metrics[label] = {"token_id": token_id, "logit": float(value),
                                "rank": int((vector > value).sum()) + 1,
                                "log_odds_vs_rest": float(value - other_lse)}
                        row = {"panel_id": manifest["panel_id"], "cell_id": cell["cell_id"],
                            "split": manifest["split"], "row": cell["row"], "col": cell["col"],
                            "left_value": cell["left_value"], "right_value": cell["right_value"],
                            "target": cell["target"], "clean_correct": cell["clean_correct"],
                            "filler_length": manifest["filler_length"],
                            "position_label": capture["label"],
                            "absolute_position": capture["absolute_position"], "layer": layer,
                            "top_token_id": int(vector.argmax()), "targets": metrics}
                        sink.write(json.dumps(row, separators=(",", ":")) + "\n")
                        row_count += 1
            sink.flush()
            print(f"scored {manifest_path.parent} rows={row_count}", flush=True)
    print(json.dumps({"manifests": len(manifests), "rows": row_count,
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
