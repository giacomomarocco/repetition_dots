#!/usr/bin/env python3
"""Add exact canonical-numeric argmax IDs to factorial Logit Lens rows."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
from transformers import AutoTokenizer

from filler.dsv4.lens import load_checkpoint_readout, project_logits
from filler.dsv4.top_object_heatmap import canonical_numeric_token_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--grid-root", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    numeric_ids = canonical_numeric_token_ids(tokenizer)
    if not numeric_ids:
        raise RuntimeError("tokenizer has no canonical integer tokens")
    device_ids = torch.tensor(numeric_ids, dtype=torch.long, device=args.device)
    weights = load_checkpoint_readout(args.checkpoint, device=args.device)
    # Select once: repeating this copy for every one of the 4,128 captures is
    # needlessly expensive. project_logits is otherwise vocabulary-size agnostic.
    numeric_weights = replace(
        weights, lm_head_weight=weights.lm_head_weight.index_select(0, device_ids)
    )
    manifests = sorted(args.grid_root.glob("k*/*/manifest.json"))
    if not manifests:
        raise RuntimeError(f"no manifests below {args.grid_root}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with args.rows.open() as source, args.output.open("w") as sink:
        for manifest_path in manifests:
            manifest = json.loads(manifest_path.read_text())
            for cell in manifest["cells"]:
                for capture in cell["captures"]:
                    capture_path = args.capture_root / "rank0" / f"pass{capture['pass_id']:05d}.pt"
                    item = torch.load(capture_path, map_location="cpu", weights_only=True)
                    states = torch.stack([item["states"][layer] for layer in range(43)]).to(args.device)
                    restricted = project_logits(states, numeric_weights)
                    winners = device_ids[restricted.argmax(dim=-1)].tolist()
                    for layer, winner in enumerate(winners):
                        line = source.readline()
                        if not line:
                            raise RuntimeError("input rows ended before manifests")
                        row = json.loads(line)
                        expected = (cell["cell_id"], capture["label"], layer)
                        actual = (row["cell_id"], row["position_label"], row["layer"])
                        if actual != expected:
                            raise RuntimeError(f"row order mismatch: expected {expected}, got {actual}")
                        row["top_numeric_token_id"] = int(winner)
                        sink.write(json.dumps(row, separators=(",", ":")) + "\n")
                        row_count += 1
            sink.flush()
        if source.readline():
            raise RuntimeError("input rows remain after all manifests")
    print(json.dumps({"rows": row_count, "numeric_token_count": len(numeric_ids),
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
