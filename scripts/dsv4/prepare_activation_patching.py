#!/usr/bin/env python3
"""Prepare the frozen one-fact residual-transplant execution manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rendered", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True,
                        help="Root containing k20/<panel>/manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=list(range(33, 43)))
    args = parser.parse_args()

    rendered = json.loads(args.rendered.read_text())
    design = json.loads(Path(rendered["design"]).read_text())
    if int(rendered["filler_length"]) <= 10:
        raise ValueError("filler_10 requires filler length of at least 11")
    roles = {(d["target_id"], d["role"]): d["donor_id"] for d in design["donors"]}
    rendered_cells = {
        cell["cell_id"]: cell
        for panel in rendered["eligible_panels"] for cell in panel["cells"]
    }
    captures = {}
    clean = {}
    source_capture_roots = set()
    for path in sorted(args.captures.glob("*/manifest.json")):
        manifest = json.loads(path.read_text())
        source_capture_roots.add(manifest["capture_root"])
        for cell in manifest["cells"]:
            captures[cell["cell_id"]] = {c["label"]: c["pass_id"] for c in cell["captures"]}
            clean[cell["cell_id"]] = {
                "generated_token_id": cell["clean_generated_token_id"],
                "correct": cell["clean_correct"],
            }
    if set(rendered_cells) != set(captures):
        raise RuntimeError("rendered cells and captured cells differ")
    if len(source_capture_roots) != 1:
        raise RuntimeError(f"expected one donor capture root, got {source_capture_roots}")

    sites = {"filler_5": ["filler_5"], "filler_10": ["filler_10"],
             "filler_5+filler_10": ["filler_5", "filler_10"]}
    runs = []
    for target_id in sorted(rendered_cells):
        for role in ("left", "both"):
            donor_id = roles[target_id, role]
            for site_name, positions in sites.items():
                for layer in args.layers:
                    runs.append({
                        "target_id": target_id, "donor_id": donor_id,
                        "donor_role": role, "donor_description": (
                            "addend_matched" if role == "left" else "both_different"
                        ),
                        "site": site_name, "positions": positions, "layer": layer,
                        "donor_pass_ids": {p: captures[donor_id][p] for p in positions},
                    })
    identity_controls = []
    for target_id in sorted(rendered_cells):
        for site_name, positions in sites.items():
            for layer in args.layers:
                identity_controls.append({
                    "target_id": target_id, "donor_id": target_id,
                    "donor_role": "identity", "donor_description": "identity",
                    "site": site_name, "positions": positions, "layer": layer,
                    "donor_pass_ids": {p: captures[target_id][p] for p in positions},
                })
    payload = {
        "schema_version": 1, "experiment": "one_fact_residual_transplant",
        "rendered": str(args.rendered), "capture_manifests": str(args.captures),
        "donor_capture_root": source_capture_roots.pop(),
        "layers": args.layers, "sites": sites,
        "semantics": {
            "residual": "complete post-block mHC residual",
            "layer_sweep": "one independently patched layer per run",
            "continuation": "replay every token after the earliest patched filler",
            "left_role": "different fact, same addend",
            "both_role": "different fact, different addend",
        },
        "clean_baselines": clean, "runs": runs, "identity_controls": identity_controls,
        "counts": {"target_examples": len(rendered_cells), "substantive_runs": len(runs),
                   "identity_controls": len(identity_controls),
                   "clean_baselines": len(clean)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["counts"], indent=2))


if __name__ == "__main__":
    main()
