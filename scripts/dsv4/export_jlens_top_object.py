"""Validate completed J-Lens rows, execute the notebook, and export its plots."""
from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from filler.dsv4.jlens_top_object import ROOT, ROWS_NAME, collect_examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    complete = json.loads((run / "COMPLETE.json").read_text())
    provenance = json.loads((run / "provenance.json").read_text())
    if complete.get("passed") is not True:
        raise ValueError("scoring did not complete")
    examples, hashes = collect_examples(Path(provenance["arguments"]["grid_root"]),
                                       Path(provenance["arguments"]["capture_root"]),
                                       complete["filler_lengths"])
    if hashes != provenance["manifest_sha256"]:
        raise ValueError("input manifests changed since scoring")
    lookup = {(row["filler_length"], row["cell_id"], row["position_label"]): row for row in examples}
    numeric_ids = set(provenance["numeric_token_ids"])
    seen = set()
    with (run / ROWS_NAME).open() as source:
        for line in source:
            row = json.loads(line)
            key = (row["filler_length"], row["cell_id"], row["position_label"])
            full_key = (*key, row["layer"])
            if full_key in seen or row["layer"] not in range(19, 40) or row["lens"] != "jlens":
                raise ValueError(f"duplicate or incompatible row: {full_key}")
            seen.add(full_key)
            expected = lookup[key]
            for name, value in expected.items():
                if name != "capture_path" and row[name] != value:
                    raise ValueError(f"row/manifest mismatch: {full_key}, {name}")
            if row["top_numeric_token_id"] not in numeric_ids:
                raise ValueError(f"non-numeric restricted argmax: {full_key}")
    if len(seen) != complete["rows"] or len(seen) != len(examples) * 21:
        raise ValueError("incomplete row grid")

    notebook_path = ROOT / "notebooks/one_fact_top_object_jlens_heatmap.ipynb"
    notebook = json.loads(notebook_path.read_text())
    namespace, outputs = {}, []

    def capture_show():
        buffer = io.BytesIO()
        plt.gcf().savefig(buffer, format="png", dpi=120)
        outputs.append({"output_type": "display_data", "metadata": {},
                        "data": {"image/png": base64.b64encode(buffer.getvalue()).decode()}})
        plt.close(plt.gcf())

    original_show = plt.show
    plt.show = capture_show
    try:
        execution_count = 0
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            execution_count += 1
            outputs = []
            source = "".join(cell["source"])
            source = source.replace("RUN = ROOT / 'runs/deepseek-v4-flash/jlens-top-object-heatmap'",
                                    f"RUN = Path({str(run)!r})")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exec(compile(source, f"{notebook_path}:cell{index}", "exec"), namespace)
            cell["execution_count"] = execution_count
            cell["outputs"] = ([{"output_type": "stream", "name": "stdout", "text": stdout.getvalue()}]
                               if stdout.getvalue() else []) + outputs
            cell["source"] = source.splitlines(keepends=True)
    finally:
        plt.show = original_show
    (run / "one_fact_top_object_jlens_heatmap.executed.ipynb").write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")

    cohort_sizes = {}
    plt.show = lambda: None
    try:
        for length in complete["filler_lengths"]:
            cohort_sizes[str(length)] = {}
            for cohort in ("all", "correct", "wrong"):
                stats = namespace["show_heatmap"](length, cohort)
                counts = set(stats["example_counts"].values())
                if len(counts) != 1:
                    raise ValueError("inconsistent cohort denominators")
                cohort_sizes[str(length)][cohort] = counts.pop()
                stem = run / f"top_object_jlens_k{length}_{cohort}"
                stem.with_suffix(".json").write_text(json.dumps(stats, indent=2) + "\n")
                plt.gcf().savefig(stem.with_suffix(".png"), dpi=180)
                plt.gcf().savefig(stem.with_suffix(".pdf"))
                plt.close(plt.gcf())
    finally:
        plt.show = original_show
        plt.close("all")
    validation = {"passed": True, "unique_rows": len(seen), "captures": len(examples),
                  "cohort_sizes": cohort_sizes, "numeric_token_count": len(numeric_ids),
                  "source_layers": complete["source_layers"],
                  "rows_sha256": hashlib.sha256((run / ROWS_NAME).read_bytes()).hexdigest(),
                  "executed_code_cells": execution_count, "figures": len(complete["filler_lengths"]) * 3}
    (run / "EXPORT_VALIDATION.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
