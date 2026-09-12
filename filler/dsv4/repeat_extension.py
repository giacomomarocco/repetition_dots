"""Historical repeat comparison and notebook export for the prepared extension."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import io
import json
from pathlib import Path

from types import SimpleNamespace

from filler.dsv4.patching import atomic_bytes, atomic_json, digest, file_digest


def freeze_history(workspace, manifest):
    root = (workspace / "runs/deepseek-v4-flash/one-fact-patching-filler-repeat").resolve()
    old = json.loads((root / "manifest.json").read_text())
    done = json.loads((root / "COMPLETE.json").read_text())
    if (done["status"] != "complete" or not done["integrity"]["passed"]
            or done["config_hash"] != old["config_hash"]
            or digest({k: v for k, v in old.items() if k != "config_hash"}) != old["config_hash"]):
        raise ValueError("historical repeat campaign is not valid and complete")
    for key in ("panels", "input_hashes", "checkpoint_files"):
        if manifest[key] != old[key]:
            raise ValueError(f"historical repeat inputs differ: {key}")
    if {s["source_filler_index"] for s in old["trials"]} != {0, 5}:
        raise ValueError("historical comparison requires sources 0 and 5")
    paths = [root / p for p in ("manifest.json", "COMPLETE.json", "results.jsonl", "analysis/bootstrap_indices.json")]
    return {"root": str(root), "config_hash": old["config_hash"],
            "input_hashes": {str(p): file_digest(p) for p in paths}}


def combined_report(manifest, journal, root, integrity):
    from filler.dsv4.repeat_analysis import export_rows, result_rows
    historical = manifest["historical_repeat"]
    for name, sha in historical["input_hashes"].items():
        if file_digest(Path(name)) != sha:
            raise ValueError(f"historical repeat changed: {name}")
    previous = Path(historical["root"])
    old = json.loads((previous / "manifest.json").read_text())
    for key in ("panels", "input_hashes", "checkpoint_files"):
        if old[key] != manifest[key]:
            raise ValueError(f"repeat comparison input mismatch: {key}")
    # Journal's constructor repairs tails and writes progress; historical input
    # must instead be read without any writes.
    records = {}
    for line in (previous / "results.jsonl").read_text().splitlines():
        envelope = json.loads(line)
        r = envelope["record"]
        if (envelope["sha256"] != digest(r) or r["config_hash"] != old["config_hash"]
                or r["record_id"] in records):
            raise ValueError("invalid historical journal envelope")
        records[r["record_id"]] = r
    old_journal = SimpleNamespace(records=records)
    rows = [{**r, "origin": "historical"} for r in result_rows(old, old_journal)]
    rows += [{**r, "origin": "new"} for r in result_rows(manifest, journal)]
    sites = [f"repeat_filler_{i}" for i in range(6)]
    indices = json.loads((previous / "analysis/bootstrap_indices.json").read_text())
    if indices["panels"] != [p["panel_id"] for p in manifest["panels"]]:
        raise ValueError("historical bootstrap panel order differs")
    metadata = {"design": "filler-repeat-combined", "config_hash": manifest["config_hash"],
        "historical": historical, "new_root": str(root), "integrity": integrity,
        "input_hashes": manifest["input_hashes"], "revisions": manifest["revisions"],
        "historical_integrity": json.loads((previous / "COMPLETE.json").read_text())["integrity"]}
    return export_rows(manifest["panels"], sites, rows, root / "combined", metadata, indices=indices["indices"])


def refresh_notebook(workspace):
    """Execute the existing notebook and preserve embedded figures without a GPU."""
    from IPython.terminal.interactiveshell import TerminalInteractiveShell
    from IPython.utils.capture import capture_output
    import matplotlib.pyplot as plt
    notebook = workspace / "notebooks/addition_accuracy.ipynb"
    doc = json.loads(notebook.read_text())
    shell = TerminalInteractiveShell.instance()
    # Terminal shells normally omit rich image formats. Explicit serialization
    # gives the notebook the same embedded PNGs as a notebook frontend.
    def display_figure(obj):
        from IPython.display import display
        if hasattr(obj, "savefig"):
            stream = io.BytesIO()
            obj.savefig(stream, format="png", dpi=140, bbox_inches="tight")
            display({"image/png": base64.b64encode(stream.getvalue()).decode()}, raw=True)
        else:
            display(obj)
    count = 0
    for cell in doc["cells"]:
        if cell["cell_type"] != "code":
            continue
        count += 1
        code = "".join(cell["source"])
        # Use this checkout for imports and artifacts, including isolated staging.
        code = code.replace("PROJECT_ROOT = Path('/pscratch/sd/m/marocco/sandbox/mech_int')", f"PROJECT_ROOT = Path({str(workspace)!r})")
        with plt.rc_context({"text.usetex": False, "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"}), capture_output() as captured:
            result = shell.run_cell(code, store_history=False)
        if result.error_before_exec or result.error_in_exec:
            raise RuntimeError(f"notebook cell {count} failed: {result.error_before_exec or result.error_in_exec}")
        shell.user_ns["display"] = display_figure
        outputs = []
        for name, text in (("stdout", captured.stdout), ("stderr", captured.stderr)):
            if text:
                outputs.append({"output_type": "stream", "name": name, "text": text.splitlines(keepends=True)})
        for out in captured.outputs:
            outputs.append({"output_type": "display_data", "data": out.data, "metadata": out.metadata})
        cell.update(outputs=outputs, execution_count=count)
    plt.close("all")
    atomic_json(notebook, doc)
    return {"passed": True, "executed_cells": count,
            "embedded_pngs": sum("image/png" in o.get("data", {}) for c in doc["cells"] for o in c.get("outputs", []))}
