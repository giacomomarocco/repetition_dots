"""Random cutoffs and historical repeat comparisons with independent export audit."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

from filler.dsv4.patching import atomic_json, digest, file_digest, score_candidates
from filler.dsv4.patching_campaign import read_response, verify_records
from filler.dsv4.repeat_analysis import METRICS, export_rows, result_rows

CAVEAT = ("Uncertainty is conditional on a single Gaussian noise realization (seed 42). "
          "Cutoff comparisons also change the number of replaced fillers. Each condition retains "
          "its own runtime baseline; historical repeat comparisons do not estimate between-runtime "
          "variance. Historical repeat captures have legacy position coverage; the new random "
          "campaign captures the last question, all 20 fillers, Answer, colon, and trailing space.")


def read_history(root, *, validate_raw=False):
    old = json.loads((root / "manifest.json").read_text())
    done = json.loads((root / "COMPLETE.json").read_text())
    if (done["status"] != "complete" or not done["integrity"]["passed"]
            or done["config_hash"] != old["config_hash"]
            or digest({k: v for k, v in old.items() if k != "config_hash"}) != old["config_hash"]):
        raise ValueError("historical repeat is not complete and valid")
    records = {}
    for line in (root / "results.jsonl").read_text().splitlines():
        envelope = json.loads(line)
        r = envelope["record"]
        if envelope["sha256"] != digest(r) or r["config_hash"] != old["config_hash"] or r["record_id"] in records:
            raise ValueError("historical journal checksum/config mismatch")
        records[r["record_id"]] = r
    for spec in old["trials"]:
        r = records[spec["record_id"]]
        baseline = records[r["baseline_id"]]
        if any(r[k] != v for k, v in spec.items()) or baseline["runtime_id"] != r["runtime_id"] or baseline["target_id"] != r["target_id"]:
            raise ValueError("historical trial/baseline mismatch")
        if validate_raw and r["scores"] != score_candidates(spec["candidates"], read_response(baseline), read_response(r)):
            raise ValueError("historical scores disagree with raw responses")
    return old, SimpleNamespace(records=records)


def freeze_history(workspace, manifest):
    history = []
    observed = set()
    for suffix in ("filler-repeat", "filler-repeat-1to4"):
        root = workspace / "runs/deepseek-v4-flash" / f"one-fact-patching-{suffix}"
        old, _ = read_history(root)
        for key in ("panels", "input_hashes", "checkpoint_files"):
            if manifest[key] != old[key]:
                raise ValueError(f"historical repeat inputs differ: {key}")
        sources = {s["source_filler_index"] for s in old["trials"]}
        if observed & sources:
            raise ValueError("overlapping historical sources")
        observed.update(sources)
        paths = [root / name for name in ("manifest.json", "COMPLETE.json", "results.jsonl", "analysis/bootstrap_indices.json")]
        history.append(dict(root=str(root), config_hash=old["config_hash"],
                            hashes={str(p): file_digest(p) for p in paths}))
    if observed != set(range(6)):
        raise ValueError("all six historical repeat cutoffs required")
    return history


def audit_exports(output):
    """Rebuild statistics from CSV cells, without reusing the exporter's arrays."""
    import numpy as np
    output = Path(output)
    rows = list(csv.DictReader((output / "per_example.csv").open()))
    summary = json.loads((output / "summary.json").read_text())
    samples = json.loads((output / "bootstrap_indices.json").read_text())
    pids = samples["panels"]
    draws = np.random.default_rng(42).integers(0, len(pids), (2000, len(pids)))
    if samples["indices"] != draws.tolist():
        raise ValueError("export bootstrap draws mismatch")
    vectors = {}
    max_error = 0.
    for entry in summary["conditions"]:
        site, metric = entry["site"], entry["metric"]
        values = np.array([sum(float(r[metric]) for r in rows if r["panel_id"] == p and r["site"] == site) / 4 for p in pids])
        vectors[site, metric] = values
    for entry in [*summary["conditions"], *summary["paired_comparisons"]]:
        if "site" in entry:
            v = vectors[entry["site"], entry["metric"]]
        else:
            first, second = entry["contrast"].split("_minus_")
            v = vectors[first, entry["metric"]] - vectors[second, entry["metric"]]
        b = v[draws].mean(1)
        lo, hi = np.quantile(b, [.025, .975])
        for key, value in dict(mean=v.mean(), bootstrap_se=b.std(ddof=1), ci_low=lo, ci_high=hi).items():
            error = abs(entry[key] - value)
            max_error = max(max_error, float(error))
            if error > 1e-10:
                raise ValueError(f"independent export reconstruction failed: {key}")
    for c in summary["accuracy_counts"]:
        selected = [r for r in rows if r["site"] == c["site"]]
        if c["n"] != len(selected):
            raise ValueError("export count mismatch")
        for key in ("clean_correct", "patched_correct", "correct_to_incorrect", "incorrect_to_correct"):
            if c[key] != sum(int(r[key]) for r in selected):
                raise ValueError("export correctness reconstruction failed")
    result = dict(passed=True, max_absolute_error=max_error, examples=len(rows),
                  whole_panel_resamples=2000, seed=42)
    atomic_json(output / "export-validation.json", result)
    return result


def summarize(manifest, journal, output, *, resamples=2000, seed=42, integrity=None):
    if resamples != 2000 or seed != 42:
        raise ValueError("random analysis requires 2,000 resamples, seed 42")
    integrity = verify_records(manifest, journal) if integrity is None else integrity
    if not integrity["passed"]:
        raise ValueError("failed campaign integrity")
    def provenance_rows(m, j, origin):
        result = []
        for row in result_rows(m, j):
            record = j.records[row["record_id"]]
            bank = record.get("replacement_bank", {})
            result.append(dict(row, origin=origin, cutoff=record.get("cutoff", record.get("source_filler_index")),
                noise_seed=42 if origin == "random" else "", replacement_bank_path=bank.get("path", ""),
                replacement_bank_sha256=bank.get("sha256", "")))
        return result
    rows = provenance_rows(manifest, journal, "random")
    sites = [f"random_after_{j}" for j in range(6)]
    metadata = {k: manifest.get(k, {}) for k in ("design", "config_hash", "source_hashes", "input_hashes", "revisions", "checkpoint_files", "historical_repeats", "position_coverage")}
    metadata.update(integrity=integrity, forward_passes=1272,
        report_title="# Independently randomized filler activations",
        description="All discovery prompts, including baseline errors, retain 20 fillers. For each cutoff j=0–5, fillers j+1 through 19 receive independent Gaussian directions at all post-block layers 0–42, scaled to each destination's clean full mHC L2 norm. The same position's replacement is shared across cutoffs and all four ranks. Full prompts are recomputed; answer-prefix tokens evolve normally. Cast norm error must be at most 0.5%, and saved replacements must agree exactly with the bank.",
        caveat=CAVEAT, interval_label="95% panel bootstrap intervals, conditional on one noise realization",
        site_labels={**{s: f"Random j={j}" for j, s in enumerate(sites)},
                     **{f"repeat_filler_{j}": f"Repeat j={j}" for j in range(6)}})
    report = export_rows(manifest["panels"], sites, rows, output, metadata)
    audit_exports(output)
    historical_rows = []
    for history in manifest.get("historical_repeats", []):
        for path, checksum in history["hashes"].items():
            if file_digest(Path(path)) != checksum:
                raise ValueError("frozen historical input changed")
        old, old_journal = read_history(Path(history["root"]), validate_raw=True)
        historical_rows.extend(provenance_rows(old, old_journal, "historical"))
        indices = json.loads((Path(history["root"]) / "analysis/bootstrap_indices.json").read_text())
        current = json.loads((output / "bootstrap_indices.json").read_text())
        if indices != current:
            raise ValueError("historical bootstrap differs")
    if historical_rows:
        combined_sites = [s for j in range(6) for s in (f"repeat_filler_{j}", f"random_after_{j}")]
        comparison = Path(output).parent / "comparison"
        export_rows(manifest["panels"], combined_sites, historical_rows + rows, comparison,
                    dict(metadata, report_title="# Random versus historical repeat filler interventions",
                         description=metadata["description"] + " Historical repeat conditions copy clean filler j to every later filler and retain their original runtime baselines."))
        audit_exports(comparison)
    return report
