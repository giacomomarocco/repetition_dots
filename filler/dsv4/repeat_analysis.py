"""Manifest-driven repeat conditions and shared whole-panel uncertainty."""
from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path

from filler.dsv4.patching import atomic_bytes, atomic_json
from filler.dsv4.patching_campaign import read_response, verify_records

METRICS = ("clean_accuracy_pct", "patched_accuracy_pct", "delta_accuracy_pp",
           "correct_to_incorrect_pct", "incorrect_to_correct_pct",
           "clean_logprob", "patched_logprob", "delta_logprob",
           "clean_logit", "patched_logit", "delta_logit")
CAVEAT = ("Source comparisons combine source-position and replacement-count effects. "
          "Each intervention uses its own runtime-matched baseline. Historical and new "
          "runtimes are identified separately; intervals do not estimate runtime variance.")


def result_rows(manifest, journal):
    rows = []
    for spec in manifest["trials"]:
        r = journal.records[spec["record_id"]]
        baseline = journal.records[r["baseline_id"]]
        clean, patched = read_response(baseline), read_response(r)
        score, = r["scores"]
        correct_id = score["token_id"]
        ca, pa = int(clean["output_ids"][0] == correct_id), int(patched["output_ids"][0] == correct_id)
        row = {k: r[k] for k in ("record_id", "panel_id", "target_id", "site", "runtime_id", "baseline_id")}
        row.update(source_positions=spec["source_positions"], destination_positions=spec["positions"],
                   replacement_count=len(spec["positions"]), correct_token_id=correct_id,
                   clean_greedy_id=clean["output_ids"][0], patched_greedy_id=patched["output_ids"][0],
                   clean_correct=ca, patched_correct=pa, correct_to_incorrect=ca * (1 - pa),
                   incorrect_to_correct=(1 - ca) * pa,
                   clean_accuracy_pct=100 * ca, patched_accuracy_pct=100 * pa,
                   delta_accuracy_pp=100 * (pa - ca), correct_to_incorrect_pct=100 * ca * (1 - pa),
                   incorrect_to_correct_pct=100 * (1 - ca) * pa,
                   **{k: score[k] for k in METRICS if k.endswith(("logprob", "logit"))})
        rows.append(row)
    return rows


def export_rows(panels, sites, rows, output, metadata, *, indices=None):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    pids = [p["panel_id"] for p in panels]
    lookup = {(r["target_id"], r["site"]): r for r in rows}
    expected = {(c["cell_id"], s) for p in panels for c in p["cells"] for s in sites}
    if len(lookup) != len(rows) or set(lookup) != expected:
        raise ValueError("missing, duplicate or extra prompts/conditions")
    if len(set(sites)) != len(sites) or len(sites) < 1:
        raise ValueError("invalid repeat conditions")
    pairs = list(combinations(range(len(sites)), 2))
    values = np.zeros((len(panels), len(sites), len(METRICS)))
    panel_rows, paired_rows = [], []
    for pi, panel in enumerate(panels):
        cells = panel["cells"]
        if len(cells) != 4:
            raise ValueError("each panel must retain all four prompts")
        for ci, site in enumerate(sites):
            selected = [lookup[c["cell_id"], site] for c in cells]
            values[pi, ci] = np.mean([[r[m] for m in METRICS] for r in selected], axis=0)
            panel_rows.extend({"panel_id": panel["panel_id"], "site": site, "metric": m,
                               "mean": float(values[pi, ci, mi])} for mi, m in enumerate(METRICS))
        for ai, bi in pairs:
            for cell in cells:
                a, b = [lookup[cell["cell_id"], sites[i]] for i in (ai, bi)]
                paired_rows.append({"panel_id": panel["panel_id"], "target_id": cell["cell_id"],
                    "first": sites[bi], "second": sites[ai],
                    "first_baseline_id": b["baseline_id"], "second_baseline_id": a["baseline_id"],
                    "first_runtime_id": b["runtime_id"], "second_runtime_id": a["runtime_id"],
                    "cross_runtime": a["runtime_id"] != b["runtime_id"],
                    **{m: b[m] - a[m] for m in METRICS}})
    canonical_indices = np.random.default_rng(42).integers(0, len(panels), (2000, len(panels)))
    if indices is not None and not np.array_equal(indices, canonical_indices):
        raise ValueError("historical bootstrap indices/panel order differ")
    indices = canonical_indices
    boot = values[indices].mean(axis=1)

    def stats(v, b):
        lo, hi = np.quantile(b, [.025, .975])
        return {"mean": float(v.mean()), "bootstrap_se": float(b.std(ddof=1)),
                "ci_low": float(lo), "ci_high": float(hi), "panels": len(panels)}

    summaries = [{"site": s, "metric": m, **stats(values[:, ci, mi], boot[:, ci, mi])}
                 for ci, s in enumerate(sites) for mi, m in enumerate(METRICS)]
    contrasts = [{"contrast": f"{sites[bi]}_minus_{sites[ai]}", "metric": m,
                  **stats(values[:, bi, mi] - values[:, ai, mi], boot[:, bi, mi] - boot[:, ai, mi])}
                 for ai, bi in pairs for mi, m in enumerate(METRICS)]
    counts = [{"site": s, "n": len(panels) * 4,
               **{k: sum(r[k] for r in rows if r["site"] == s) for k in
                  ("clean_correct", "patched_correct", "correct_to_incorrect", "incorrect_to_correct")}}
              for s in sites]
    output.mkdir(parents=True, exist_ok=True)
    for name, data in (("per_example", rows), ("paired_examples", paired_rows), ("panel_means", panel_rows),
                       ("conditions", summaries), ("paired_comparisons", contrasts), ("accuracy_counts", counts)):
        with (output / f"{name}.csv").open("w", newline="") as sink:
            if data:
                writer = csv.DictWriter(sink, list(data[0]))
                writer.writeheader()
                writer.writerows(data)
    atomic_json(output / "bootstrap_indices.json", {"seed": 42, "panels": pids, "indices": indices.tolist()})
    report = {**metadata, "resamples": 2000, "seed": 42, "bootstrap_unit": "panel",
              "paired_indices_shared_across_conditions": True, "accuracy_counts": counts,
              "conditions": summaries, "paired_comparisons": contrasts,
              "clean_prediction_disagreements_across_runtimes": sum(
                  len({lookup[c["cell_id"], s]["clean_greedy_id"] for s in sites}) > 1
                  for p in panels for c in p["cells"]),
              "comparison_caveat": metadata.get("caveat", CAVEAT),
              "interval": "descriptive percentile 95%; no multiplicity adjustment or separate runtime variance estimate"}
    atomic_json(output / "summary.json", report)
    atomic_json(output / "reproducibility.json", {**metadata,
        "runtime_ids": sorted({r["runtime_id"] for r in rows}),
        "bootstrap": {"resamples": 2000, "seed": 42, "unit": "panel"}})
    estimates = {(r["site"], r["metric"]): r for r in summaries}
    destination_counts = {}
    for site in sites:
        observed = {r["replacement_count"] for r in rows if r["site"] == site}
        if len(observed) != 1:
            raise ValueError("replacement count varies within a repeat condition")
        destination_counts[site] = observed.pop()
    with plt.rc_context({"text.usetex": False, "font.family": "DejaVu Sans"}):
        for name, metrics, labels in (
            ("accuracy", ("patched_accuracy_pct", "delta_accuracy_pp"), ("Greedy accuracy (%)", "Patched − clean (percentage points)")),
            ("logprob", ("patched_logprob", "delta_logprob"), ("Correct-answer log probability", "Patched − clean log probability")),
            ("logit", ("patched_logit", "delta_logit"), ("Correct-answer raw logit", "Patched − clean raw logit"))):
            fig, axes = plt.subplots(1, 2, figsize=(max(10, len(sites) * 2.2), 4.4))
            for ax, metric, label in zip(axes, metrics, labels):
                for i, site in enumerate(sites):
                    e = estimates[site, metric]
                    historical = all(r.get("origin") == "historical" for r in rows if r["site"] == site)
                    color = "tab:orange" if historical else "tab:blue"
                    ax.plot(i, e["mean"], "o", color=color)
                    ax.vlines(i, e["ci_low"], e["ci_high"], color=color)
                    if metric.startswith("patched"):
                        clean = estimates[site, metric.replace("patched", "clean")]
                        ax.plot(i, clean["mean"], "x", color="black", label="Matched clean" if i == 0 else None)
                ax.set(xticks=list(range(len(sites))),
                       xticklabels=[metadata.get("site_labels", {}).get(s, f"Source {s.rsplit('_', 1)[1]}") + f"\n{destination_counts[s]} replaced" for s in sites],
                       ylabel=label, xlim=(-.5, len(sites)-.5))
                if metric.startswith("delta"):
                    ax.axhline(0, color="gray", lw=.8)
                else:
                    ax.legend()
            fig.suptitle(metadata.get("interval_label", "95% panel bootstrap intervals") + ("; orange = historical runtime" if any(r.get("origin") == "historical" for r in rows) else ""), fontsize=10)
            fig.tight_layout()
            for ext in ("png", "pdf"):
                fig.savefig(output / f"{name}.{ext}", dpi=180)
            plt.close(fig)
    lines = [metadata.get("report_title", "# Repeat filler activations: performance"), "",
        metadata.get("description", f"All {len(panels)*4} discovery prompts across {len(panels)} panels are retained, including incorrect baselines. Each has 20 fillers. Clean source residuals replace every later filler simultaneously at layers 0–42 on all four ranks; downstream and answer-prefix tokens evolve normally."), "",
        "Primary outcome: canonical single-token greedy answer accuracy. Secondary outcomes: correct-answer natural log probability and raw logit. Raw-logit changes alone do not establish degradation because the normalization also changes.", "",
        "| Condition | Clean correct | Patched correct | Δ accuracy (pp) ± SE | 95% CI | Correct→incorrect | Incorrect→correct |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for c in counts:
        e = estimates[c["site"], "delta_accuracy_pp"]
        lines.append(f"| {c['site']} | {c['clean_correct']}/{c['n']} | {c['patched_correct']}/{c['n']} | {e['mean']:+.2f} ± {e['bootstrap_se']:.2f} | [{e['ci_low']:+.2f}, {e['ci_high']:+.2f}] | {c['correct_to_incorrect']} | {c['incorrect_to_correct']} |")
    lines += ["", "Uncertainty uses 2,000 shared whole-panel bootstrap resamples, seed 42, with all four prompts retained per panel. SE uses ddof=1. Intervals are descriptive and have no multiplicity adjustment.", "", metadata.get("caveat", CAVEAT), ""]
    for site in sites:
        a = estimates[site, "delta_accuracy_pp"]
        lines.append(f"{site}: {destination_counts[site]} replacements; accuracy change {a['mean']:+.2f} pp relative to its clean baseline.")
        for unit in ("logprob", "logit"):
            e = estimates[site, f"delta_{unit}"]
            lines.append(f"Δ{unit} {e['mean']:+.4f} ± {e['bootstrap_se']:.4f} SE; 95% interval [{e['ci_low']:+.4f}, {e['ci_high']:+.4f}].")
    lines += ["", "All pairwise condition contrasts and both runtime IDs are exported in the paired CSVs. Negative accuracy and log-probability changes indicate degradation against the matched baseline.", "",
        "Native final-layer equivalence, identity controls, layer-42-only invariance, and exact replacement on four ranks are required. Native score tolerance is 0.15 with identical greedy IDs; replacement tensors must match exactly.", "",
        f"Instrumented forwards for this design in one uninterrupted runtime: {metadata.get('forward_passes', 'see individual campaign manifests')}. Restarts add fresh baselines and validation passes.", "",
        "![Accuracy](accuracy.png)", "![Log probability](logprob.png)", "![Raw logits](logit.png)", "",
        "[Per-example results](per_example.csv) · [Paired examples](paired_examples.csv) · [Summaries](conditions.csv) · [Paired contrasts](paired_comparisons.csv) · [Reproducibility](reproducibility.json) · [Bootstrap draws](bootstrap_indices.json)", ""]
    atomic_bytes(output / "REPORT.md", "\n".join(lines).encode())
    return report


def summarize(manifest, journal, output: Path, *, resamples=2000, seed=42, integrity=None):
    if resamples != 2000 or seed != 42:
        raise ValueError("filler-repeat requires 2,000 shared panel resamples, seed 42")
    integrity = verify_records(manifest, journal) if integrity is None else integrity
    if not integrity["passed"]:
        raise ValueError("cannot report a campaign with failed integrity")
    from filler.dsv4.patching import layer42_diagnostics
    sites = list(dict.fromkeys(s["site"] for s in manifest["trials"]))
    metadata = {k: manifest.get(k, {}) for k in ("design", "config_hash", "source_hashes", "input_hashes", "revisions", "checkpoint_files")}
    metadata.update(integrity=integrity,
        forward_passes=sum(manifest["counts"][k] for k in ("baselines", "trials", "identity")) + len(layer42_diagnostics(manifest, manifest["pilot_panel"])),
        native_validation=[r for r in journal.records.values() if r["kind"] == "native_validation"])
    return export_rows(manifest["panels"], sites, result_rows(manifest, journal), output, metadata)
