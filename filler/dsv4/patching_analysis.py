"""Panel means and paired panel bootstrap comparisons for the frozen campaign."""
from __future__ import annotations

import csv
from itertools import combinations
from collections import defaultdict
from pathlib import Path

from filler.dsv4.patching import LABELS, Journal, atomic_json
from filler.dsv4.patching_campaign import verify_records

FACTORS = ("site", "layer_set", "recomputation", "donor_role")


def observed_conditions(manifest: dict) -> list[tuple]:
    """Use only frozen conditions, including older manifests without a design field."""
    levels = [list(dict.fromkeys(s[f] for s in manifest["trials"])) for f in FACTORS]
    return sorted({tuple(s[f] for f in FACTORS) for s in manifest["trials"]},
                  key=lambda c: tuple(levels[i].index(v) for i, v in enumerate(c)))


def summarize(manifest: dict, journal: Journal, output: Path, *, resamples=None, seed=42,
              integrity: dict | None = None) -> dict:
    if manifest.get("design") == "filler-random":
        from filler.dsv4.random_analysis import summarize as random_summary
        return random_summary(manifest, journal, output, resamples=2000 if resamples is None else resamples, seed=seed, integrity=integrity)
    if manifest.get("design") == "filler-repeat":
        from filler.dsv4.repeat_analysis import summarize as repeat_summary
        return repeat_summary(manifest, journal, output, resamples=2000 if resamples is None else resamples, seed=seed, integrity=integrity)
    resamples = 20000 if resamples is None else resamples
    if manifest.get("design") == "filler-redundancy":
        from filler.dsv4.redundancy_analysis import summarize as redundancy_summary
        return redundancy_summary(manifest, journal, output, resamples=resamples, seed=seed, integrity=integrity)
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if integrity is None:
        integrity = verify_records(manifest, journal)
    panels = [p["panel_id"] for p in manifest["panels"]]
    grouped = defaultdict(list)
    trial_rows = []
    for spec in manifest["trials"]:
        r = journal.records[spec["record_id"]]
        condition = tuple(r[f] for f in FACTORS)
        for score in r["scores"]:
            grouped[r["panel_id"], condition, score["label"]].append(score)
            trial_rows.append({**{k: r[k] for k in ("record_id", "panel_id", "target_id", "donor_id", "runtime_id", "baseline_id", *FACTORS)}, **score})
    conditions = observed_conditions(manifest)
    values = np.zeros((len(panels), len(conditions), len(LABELS)))
    clean_values, patched_values = np.zeros_like(values), np.zeros_like(values)
    panel_rows = []
    for pi, panel in enumerate(panels):
        for ci, condition in enumerate(conditions):
            for li, label in enumerate(LABELS):
                rows = grouped[panel, condition, label]
                if len(rows) != 4:
                    raise ValueError("each condition must contain all four target roles in every panel")
                values[pi, ci, li] = np.mean([r["delta_logprob"] for r in rows])
                clean_values[pi, ci, li] = np.mean([r["clean_logprob"] for r in rows])
                patched_values[pi, ci, li] = np.mean([r["patched_logprob"] for r in rows])
                panel_rows.append({"panel_id": panel, **dict(zip(FACTORS, condition)),
                                   "mean_clean_logprob": float(clean_values[pi, ci, li]),
                                   "mean_patched_logprob": float(patched_values[pi, ci, li]),
                                   "candidate": label, "mean_delta_logprob": float(values[pi, ci, li])})
    # The same resampled panel indices are used for every condition and contrast.
    indices = np.random.default_rng(seed).integers(0, len(panels), (resamples, len(panels)))
    boot = values[indices].mean(axis=1)
    summary = []
    for ci, condition in enumerate(conditions):
        for li, label in enumerate(LABELS):
            lo, hi = np.quantile(boot[:, ci, li], [0.025, 0.975])
            summary.append({**dict(zip(FACTORS, condition)), "candidate": label,
                            "mean_clean_logprob": float(clean_values[:, ci, li].mean()),
                            "mean_patched_logprob": float(patched_values[:, ci, li].mean()),
                            "mean_delta_logprob": float(values[:, ci, li].mean()),
                            "bootstrap_se": float(boot[:, ci, li].std(ddof=1)),
                            "ci_low": float(lo), "ci_high": float(hi), "panels": len(panels)})
    contrasts = []
    for ci, cj in combinations(range(len(conditions)), 2):
        condition, other = conditions[ci], conditions[cj]
        axes = [i for i in range(len(FACTORS)) if condition[i] != other[i]]
        # Never label the confounded position+layer change as a position effect.
        if len(axes) != 1:
            continue
        axis = axes[0]
        for li, label in enumerate(LABELS):
            differences = values[:, ci, li] - values[:, cj, li]
            lo, hi = np.quantile(boot[:, ci, li] - boot[:, cj, li], [0.025, 0.975])
            contrasts.append({"factor": FACTORS[axis], "first": condition[axis], "second": other[axis],
                **{f: condition[i] for i, f in enumerate(FACTORS) if i != axis},
                "candidate": label, "mean_paired_difference": float(differences.mean()),
                "bootstrap_se": float((boot[:, ci, li] - boot[:, cj, li]).std(ddof=1)),
                "ci_low": float(lo), "ci_high": float(hi), "panels": len(panels)})
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("trial_scores", trial_rows), ("panel_means", panel_rows), ("conditions", summary), ("paired_comparisons", contrasts)):
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with (output / f"{name}.csv").open("w", newline="") as sink:
            writer = csv.DictWriter(sink, fields)
            writer.writeheader()
            writer.writerows(rows)
    labels = [f"{c[0]} / L{c[1]} / {'full' if c[2] == 'full_downstream' else 'answer only'} / {c[3]}" for c in conditions]
    # Keep automated reports independent of personal Matplotlib/TeX settings.
    with plt.rc_context({"text.usetex": False, "font.family": "DejaVu Sans"}):
        fig, axes = plt.subplots(1, 3, figsize=(17, 8), sharey=True)
        for li, (label, ax) in enumerate(zip(LABELS, axes)):
            rows = [r for r in summary if r["candidate"] == label]
            y = np.arange(len(rows))
            means = [r["mean_delta_logprob"] for r in rows]
            ax.hlines(y, [r["ci_low"] for r in rows], [r["ci_high"] for r in rows], color="tab:blue")
            ax.plot(means, y, "o", color="tab:blue")
            ax.axvline(0, color="gray", lw=1)
            ax.set_title(label.replace("_", " "))
            ax.set_xlabel("Patched − clean log probability (nats)")
            ax.set_yticks(y, labels)
            ax.grid(axis="x", alpha=.2)
        axes[0].invert_yaxis()
        fig.suptitle(f"{manifest.get('split', 'discovery').capitalize()} panel means and 95% panel bootstrap intervals")
        fig.tight_layout()
        for extension in ("png", "pdf"):
            fig.savefig(output / f"candidate_changes.{extension}", dpi=180)
        plt.close(fig)
    report = {"integrity": integrity, "config_hash": manifest["config_hash"],
              "design": manifest.get("design", "original"), "condition_count": len(conditions),
              "resamples": resamples, "seed": seed,
              "bootstrap_unit": "panel", "standard_error": "bootstrap standard deviation, ddof=1",
              "split": manifest.get("split", "discovery"),
              "paired_indices_shared_across_conditions": True,
              "interval": "percentile 95%; descriptive, no multiplicity adjustment",
              "conditions": summary, "paired_comparisons": contrasts}
    atomic_json(output / "summary.json", report)
    lines = [f"# One-fact addition patching: {manifest.get('design', 'original')}", "",
             f"All {len(panels)} panels retained, including incorrect clean answers. Effects are natural-log probability changes.", "",
             "Donor role `left` means different fact/same addend; `both` means different fact/different addend.", "",
             f"Intervals use {resamples:,} whole-panel bootstrap resamples (seed {seed}) with shared indices across conditions. Errors are ±1σ bootstrap standard errors (ddof=1); between-runtime variability is not separately estimated. Pairwise comparisons are first minus second; intervals are descriptive without multiplicity adjustment.", "",
             *(["All 20 fillers are replaced together at layers 0–42; filler_5 is replaced at layers 32–37 inclusive. Indices are zero-based. In answer-only mode, all other rows except the final prompt row are restored after every block. Filler_5 and the answer row continue evolving through layer 42 after replacement ends at layer 37.", "",
                "For all-fillers patching, the mode contrast tests propagation through the two intervening non-filler tokens before the answer-prediction position. Fillers are overwritten at every layer in both modes. Position set and layer range change together between intervention families, so their comparison cannot isolate a position effect.", ""] if manifest.get("design") == "filler-coverage" else []),
             "| Position | Layers | Mode | Donor | Candidate | Mean Δlog p ± SE | 95% CI |",
             "|---|---|---|---|---|---:|---|" ]
    for r in summary:
        lines.append(f"| {r['site']} | {r['layer_set']} | {r['recomputation']} | {r['donor_role']} | {r['candidate']} | {r['mean_delta_logprob']:+.4f} ± {r['bootstrap_se']:.4f} | [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] |")
    lines += ["", "![Candidate probability changes](candidate_changes.png)", "",
              "[Trial clean/patched scores](trial_scores.csv) · [Panel means](panel_means.csv) · [Condition intervals](conditions.csv) · [Paired comparisons](paired_comparisons.csv) · [JSON](summary.json) · [PDF](candidate_changes.pdf)", "",
              "## Matched contrasts", "",
              "| Factor | First − second | Fixed conditions | Candidate | Mean difference ± SE | 95% CI |",
              "|---|---|---|---|---:|---|"]
    for r in contrasts:
        fixed = ", ".join(f"{f}={r[f]}" for f in FACTORS if f != r["factor"])
        lines.append(f"| {r['factor']} | {r['first']} − {r['second']} | {fixed} | {r['candidate']} | {r['mean_paired_difference']:+.4f} ± {r['bootstrap_se']:.4f} | [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] |")
    lines.append("")
    from filler.dsv4.patching import atomic_bytes
    atomic_bytes(output / "REPORT.md", "\n".join(lines).encode())
    return report
