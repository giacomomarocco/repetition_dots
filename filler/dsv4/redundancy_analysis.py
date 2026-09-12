"""Draw/target averages followed by shared whole-panel bootstrap inference."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from filler.dsv4.patching import LABELS, atomic_bytes, atomic_json

GAPS = (("mixed_minus_target", "mixed_sum", "target_sum"),
        ("donor_minus_target", "donor_sum", "target_sum"),
        ("donor_minus_mixed", "donor_sum", "mixed_sum"))


def summarize(manifest, journal, output: Path, *, resamples=20000, seed=42, integrity=None):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from filler.dsv4.patching_campaign import verify_records
    if integrity is None:
        integrity = verify_records(manifest, journal)
    output.mkdir(parents=True, exist_ok=True)
    panels = [p["panel_id"] for p in manifest["panels"]]
    conditions = [(family, size, role) for family, size in
                  [("early_block", 5), *[("later_subset", n) for n in range(1, 6)]] for role in ("left", "both")]
    metrics = [f"{stage}_{unit}:{label}" for unit in ("logit", "logprob")
               for label in LABELS for stage in ("clean", "patched", "delta")]
    metrics += [f"{stage}_logit_gap:{name}" for name, _, _ in GAPS for stage in ("clean", "patched", "delta")]
    grouped = defaultdict(dict)
    scores, gaps = [], []
    for spec in manifest["trials"]:
        record = journal.records[spec["record_id"]]
        metadata = {k: record[k] for k in ("record_id", "panel_id", "target_id", "donor_id", "runtime_id",
                                          "baseline_id", "family", "subset_size", "draw_id", "donor_role")}
        metadata.update(filler_indices=spec["filler_indices"], positions=spec["positions"])
        candidates = {c["label"]: c for c in record["scores"]}
        vector = {}
        for label, candidate in candidates.items():
            scores.append({**metadata, **candidate})
            for unit in ("logit", "logprob"):
                for stage in ("clean", "patched", "delta"):
                    vector[f"{stage}_{unit}:{label}"] = candidate[f"{stage}_{unit}"]
        for name, first, second in GAPS:
            row = {**metadata, "gap": name}
            for stage in ("clean", "patched", "delta"):
                value = candidates[first][f"{stage}_logit"] - candidates[second][f"{stage}_logit"]
                row[f"{stage}_logit_gap"] = value
                vector[f"{stage}_logit_gap:{name}"] = value
            gaps.append(row)
        condition = (record["family"], record["subset_size"], record["donor_role"])
        key = (record["target_id"], record["draw_id"])
        bucket = grouped[record["panel_id"], condition]
        if key in bucket:
            raise ValueError("duplicate target/draw observation")
        bucket[key] = vector
    values = np.zeros((len(panels), len(conditions), len(metrics)))
    panel_rows = []
    for pi, panel in enumerate(panels):
        targets = {c["cell_id"] for p in manifest["panels"] if p["panel_id"] == panel for c in p["cells"]}
        for ci, condition in enumerate(conditions):
            draws = range(manifest["draws"]) if condition[0] == "later_subset" else [0]
            bucket = grouped[panel, condition]
            if set(bucket) != {(t, d) for t in targets for d in draws} or len(targets) != 4:
                raise ValueError("each panel requires all four targets and every frozen draw")
            # Explicitly average draws within targets, then four targets within panel.
            values[pi, ci] = np.mean([np.mean([[bucket[t, d][m] for m in metrics] for d in draws], axis=0)
                                     for t in sorted(targets)], axis=0)
            for mi, metric in enumerate(metrics):
                panel_rows.append({"panel_id": panel, "family": condition[0], "subset_size": condition[1],
                                   "donor_role": condition[2], "metric": metric, "mean": float(values[pi, ci, mi]),
                                   "targets": 4, "draws_per_target": len(draws)})
    indices = np.random.default_rng(seed).integers(0, len(panels), (resamples, len(panels)))
    # Equivalent to values[indices].mean(1), without materializing a large 4D array.
    weights = np.stack([(indices == i).sum(1) for i in range(len(panels))], axis=1) / len(panels)
    boot = np.einsum("bp,pcm->bcm", weights, values)

    def stats(v, b):
        low, high = np.quantile(b, [.025, .975])
        return {"mean": float(v.mean()), "bootstrap_se": float(b.std(ddof=1)),
                "ci_low": float(low), "ci_high": float(high), "panels": len(panels)}

    summary = [{"family": c[0], "subset_size": c[1], "donor_role": c[2], "metric": m,
                **stats(values[:, ci, mi], boot[:, ci, mi])}
               for ci, c in enumerate(conditions) for mi, m in enumerate(metrics)]
    pairs = []
    for role in ("left", "both"):
        pairs += [("adjacent_size", ("later_subset", n + 1, role), ("later_subset", n, role)) for n in range(1, 5)]
        pairs.append(("early_minus_later5", ("early_block", 5, role), ("later_subset", 5, role)))
    pairs += [("donor_type", (family, size, "both"), (family, size, "left"))
              for family, size in [("early_block", 5), *[("later_subset", n) for n in range(1, 6)]]]
    contrasts = []
    for kind, first, second in pairs:
        a, b = conditions.index(first), conditions.index(second)
        for mi, m in enumerate(metrics):
            contrasts.append({"contrast": kind, "first": list(first), "second": list(second), "metric": m,
                              **stats(values[:, a, mi] - values[:, b, mi], boot[:, a, mi] - boot[:, b, mi])})
    for name, rows in (("trial_scores", scores), ("trial_gaps", gaps), ("panel_means", panel_rows),
                       ("conditions", summary), ("paired_comparisons", contrasts)):
        with (output / f"{name}.csv").open("w", newline="") as sink:
            writer = csv.DictWriter(sink, list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    report = {"design": "filler-redundancy", "config_hash": manifest["config_hash"], "integrity": integrity,
              "resamples": resamples, "seed": seed, "bootstrap_unit": "panel", "draws_per_size": manifest["draws"],
              "aggregation": "mean draws within each target, mean four targets within panel, mean panels",
              "paired_indices_shared_across_conditions": True, "standard_error": "bootstrap std ddof=1",
              "interval": "descriptive percentile 95%; no multiplicity adjustment",
              "conditions": summary, "paired_comparisons": contrasts}
    atomic_json(output / "summary.json", report)
    with plt.rc_context({"text.usetex": False, "font.family": "DejaVu Sans"}):
        for unit, names in (("logit", LABELS), ("logprob", LABELS), ("logit_gap", [g[0] for g in GAPS])):
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
            for ax, name in zip(axes, names):
                for role, color in (("left", "tab:blue"), ("both", "tab:orange")):
                    rows = [r for r in summary if r["family"] == "later_subset" and r["donor_role"] == role
                            and r["metric"] == f"delta_{unit}:{name}"]
                    ax.plot([r["subset_size"] for r in rows], [r["mean"] for r in rows], "o-", color=color, label=role)
                    ax.fill_between([r["subset_size"] for r in rows], [r["ci_low"] for r in rows],
                                    [r["ci_high"] for r in rows], color=color, alpha=.15)
                    early = next(r for r in summary if r["family"] == "early_block" and r["donor_role"] == role
                                 and r["metric"] == f"delta_{unit}:{name}")
                    ax.axhline(early["mean"], color=color, ls="--", label=f"{role}: early block")
                    ax.axhspan(early["ci_low"], early["ci_high"], color=color, alpha=.06)
                ax.axhline(0, color="gray", lw=.7)
                ax.set(title=name.replace("_", " "), xlabel="Number of later fillers patched", xticks=range(1, 6),
                       ylabel=f"Patched − clean {unit.replace('_', ' ')}")
                ax.legend(fontsize=7)
            fig.tight_layout()
            for ext in ("png", "pdf"):
                fig.savefig(output / f"{unit}_changes.{ext}", dpi=180)
            plt.close(fig)
    lines = ["# k=20 filler redundancy patching", "",
             f"All {len(panels)} discovery panels retained, including incorrect clean answers. Early filler_0–4 is a separate intervention from random later subsets drawn from filler_5–19. Complete post-block mHC residuals are replaced at layers 0–42 on every TP rank. Every request reruns the full target prompt, allowing downstream fillers and all of Answer: plus its trailing space to respond.", "",
             "Candidates retain all three labels even when token IDs coincide. Raw logits are captured before sampling, with full-vocabulary log normalizers; log probabilities are checked against native returned scores. The donor-minus-mixed gap compares addends with the donor fact fixed. `left` denotes different fact/same addend; `both` denotes different fact/different addend.", "",
             f"Each panel averages {manifest['draws']} draws per later subset size and four targets before aggregation. Uncertainty uses {resamples:,} shared whole-panel bootstrap resamples, seed {seed}, SE with ddof=1 and descriptive 95% percentile intervals. Draws are not independent panels. Intervals have no multiplicity adjustment and do not separately estimate runtime variability.", "",
             "| Family | Size | Donor | Candidate | Δlogit ± SE | Δlog p ± SE |",
             "|---|---:|---|---|---:|---:|"]
    lookup = {(r["family"], r["subset_size"], r["donor_role"], r["metric"]): r for r in summary}
    for family, size, role in conditions:
        for label in LABELS:
            a, b = [lookup[family, size, role, f"delta_{unit}:{label}"] for unit in ("logit", "logprob")]
            lines.append(f"| {family} | {size} | {role} | {label} | {a['mean']:+.4f} ± {a['bootstrap_se']:.4f} | {b['mean']:+.4f} ± {b['bootstrap_se']:.4f} |")
    lines += ["", "## Interpretation", ""]
    for role in ("left", "both"):
        for size in (1, 5):
            r = lookup["later_subset", size, role, "delta_logprob:target_sum"]
            lines.append(f"With {role} donors and {size} later fillers patched, target-answer Δlog p is {r['mean']:+.4f} ± {r['bootstrap_se']:.4f}, 95% interval [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}].")
        adjacent = [r for r in contrasts if r["contrast"] == "adjacent_size" and r["first"][2] == role
                    and r["metric"] == "delta_logprob:target_sum"]
        lines.append("Adjacent-size target-answer changes (larger minus smaller) are " +
                     "; ".join(f"{r['first'][1]}−{r['second'][1]}: {r['mean']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
                               for r in adjacent) + ". These show how the marginal effect changes across subset sizes.")
        rows = [lookup["later_subset", n, role, "delta_logit_gap:donor_minus_mixed"] for n in range(1, 6)]
        lines.append(f"For {role} donors, donor-minus-mixed gap changes at sizes 1–5 are " +
                     ", ".join(f"{r['mean']:+.4f}" for r in rows) + ".")
        for row in contrasts:
            if row["contrast"] == "early_minus_later5" and row["first"][2] == role and row["metric"] == "delta_logprob:target_sum":
                lines.append(f"The early block minus five later fillers contrast in target-answer Δlog p is {row['mean']:+.4f} ± {row['bootstrap_se']:.4f}, 95% interval [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}].")
    lines += ["", "Resilience to small subsets or nonlinear size effects would be evidence relevant to redundancy. Small effects or intervals spanning zero do not prove invariance. This experiment alone cannot establish active error correction or isolate an addend-specific representation; whole-residual replacement changes multiple features. The early/later comparison also changes position, so it is not a pure size contrast.", "",
              "![Logit changes](logit_changes.png)", "![Log probability changes](logprob_changes.png)",
              "![Logit gap changes](logit_gap_changes.png)", "",
              "[Trial scores](trial_scores.csv) · [Trial gaps](trial_gaps.csv) · [Panel means](panel_means.csv) · [All estimates and intervals](conditions.csv) · [Paired contrasts](paired_comparisons.csv) · [JSON](summary.json)", ""]
    atomic_bytes(output / "REPORT.md", "\n".join(lines).encode())
    return report
