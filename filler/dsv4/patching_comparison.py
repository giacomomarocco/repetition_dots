"""Read-only campaign reuse and covariance-preserving patching comparisons.

Historical tensor validation is reused from COMPLETE.json; raw responses,
journal membership, candidate scores and runtime baselines are checked again.
Only the caller's new output directory is written. No Journal is instantiated.
"""
from __future__ import annotations

from collections import defaultdict
import csv
from datetime import datetime, timezone
from itertools import combinations, product
import json
from pathlib import Path
import shlex

import numpy as np

from filler.dsv4.patching import LABELS, atomic_bytes, atomic_json, digest, file_digest, score_candidates
from filler.dsv4.patching_analysis import FACTORS
from filler.dsv4.patching_campaign import read_response

STAGES = ("clean", "patched", "shift")
METRICS = ("target_sum", "mixed_sum", "donor_sum", "donor_minus_target",
           "mixed_minus_target", "donor_minus_mixed")
LEVELS = (("filler_5", "filler_10"), ("0-42", "32-37", "33-42"),
          ("full_downstream", "answer_only"), ("left", "both"))
HISTORY_FILES = ("manifest.json", "COMPLETE.json", "results.jsonl",
    "analysis/panel_means.csv", "analysis/conditions.csv", "analysis/paired_comparisons.csv",
    "analysis/conditions_one_sigma.csv", "analysis/logit_gaps_one_sigma.csv",
    "analysis/donor_vs_mixed_one_sigma.csv", "analysis/filler5_layer_comparison.csv")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_campaign(root: Path, *, integrity: dict | None = None) -> dict:
    """Strict, non-repairing reader: never touch historical progress or journals."""
    completion_pending = integrity is not None
    manifest = json.loads((root / "manifest.json").read_text())
    require(digest({k: v for k, v in manifest.items() if k != "config_hash"}) == manifest["config_hash"],
            "manifest configuration checksum mismatch")
    if integrity is None:
        complete = json.loads((root / "COMPLETE.json").read_text())
        require(complete["status"] == "complete" and complete["config_hash"] == manifest["config_hash"],
                "completed matching campaign required")
        integrity = complete["integrity"]
    require(integrity["passed"], "campaign integrity failed")
    records = {}
    with (root / "results.jsonl").open("rb") as source:
        for line in source:
            require(line.endswith(b"\n"), "incomplete historical journal line; refusing repair")
            envelope = json.loads(line)
            r = envelope["record"]
            require(envelope["sha256"] == digest(r) and r["config_hash"] == manifest["config_hash"],
                    "journal checksum/config mismatch")
            require(r["record_id"] not in records, "duplicate journal record")
            records[r["record_id"]] = r
    for key, kind in (("trials", "trial"), ("identity_controls", "identity")):
        expected = {s["record_id"] for s in manifest[key]}
        require(len(expected) == len(manifest[key]) and expected == {
            rid for rid, r in records.items() if r["kind"] == kind}, "incomplete or extra trial/control membership")
        require(len(expected) == integrity["trials" if kind == "trial" else "identity"],
                "completion counts disagree with manifest")
    raw_cache, aliases = {}, 0
    def raw(record):
        key = record["record_id"]
        if key not in raw_cache:
            raw_cache[key] = read_response(record)
        return raw_cache[key]
    for spec in (*manifest["trials"], *manifest["identity_controls"]):
        r = records[spec["record_id"]]
        require(all(r[k] == v for k, v in spec.items()), "trial differs from manifest")
        b = records[r["baseline_id"]]
        require(b["kind"] == "baseline" and all(r[k] == b[k] for k in (
            "runtime_id", "target_id", "panel_id", "recomputation")), "baseline runtime/mode/target/panel mismatch")
        require(r["clean_capture"] == b["clean_capture"] and
                r["donor_capture"]["runtime_id"] == r["runtime_id"] and
                r["donor_capture"]["cell_id"] == r["donor_id"], "capture baseline mismatch")
        require(records[f"native-validation|{r['runtime_id']}"]["report"]["passed"], "native gate missing")
        if spec["kind"] == "trial":
            gate = records[f"panel-validation|{r['runtime_id']}|{r['panel_id']}"]
            require(gate["passed"] and r["baseline_id"] in gate["baseline_ids"], "panel gate missing")
        else:
            from filler.dsv4.patching import invariant
            require(invariant(raw(b), raw(r), b["scored_token_ids"])["passed"], "identity gate failed")
        require(score_candidates(spec["candidates"], raw(b), raw(r)) == r["scores"],
                "candidate scores differ from original raw responses")
        require(len(r["scores"]) == 3 and {s["label"] for s in r["scores"]} == set(LABELS),
                "candidate labels incomplete")
        if r["donor_role"] == "left":
            scores = {s["label"]: s for s in r["scores"]}
            require({k: v for k, v in scores["mixed_sum"].items() if k != "label"} ==
                    {k: v for k, v in scores["donor_sum"].items() if k != "label"}, "same-addend alias differs")
            aliases += 1
    inputs = {str(root / name): file_digest(root / name) for name in HISTORY_FILES
              if (root / name).exists() and not (completion_pending and name == "COMPLETE.json")}
    return {"root": root, "manifest": manifest, "records": records,
            "provenance": {"root": str(root), "config_hash": manifest["config_hash"],
                "integrity": integrity, "input_sha256": inputs,
                "journal_records": len(records), "raw_responses_checked": len(raw_cache),
                "same_addend_aliases_checked": aliases,
                "runtimes": sorted({r["runtime_id"] for r in records.values() if r["kind"] == "trial"}),
                "tensor_validation": "reuse completed campaign integrity; no historical tensor reread"}}


def trial_metrics(scores: list[dict]) -> np.ndarray:
    by_label = {s["label"]: s for s in scores}
    values = np.array([[by_label[label][key] for key in
        ("clean_logprob", "patched_logprob", "delta_logprob")] for label in METRICS[:3]])
    gaps = []
    for first, second in ((2, 0), (1, 0), (2, 1)):
        gap = values[first] - values[second]
        # Form paired gaps before averaging. No independent-error propagation.
        gap[2] = gap[1] - gap[0]
        gaps.append(gap)
    result = np.concatenate((values, gaps))
    np.testing.assert_allclose(result[:, 2], result[:, 1] - result[:, 0], atol=1e-12, rtol=0)
    require(np.isfinite(result).all(), "nonfinite scores")
    return result


def panel_values(campaigns: list[dict]):
    panels = [p["panel_id"] for p in campaigns[0]["manifest"]["panels"]]
    grouped, origins, trials = defaultdict(dict), {}, []
    for campaign in campaigns:
        for spec in campaign["manifest"]["trials"]:
            r = campaign["records"][spec["record_id"]]
            condition = tuple(r[f] for f in FACTORS)
            key = r["panel_id"], condition
            require(r["target_id"] not in grouped[key], "overlapping campaign trial")
            grouped[key][r["target_id"]] = trial_metrics(r["scores"])
            origins.setdefault(condition, set()).add(str(campaign["root"]))
            for score in r["scores"]:
                trials.append({"campaign": str(campaign["root"]), "config_hash": r["config_hash"],
                    **{k: r[k] for k in ("record_id", "panel_id", "target_id", "donor_id", "runtime_id", "baseline_id", *FACTORS)},
                    **score})
    conditions = list(origins)
    values = np.empty((len(panels), len(conditions), len(METRICS), len(STAGES)))
    for pi, panel in enumerate(panels):
        for ci, condition in enumerate(conditions):
            group = grouped[panel, condition]
            expected = {c["cell_id"] for c in campaigns[0]["manifest"]["panels"][pi]["cells"]}
            require(set(group) == expected and len(group) == 4, "four matched targets required per panel/condition")
            values[pi, ci] = np.mean(list(group.values()), axis=0)
            if condition[-1] == "left":
                np.testing.assert_array_equal(values[pi, ci, 1], values[pi, ci, 2])
                np.testing.assert_array_equal(values[pi, ci, 5], np.zeros(3))
    return panels, conditions, values, trials


def bootstrap_stats(values: np.ndarray, indices: np.ndarray) -> dict:
    sampled = values[indices].mean(axis=1)
    lo, hi = np.quantile(sampled, [.025, .975], axis=0)
    return {"mean": values.mean(axis=0), "se": sampled.std(axis=0, ddof=1), "ci_low": lo, "ci_high": hi}


def condition_stats(values, indices):
    # One condition at a time keeps shared-login-node memory bounded.
    return [bootstrap_stats(values[:, ci], indices) for ci in range(values.shape[1])]


def reproduce_history(campaigns, panels, conditions, values, stats) -> dict:
    """Compare all overlapping saved tables, including covariance-sensitive gaps."""
    count, max_error = 0, 0.
    files, paired_cache = {}, {}
    indices = np.random.default_rng(42).integers(0, len(panels), (20000, len(panels)))
    def check(actual, expected):
        nonlocal count, max_error
        error = abs(float(actual) - float(expected))
        require(np.isfinite(error) and error <= 1e-12, f"historical statistic differs by {error}")
        count += 1
        max_error = max(max_error, error)
    for campaign in campaigns:
        for filename in HISTORY_FILES[3:]:
            path = campaign["root"] / filename
            if not path.exists():
                continue
            before = count
            with path.open() as source:
                for row in csv.DictReader(source):
                    if path.name == "paired_comparisons.csv":
                        first = tuple(row["first"] if f == row["factor"] else row[f] for f in FACTORS)
                        second = tuple(row["second"] if f == row["factor"] else row[f] for f in FACTORS)
                        key = first, second
                        if key not in paired_cache:
                            paired_cache[key] = bootstrap_stats(values[:, conditions.index(first)] -
                                                                values[:, conditions.index(second)], indices)
                        result = paired_cache[key]
                        mi = METRICS.index(row["candidate"])
                        for name, statistic in (("mean_paired_difference", "mean"), ("ci_low", "ci_low"), ("ci_high", "ci_high")):
                            check(result[statistic][mi, 2], row[name])
                        continue
                    condition = tuple(row[f] for f in FACTORS)
                    ci = conditions.index(condition)
                    candidate = row.get("candidate", "donor_sum")
                    metric = ("donor_minus_mixed" if path.name == "donor_vs_mixed_one_sigma.csv" else
                              candidate.removesuffix("_sum") + "_minus_target" if "mean_clean_gap" in row else candidate)
                    mi = METRICS.index(metric)
                    if "panel_id" in row:
                        pi = panels.index(row["panel_id"])
                        for si, name in enumerate(("mean_clean_logprob", "mean_patched_logprob", "mean_delta_logprob")):
                            if name in row:
                                check(values[pi, ci, mi, si], row[name])
                    else:
                        for si, name in enumerate(("clean", "patched", "delta")):
                            key = "mean_" + name + "_logprob"
                            if key in row:
                                check(stats[ci]["mean"][mi, si], row[key])
                        for si, name in enumerate(("clean_gap", "patched_gap", "gap_shift")):
                            for prefix, suffix, statistic in (("mean_", "", "mean"), ("", "_bootstrap_se_1sigma", "se"),
                                                              ("", "_ci95_low", "ci_low"), ("", "_ci95_high", "ci_high")):
                                key = prefix + name + suffix
                                if key in row:
                                    check(stats[ci][statistic][mi, si], row[key])
                        for name, statistic in (("bootstrap_se_1sigma", "se"), ("ci_low", "ci_low"), ("ci_high", "ci_high")):
                            if name in row:
                                check(stats[ci][statistic][mi, 2], row[name])
                        for name, divisor in (("panel_sd", 1), ("analytic_panel_se", np.sqrt(len(panels)))):
                            if name in row:
                                check(values[:, ci, mi, 2].std(ddof=1) / divisor, row[name])
            files[str(path)] = count - before
    return {"passed": True, "absolute_tolerance": 1e-12, "statistics_checked": count,
            "maximum_absolute_error": max_error, "files": files}


def historical_preflight(manifest: dict, roots: list[Path]):
    from scripts.dsv4.one_fact_patching import verify_reused_inputs
    campaigns = [read_campaign(root) for root in roots]
    for campaign in campaigns:
        verify_reused_inputs(manifest, campaign["manifest"], json.loads((campaign["root"] / "COMPLETE.json").read_text()))
    panels, conditions, values, _ = panel_values(campaigns)
    indices = np.random.default_rng(42).integers(0, len(panels), (20000, len(panels)))
    stats = condition_stats(values, indices)
    reproduction = reproduce_history(campaigns, panels, conditions, values, stats)
    verify_unchanged(campaigns)
    return {"passed": True, "panels_prompts_tokenizer_checkpoint_equal": True,
            "historical_reproduction": reproduction,
            "campaigns": [c["provenance"] for c in campaigns]}


def flat_stats(stats, mi):
    return {f"{stage}_{name}": float(stats[name][mi, si]) for si, stage in enumerate(STAGES)
            for name in ("mean", "se", "ci_low", "ci_high")}


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as sink:
        writer = csv.DictWriter(sink, fields)
        writer.writeheader()
        writer.writerows(rows)


def verify_unchanged(campaigns):
    for campaign in campaigns:
        for path, checksum in campaign["provenance"]["input_sha256"].items():
            require(file_digest(Path(path)) == checksum, "input changed during comparison")


def compare(root: Path, *, history_roots: list[Path] | None = None, integrity: dict | None = None) -> dict:
    from scripts.dsv4.one_fact_patching import DEFAULT_ROOT, COVERAGE_ROOT, verify_reused_inputs
    roots = history_roots if history_roots is not None else [DEFAULT_ROOT, COVERAGE_ROOT]
    require(root.resolve() not in {p.resolve() for p in roots}, "comparison output must use a new campaign root")
    current = read_campaign(root, integrity=integrity)
    require(current["manifest"].get("design") == "filler10-coverage", "compare requires filler10-coverage")
    history = [read_campaign(path) for path in roots]
    for campaign in history:
        verify_reused_inputs(current["manifest"], campaign["manifest"], json.loads((campaign["root"] / "COMPLETE.json").read_text()))
    campaigns = [*history, current]
    panels, conditions, values, trials = panel_values(campaigns)
    require(len(panels) == 24, "24 panels required")
    selected = list(product(*LEVELS))
    require({c for c in conditions if c[0] in LEVELS[0]} == set(selected), "incomplete three-range comparison design")
    indices = np.random.default_rng(42).integers(0, len(panels), (20000, len(panels)))
    stats = condition_stats(values, indices)
    reproduction = reproduce_history(history, panels, conditions, values, stats)
    rows, panel_rows, contrasts = [], [], []
    for condition in selected:
        ci = conditions.index(condition)
        for mi, metric in enumerate(METRICS):
            rows.append({**dict(zip(FACTORS, condition)), "metric": metric, **flat_stats(stats[ci], mi)})
            for pi, panel in enumerate(panels):
                panel_rows.append({"panel_id": panel, **dict(zip(FACTORS, condition)), "metric": metric,
                                   **dict(zip(STAGES, map(float, values[pi, ci, mi])))})
    for first, second in combinations(selected, 2):
        axes = [i for i in range(len(FACTORS)) if first[i] != second[i]]
        if len(axes) != 1:
            continue
        axis = axes[0]
        # Subtract aligned panel values BEFORE drawing shared bootstrap samples.
        paired = values[:, conditions.index(first)] - values[:, conditions.index(second)]
        result = bootstrap_stats(paired, indices)
        for mi, metric in enumerate(METRICS):
            contrasts.append({"factor": FACTORS[axis], "first": first[axis], "second": second[axis],
                **{f: first[i] for i, f in enumerate(FACTORS) if i != axis},
                "metric": metric, **flat_stats(result, mi)})
    output = root / "comparison"
    output.mkdir(parents=True, exist_ok=True)
    selected_trials = [r for r in trials if r["site"] in LEVELS[0]]
    for name, data in (("conditions", rows), ("panel_means", panel_rows), ("matched_contrasts", contrasts),
                       ("trial_scores", selected_trials)):
        write_csv(output / f"{name}.csv", data)
    require(len(selected_trials) == 2304 * 3, "expected 2304 selected substantive trials")
    report = {"passed": True, "generated_at": datetime.now(timezone.utc).isoformat(),
        "panels": 24, "targets_per_condition": 96, "selected_trials": len(selected_trials) // 3,
        "conditions": len(selected), "contrast_pairs": len(contrasts) // len(METRICS),
        "bootstrap": {"resamples": 20000, "seed": 42, "unit": "panel", "shared_indices": True, "ddof": 1,
            "error": "±1 sigma standard error of bootstrap means", "interval": "descriptive percentile 95%; no multiplicity adjustment",
            "limitation": "errors do not separately estimate between-runtime variability",
            "indices_sha256": digest(indices.tolist())},
        "historical_reproduction": reproduction, "campaigns": [c["provenance"] for c in campaigns],
        "panels_prompts_tokenizer_checkpoint_equal": True,
        "code_sha256": {str(Path(__file__).resolve()): file_digest(Path(__file__))}}
    write_reports(output, rows, contrasts, report)
    verify_unchanged(campaigns)
    atomic_json(output / "validation.json", report)
    command = f"OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-sglang/bin/python -m scripts.dsv4.one_fact_patching compare --root {shlex.quote(str(root))}\n"
    atomic_bytes(output / "REPRODUCE.sh", ("#!/usr/bin/env bash\nset -euo pipefail\ncd " +
        shlex.quote(str(Path(__file__).resolve().parents[2])) + "\n" + command).encode())
    return report


def write_reports(output, rows, contrasts, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    intro = ["All 24 panels and 96 targets per condition are retained without correctness filtering. "
        "Scores use full-vocabulary natural-log probabilities (nats). Target = A_t+X_t, mixed = A_d+X_t, donor = A_d+X_d. "
        "Donor `left` changes the fact with the same addend; `both` changes both.", "",
        "Entries are mean ±1σ panel-bootstrap standard errors (ddof=1); brackets are descriptive 95% percentile intervals. "
        "All tables and contrasts share 20,000 whole-panel resamples, seed 42. Gaps are formed within trials; four targets "
        "are averaged per panel; matched condition differences are formed within panels before bootstrapping. "
        "Errors do not separately estimate between-runtime variability; no multiplicity adjustment.", "",
        "Sites are zero-based and layer ranges inclusive. Complete post-block mHC residuals are replaced on every TP rank. "
        "Answer-only restores all other rows except the final answer-prediction row after every block. "
        "For layers 32–37, the selected filler and answer row continue evolving through layer 42. "
        "Layers 0–42 and 33–42 reuse the original campaign; filler_5/32–37 reuses the coverage campaign. "
        "Every trial retains its original runtime/mode baseline. Same-addend donor and mixed candidates are exact aliases.", ""]
    def entry(r, stage):
        return (f"{r[stage + '_mean']:+.4f} ± {r[stage + '_se']:.4f} "
                f"[{r[stage + '_ci_low']:+.4f}, {r[stage + '_ci_high']:+.4f}]")
    for site in LEVELS[0]:
        lines = [f"# {site}: three layer ranges", "", *intro,
            "| Layers | Mode | Donor | Candidate | Clean log p | Patched log p | Patched − clean |",
            "|---|---|---|---|---:|---:|---:|"]
        for r in rows:
            if r["site"] == site and r["metric"] in METRICS[:3]:
                lines.append("| " + " | ".join([r[f] for f in FACTORS[1:]] + [r["metric"]] +
                    [entry(r, stage) for stage in STAGES]) + " |")
        atomic_bytes(output / f"{site.upper()}_THREE_RANGES.md", ("\n".join(lines) + "\n").encode())
    lines = ["# Paired logit gaps", "", *intro,
        "A gap is log p(first) − log p(second), equal to their logit difference because normalization cancels. "
        "Positive gaps favor the first candidate; positive shifts need not make the patched gap positive.", "",
        "| Site | Layers | Mode | Donor | Gap | Clean | Patched | Shift |", "|---|---|---|---|---|---:|---:|---:|"]
    for r in rows:
        if r["metric"] in METRICS[3:]:
            lines.append("| " + " | ".join([r[f] for f in FACTORS] + [r["metric"]] + [entry(r, s) for s in STAGES]) + " |")
    atomic_bytes(output / "LOGIT_GAPS.md", ("\n".join(lines) + "\n").encode())
    lines = ["# Matched contrasts: first minus second", "", *intro,
        "Each row changes only the named factor. Clean and patched columns compare the respective scores/gaps; "
        "the shift column compares intervention effects.", "",
        "| Factor | First − second | Fixed factors | Metric | Clean difference | Patched difference | Shift difference |",
        "|---|---|---|---|---:|---:|---:|"]
    for r in contrasts:
        fixed = ", ".join(f"{f}={r[f]}" for f in FACTORS if f != r["factor"])
        lines.append("| " + " | ".join([r["factor"], r["first"] + " − " + r["second"], fixed, r["metric"]] +
            [entry(r, s) for s in STAGES]) + " |")
    atomic_bytes(output / "MATCHED_CONTRASTS.md", ("\n".join(lines) + "\n").encode())
    with plt.rc_context({"text.usetex": False, "font.family": "DejaVu Sans"}):
        for name, metrics in (("candidate_changes", METRICS[:3]), ("gap_shifts", METRICS[3:])):
            fig, axes = plt.subplots(1, 3, figsize=(18, 10), sharey=True)
            for metric, ax in zip(metrics, axes):
                selected = [r for r in rows if r["metric"] == metric]
                y = np.arange(len(selected))
                ax.hlines(y, [r["shift_ci_low"] for r in selected], [r["shift_ci_high"] for r in selected], color="tab:blue")
                ax.errorbar([r["shift_mean"] for r in selected], y, xerr=[r["shift_se"] for r in selected],
                            fmt="o", color="tab:blue", capsize=3)
                ax.set_yticks(y, [" / ".join(r[f] for f in FACTORS) for r in selected], fontsize=8)
                ax.set_title(metric.replace("_", " "))
                ax.set_xlabel("Paired shift (nats); caps ±1σ, line 95% interval")
                ax.axvline(0, color="gray", lw=1)
            axes[0].invert_yaxis()
            fig.tight_layout()
            for ext in ("png", "pdf"):
                fig.savefig(output / f"{name}.{ext}", dpi=180)
            plt.close(fig)
    interpretation = ["# Interpretation", "", *intro]
    for donor in ("left", "both"):
        subset = [r for r in rows if r["site"] == "filler_10" and r["donor_role"] == donor and r["metric"] == "donor_sum"]
        effects = [r["shift_mean"] for r in subset]
        positive = sum(r["shift_ci_low"] > 0 for r in subset)
        interpretation += [f"For filler_10 with donor `{donor}`, mean donor-sum changes span {min(effects):+.4f} to "
            f"{max(effects):+.4f} nats across the six range/mode conditions; {positive}/6 descriptive 95% intervals lie above zero.", ""]
    for metric in ("donor_sum", "donor_minus_mixed"):
        subset = [r for r in contrasts if r["factor"] == "site" and r["metric"] == metric and r["donor_role"] == "both"]
        interpretation += [f"For different-addend donors, filler_5 minus filler_10 shifts in `{metric}` span "
            f"{min(r['shift_mean'] for r in subset):+.4f} to {max(r['shift_mean'] for r in subset):+.4f} nats; "
            f"{sum(r['shift_ci_low'] > 0 for r in subset)}/6 intervals lie above zero and "
            f"{sum(r['shift_ci_high'] < 0 for r in subset)}/6 lie below zero.", ""]
    interpretation += ["A donor-over-mixed shift measures preference for the complete donor sum beyond the mixed sum. "
        "Whole-residual replacements cannot distinguish a stored sum from operand or other answer-related features "
        "used downstream. Simultaneous multi-layer interventions do not localize a representation to one layer. "
        "Intervals containing zero do not establish zero effect. Cross-runtime contrasts include any runtime differences.", ""]
    atomic_bytes(output / "INTERPRETATION.md", "\n".join(interpretation).encode())
    lines = ["# Complete filler_10 activation-patching comparison", "", *intro,
        "- [filler_10 three-range scores](FILLER_10_THREE_RANGES.md)",
        "- [filler_5 reference](FILLER_5_THREE_RANGES.md)",
        "- [Target-relative gaps and direct donor versus mixed](LOGIT_GAPS.md)",
        "- [Matched mode, layer, donor and position contrasts](MATCHED_CONTRASTS.md)",
        "- [Interpretation](INTERPRETATION.md)",
        "- Full-precision CSVs: [conditions](conditions.csv), [panels](panel_means.csv), [contrasts](matched_contrasts.csv), [trial provenance](trial_scores.csv)",
        "- [Validation and source hashes](validation.json); [reproducible command](REPRODUCE.sh)", "",
        f"Historical reproduction checked {report['historical_reproduction']['statistics_checked']:,} scalar statistics "
        "to absolute tolerance 1e-12. Raw responses were rechecked; historical tensor validation was reused from completed campaigns.", "",
        "![Candidate changes](candidate_changes.png)", "[PDF](candidate_changes.pdf)", "",
        "![Logit gap shifts](gap_shifts.png)", "[PDF](gap_shifts.pdf)", ""]
    atomic_bytes(output / "REPORT.md", "\n".join(lines).encode())
