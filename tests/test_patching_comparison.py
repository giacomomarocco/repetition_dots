import copy
import csv
import json
from itertools import product
from pathlib import Path

import numpy as np
import pytest

from test_one_fact_patching import FakeTransport, manifest, make_campaign
from filler.dsv4.patching import atomic_json, digest, file_digest, layer42_diagnostics
from filler.dsv4.patching_comparison import (
    FACTORS, LEVELS, METRICS, bootstrap_stats, compare, panel_values,
    read_campaign, reproduce_history, trial_metrics, verify_unchanged,
)


def test_filler10_design_membership_and_cli():
    from scripts.dsv4.one_fact_patching import FILLER10_ROOT, parse_args
    m = manifest(design="filler10-coverage")
    assert m["panels"] == manifest()["panels"] == manifest(design="filler-coverage")["panels"]
    assert m["counts"] == dict(panels=24, targets=96, trials=384, identity=192, baselines=192)
    assert m["pilot_counts"] == dict(targets=4, trials=16, identity=8, baselines=8)
    specs = m["trials"] + m["identity_controls"]
    assert len({s["record_id"] for s in specs}) == 576
    assert all(s["site"] == "filler_10" and s["positions"] == [21] and s["layers"] == list(range(32, 38)) for s in specs)
    assert {s["donor_role"] for s in m["trials"]} == {"left", "both"}
    assert len(layer42_diagnostics(m, m["pilot_panel"])) == 16
    assert parse_args(["prepare", "--design", "filler10-coverage"]).root == FILLER10_ROOT
    assert parse_args(["compare", "--root", str(FILLER10_ROOT)]).root == FILLER10_ROOT
    with pytest.raises(SystemExit):
        parse_args(["compare"])


def test_read_only_reader_validates_raw_scores_and_refuses_tail_repair(tmp_path):
    m = manifest(two_panels=True, design="filler10-coverage")
    m["config_hash"] = digest({k: v for k, v in m.items() if k != "config_hash"})
    atomic_json(tmp_path / "manifest.json", m)
    campaign = make_campaign(m, tmp_path, "r1", FakeTransport(m))
    integrity = campaign.run()
    atomic_json(tmp_path / "COMPLETE.json", dict(status="complete", config_hash=m["config_hash"], integrity=integrity))
    before = {p.name: (p.stat().st_mtime_ns, file_digest(p)) for p in tmp_path.iterdir() if p.is_file()}
    saved = read_campaign(tmp_path)
    assert saved["provenance"]["raw_responses_checked"] == 64
    assert saved["provenance"]["same_addend_aliases_checked"] == 16
    assert before == {p.name: (p.stat().st_mtime_ns, file_digest(p)) for p in tmp_path.iterdir() if p.is_file()}
    verify_unchanged([saved])
    raw_path = Path(next(r for r in campaign.journal.records.values() if r["kind"] == "trial")["response"]["path"])
    raw_path.write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        read_campaign(tmp_path)
    with (tmp_path / "results.jsonl").open("ab") as sink:
        sink.write(b'{"incomplete":')
    broken = (tmp_path / "results.jsonl").read_bytes()
    with pytest.raises(ValueError, match="refusing repair"):
        read_campaign(tmp_path)
    assert (tmp_path / "results.jsonl").read_bytes() == broken
    assert not list(tmp_path.glob("interrupted-tail*"))


def test_gap_and_shared_bootstrap_preserve_covariance_and_aliases():
    samples = []
    for panel in range(24):
        # Large perfectly correlated candidate movements, constant relative shift.
        scores = [dict(label=label, clean_logprob=-50 + panel + offset,
                       patched_logprob=-49 + 2 * panel + offset + effect,
                       delta_logprob=1 + panel + effect)
                  for label, offset, effect in (("target_sum", 0, 0), ("mixed_sum", -3, .5), ("donor_sum", -2, .75))]
        samples.append(trial_metrics(scores))
    values = np.array(samples)
    indices = np.random.default_rng(42).integers(0, 24, (20000, 24))
    result = bootstrap_stats(values, indices)
    assert result["se"][0, 2] > 1
    np.testing.assert_array_equal(result["se"][3:, 2], np.zeros(3))
    np.testing.assert_array_equal(result["mean"][3:, 2], [.75, .5, .25])
    scores[2] = {**scores[1], "label": "donor_sum"}
    np.testing.assert_array_equal(trial_metrics(scores)[5], np.zeros(3))


def synthetic_campaign(root, design):
    m = manifest(design=design)
    records = {}
    for spec in m["trials"]:
        pi = next(i for i, p in enumerate(m["panels"]) if p["panel_id"] == spec["panel_id"])
        scope = .25 if spec["site"] == "filler_10" else .5
        layer = .1 if spec["layer_set"] == "32-37" else .2
        mode = 1 if spec["recomputation"] == "full_downstream" else 2
        r = {**spec, "config_hash": m["config_hash"], "runtime_id": root.name,
             "baseline_id": root.name + spec["target_id"] + spec["recomputation"], "scores": []}
        for c in spec["candidates"]:
            clean = -10 - c["token_id"] / 1000 + pi / 20
            effect = (scope + layer) * mode * (pi + 1) / 50 + c["token_id"] / 100000
            r["scores"].append({**c, "clean_logprob": clean, "patched_logprob": clean + effect,
                                "delta_logprob": (clean + effect) - clean})
        records[spec["record_id"]] = r
    return {"root": root, "manifest": m, "records": records,
            "provenance": {"root": str(root), "input_sha256": {}}}


def test_full_comparison_outputs_matched_contrasts_and_historical_reproduction(tmp_path, monkeypatch):
    roots = [tmp_path / name for name in ("original", "coverage", "new")]
    campaigns = [synthetic_campaign(root, design) for root, design in zip(roots, ("original", "filler-coverage", "filler10-coverage"))]
    for c in campaigns:
        c["root"].mkdir()
        atomic_json(c["root"] / "COMPLETE.json", {})
    monkeypatch.setattr("filler.dsv4.patching_comparison.read_campaign", lambda root, **kw: campaigns[roots.index(root)])
    monkeypatch.setattr("scripts.dsv4.one_fact_patching.verify_reused_inputs", lambda *args: None)
    report = compare(roots[2], history_roots=roots[:2])
    assert report["conditions"] == 24 and report["contrast_pairs"] == 60
    assert report["selected_trials"] == 2304
    assert report["bootstrap"]["ddof"] == 1 and report["bootstrap"]["resamples"] == 20000
    output = roots[2] / "comparison"
    with (output / "matched_contrasts.csv").open() as source:
        contrasts = list(csv.DictReader(source))
    assert len(contrasts) == 360
    assert {r["factor"] for r in contrasts} == set(FACTORS)
    panels, conditions, values, _ = panel_values(campaigns)
    indices = np.random.default_rng(42).integers(0, 24, (20000, 24))
    r = next(r for r in contrasts if r["factor"] == "site" and r["metric"] == "donor_sum")
    first = tuple(r["first"] if f == "site" else r[f] for f in FACTORS)
    second = tuple(r["second"] if f == "site" else r[f] for f in FACTORS)
    expected = bootstrap_stats(values[:, conditions.index(first)] - values[:, conditions.index(second)], indices)
    for name in ("mean", "se", "ci_low", "ci_high"):
        assert float(r["shift_" + name]) == pytest.approx(expected[name][2, 2], abs=1e-12)
    for name in ("candidate_changes.png", "candidate_changes.pdf", "gap_shifts.png", "gap_shifts.pdf",
                 "FILLER_10_THREE_RANGES.md", "FILLER_5_THREE_RANGES.md", "LOGIT_GAPS.md",
                 "MATCHED_CONTRASTS.md", "INTERPRETATION.md", "validation.json", "REPRODUCE.sh"):
        assert (output / name).stat().st_size > 0
    assert not (roots[0] / "comparison").exists()
    assert not (roots[1] / "comparison").exists()
    assert "between-runtime variability" in (output / "REPORT.md").read_text()
    # Exercise historical table reproduction and intentional corruption.
    from filler.dsv4.patching_comparison import condition_stats
    stats = condition_stats(values, indices)
    prior = roots[0] / "analysis"
    prior.mkdir()
    c = conditions[0]
    row = {**dict(zip(FACTORS, c)), "candidate": "target_sum", "mean_delta_logprob": stats[0]["mean"][0, 2],
           "ci_low": stats[0]["ci_low"][0, 2], "ci_high": stats[0]["ci_high"][0, 2]}
    from filler.dsv4.patching_comparison import write_csv
    write_csv(prior / "conditions.csv", [row])
    assert reproduce_history(campaigns[:1], panels, conditions, values, stats)["statistics_checked"] == 3
    write_csv(prior / "conditions.csv", [{**row, "mean_delta_logprob": 42}])
    with pytest.raises(ValueError, match="historical statistic"):
        reproduce_history(campaigns[:1], panels, conditions, values, stats)
    duplicate = copy.deepcopy(campaigns[0])
    with pytest.raises(ValueError, match="overlapping campaign"):
        panel_values([campaigns[0], duplicate])


def test_finish_automatically_compares_before_complete_and_failure_stops_marker(tmp_path, monkeypatch):
    from scripts.dsv4.one_fact_patching import finish
    from types import SimpleNamespace
    m = manifest(design="filler10-coverage")
    (tmp_path / "DEEPSEEK_V4_LOGIT_LENS.md").write_text(f"Campaign `{m['config_hash']}`")
    monkeypatch.setattr("scripts.dsv4.one_fact_patching.WORKSPACE", tmp_path)
    integrity = {"passed": True}
    monkeypatch.setattr("filler.dsv4.patching_analysis.summarize", lambda *args, **kw: {"integrity": integrity})
    def fail(*args, **kwargs):
        assert not (tmp_path / "COMPLETE.json").exists()
        raise ValueError("comparison failed")
    monkeypatch.setattr("filler.dsv4.patching_comparison.compare", fail)
    journal = SimpleNamespace(progress=lambda **kwargs: None)
    with pytest.raises(ValueError, match="comparison failed"):
        finish(m, journal, tmp_path)
    assert not (tmp_path / "COMPLETE.json").exists()
    monkeypatch.setattr("filler.dsv4.patching_comparison.compare", lambda *args, **kwargs: {"passed": True})
    finish(m, journal, tmp_path)
    assert json.loads((tmp_path / "COMPLETE.json").read_text())["comparison"]["passed"]
