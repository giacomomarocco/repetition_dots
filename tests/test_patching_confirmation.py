"""Full confirmation membership, automatic progression, resume and paired reporting."""
import copy
import json

import numpy as np
import pytest

from filler.dsv4.patching import build_manifest, layer42_diagnostics
from scripts.dsv4.one_fact_patching import validate_confirmation_inputs
from test_one_fact_patching import Tokenizer, rendered, FakeTransport, make_campaign


def confirmation_inputs():
    template = rendered()["eligible_panels"][0]
    panels, canonical = [], []
    for i in range(60):
        panel = copy.deepcopy(template)
        pid = f"confirmation-filler-{i:020}"
        panel["panel_id"] = pid
        for cell in panel["cells"]:
            cell.update(panel_id=pid, cell_id=f"{pid}:{cell['row']}{cell['col']}",
                        split="confirmation", kind="one_fact", left_id=f"fact-{i}-{cell['row']}")
            canonical.append({k: v for k, v in cell.items() if k not in {"input_ids", "rendered_prompt"}})
        panels.append(panel)
    return {"filler_length": 20, "split": "confirmation", "eligible_panels": panels}, {"cells": canonical}


def test_full_confirmation_membership_and_exact_design():
    r, design = confirmation_inputs()
    discovery = {"eligible_panels": [{"cells": [{"left_id": "discovery-only"}]}]}
    assert validate_confirmation_inputs(r, design, discovery)["distinct_facts"] == 120
    m = build_manifest(r, Tokenizer(), "filler10-confirmation")
    assert m["counts"] == dict(panels=60, targets=240, trials=2880, identity=1440, baselines=480)
    assert m["pilot_counts"] == dict(targets=4, trials=48, identity=24, baselines=8)
    assert m["diagnostics_per_runtime"] == 16
    assert {s["site"] for s in m["trials"]} == {"filler_10"}
    assert {tuple(s["layers"]) for s in m["trials"]} == {tuple(range(43)), tuple(range(32,38)), tuple(range(33,43))}
    assert {tuple(s["positions"]) for s in m["trials"]} == {(21,)}
    assert len({s["record_id"] for s in m["trials"] + m["identity_controls"]}) == 4320
    assert m["split"] == "confirmation" and m["pilot_panel"] == m["panels"][0]["panel_id"]
    for mutate in (lambda x: x["eligible_panels"].pop(),
                   lambda x: x["eligible_panels"].__setitem__(1, copy.deepcopy(x["eligible_panels"][0])),
                   lambda x: x.update(rejected_panel_count=1)):
        bad = copy.deepcopy(r); mutate(bad)
        with pytest.raises(ValueError):
            build_manifest(bad, Tokenizer(), "filler10-confirmation")
    with pytest.raises(ValueError, match="discovery"):
        build_manifest(r, Tokenizer())
    with pytest.raises(ValueError, match="confirmation"):
        build_manifest(rendered(), Tokenizer(), "filler10-confirmation")
    bad = copy.deepcopy(r); bad["eligible_panels"][0]["cells"][0]["left_value"] += 1
    with pytest.raises(ValueError, match="metadata"):
        validate_confirmation_inputs(bad, design, discovery)
    with pytest.raises(ValueError, match="disjoint"):
        validate_confirmation_inputs(r, design, {"eligible_panels": [{"cells": [{"left_id": "fact-0-0"}]}]})


def small_manifest():
    m = build_manifest(confirmation_inputs()[0], Tokenizer(), "filler10-confirmation")
    m["panels"] = m["panels"][:2]
    ids = {p["panel_id"] for p in m["panels"]}
    for key in ("trials", "identity_controls"):
        m[key] = [s for s in m[key] if s["panel_id"] in ids]
    return m


@pytest.mark.parametrize("fail_at", [10, 50, 115])
def test_confirmation_resume_preserves_baselines_and_validates_pilot(tmp_path, fail_at):
    m = small_manifest()
    first = make_campaign(m, tmp_path, "old", FakeTransport(m, fail_at=fail_at))
    with pytest.raises(InterruptedError):
        first.run()
    saved = copy.deepcopy(first.journal.records)
    second = make_campaign(m, tmp_path, "new", FakeTransport(m))
    assert second.run() == dict(trials=96, identity=48, matched_baseline_cells=16, passed=True)
    assert all(second.journal.records[k] == v for k, v in saved.items())
    records = list(second.journal.records.values())
    assert any(r["kind"] == "pilot_passed" for r in records)
    for r in records:
        if r["kind"] == "trial":
            assert second.journal.records[r["baseline_id"]]["runtime_id"] == r["runtime_id"]
    assert len(second.transport.controls) == second.planned_passes


def test_confirmation_reports_shared_panel_se_and_completion(tmp_path, monkeypatch):
    from filler.dsv4.patching_analysis import summarize
    from scripts.dsv4 import one_fact_patching as cli
    m = small_manifest()
    class Varied(FakeTransport):
        def run(self, control, input_ids, scored_ids):
            raw, acks = super().run(control, input_ids, scored_ids)
            if control["layers"] and control["layers"] != [42] and control["donor_capture"]["cell_id"] != control["cell_id"]:
                if self.cells[control["cell_id"]]["panel_id"] != m["pilot_panel"]:
                    for row in raw["meta_info"]["output_token_ids_logprobs"][0]:
                        row[0] += .2
            return raw, acks
    campaign = make_campaign(m, tmp_path / "run", "r1", Varied(m))
    integrity = campaign.run()
    records = list(campaign.journal.records.values())
    pilot_index = next(i for i,r in enumerate(records) if r["kind"] == "pilot_passed")
    assert sum(r["kind"] == "trial" for r in records[:pilot_index]) == 48
    assert sum(r["kind"] == "identity" for r in records[:pilot_index]) == 24
    summary = summarize(m, campaign.journal, tmp_path / "analysis", resamples=200, seed=42)
    draws = np.random.default_rng(42).integers(0, 2, (200, 2))
    se = np.array([.5,.7])[draws].mean(axis=1).std(ddof=1)
    assert len(summary["conditions"]) == 36
    assert summary["split"] == "confirmation"
    assert all(row["bootstrap_se"] == pytest.approx(se) for row in summary["conditions"])
    assert all(row["bootstrap_se"] == pytest.approx(0, abs=1e-14) for row in summary["paired_comparisons"])
    for name in ("DEEPSEEK_V4_LOGIT_LENS.md", "ONE_FACT_PATCHING_HANDOFF.md"):
        (tmp_path / name).write_text("# Test lab\n")
    monkeypatch.setattr(cli, "WORKSPACE", tmp_path)
    cli.finish(m, campaign.journal, tmp_path / "run", integrity=integrity)
    assert json.loads((tmp_path / "run/COMPLETE.json").read_text())["status"] == "complete"
    assert "confirmation completed" in (tmp_path / "DEEPSEEK_V4_LOGIT_LENS.md").read_text()
    assert "±" in (tmp_path / "run/analysis/REPORT.md").read_text()


def test_confirmation_cli_uses_separate_root():
    from scripts.dsv4.one_fact_patching import CONFIRMATION_ROOT, parse_args
    assert parse_args(["prepare", "--design", "filler10-confirmation"]).root == CONFIRMATION_ROOT
    assert parse_args(["run", "--root", str(CONFIRMATION_ROOT)]).root == CONFIRMATION_ROOT
