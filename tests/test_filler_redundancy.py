import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from filler.dsv4.patching import (atomic_json, baseline_modes, build_manifest, campaign_counts,
    digest, file_digest, invariant, layer42_diagnostics, redundancy_interventions, score_candidates)
from filler.dsv4.patching_logits import IDENTITY, collect_logits, make_logits_hook, validate_logits
from test_one_fact_patching import (CausalLayer, FakeTransport, Tokenizer, make_campaign,
                                   rendered, run_model)


def test_design_counts_streams_masks_alignment_and_aliases():
    data = rendered()
    m = build_manifest(data, Tokenizer(), "filler-redundancy")
    assert m["counts"] == dict(panels=24, targets=96, trials=4992, identity=2496, baselines=96)
    assert m["pilot_counts"] == dict(targets=4, trials=208, identity=104, baselines=4)
    assert baseline_modes(m) == ("full_downstream",)
    assert m == build_manifest(data, Tokenizer(), "filler-redundancy")
    assert m != build_manifest(data, Tokenizer(), "filler-redundancy", seed=43)
    assert len({s["record_id"] for k in ("trials", "identity_controls") for s in m[k]}) == 7488
    old = build_manifest(data, Tokenizer())
    assert old["schema_version"] == 2 and old["counts"]["baselines"] == 192
    assert m["panels"] == old["panels"]
    for panel in m["panels"]:
        cells = {c["cell_id"]: c for c in panel["cells"]}
        masks = {}
        for s in [r for r in m["trials"] + m["identity_controls"] if r["panel_id"] == panel["panel_id"]]:
            key = s["family"], s["subset_size"], s["draw_id"]
            assert masks.setdefault(key, s["filler_indices"]) == s["filler_indices"]
            assert s["layers"] == list(range(43))
            assert s["positions"] == [panel["positions"]["fillers"][i] for i in s["filler_indices"]]
            if s["family"] == "early_block":
                assert s["filler_indices"] == list(range(5))
            else:
                assert len(set(s["filler_indices"])) == s["subset_size"]
                assert set(s["filler_indices"]) <= set(range(5, 20))
            t, d = cells[s["target_id"]], cells[s["donor_id"]]
            assert m["input_hashes"][t["cell_id"]]["input_ids"] == digest(t["input_ids"])
            if s["donor_role"] != "identity":
                assert t["row"] != d["row"]
                assert (t["col"] == d["col"]) == (s["donor_role"] == "left")
            if s["donor_role"] == "left":
                assert s["candidates"][1]["token_id"] == s["candidates"][2]["token_id"]
        assert len(masks) == 26
        # Changing draw count preserves individual independent streams.
        assert redundancy_interventions(panel["panel_id"], 6)[1:6] == redundancy_interventions(panel["panel_id"], 5)[1:6]
    masks = m["interventions_by_panel"][m["pilot_panel"]]
    assert any(not set(a["filler_indices"]) <= set(b["filler_indices"])
               for a in masks[1:6] for b in masks[6:11])
    # Force repeated random choices; draw IDs and observations must remain distinct.
    many = redundancy_interventions(m["pilot_panel"], draws=20)
    singles = [r for r in many if r["subset_size"] == 1]
    assert len({tuple(r["filler_indices"]) for r in singles}) < len(singles)
    assert len({r["site"] for r in singles}) == 20
    diagnostics = layer42_diagnostics(m, m["pilot_panel"])
    assert len({d["record_id"] for d in diagnostics}) == len(diagnostics)
    assert all(s["layers"] == [42] for s in diagnostics)
    with pytest.raises(ValueError, match="draw count"):
        build_manifest(data, Tokenizer(), "filler-redundancy", draws=0)


def raw_response(logits, ids):
    lp = torch.log_softmax(logits.float(), -1)
    return {"output_ids": [int(logits.argmax())], "meta_info": {"cached_tokens": 0,
        "output_token_ids_logprobs": [[[float(lp[t]), t, None] for t in ids]]}}


def write_raw(control, response, logits):
    for rank in range(4):
        atomic_json(Path(control["output_root"]) / f"logits.rank{rank}.json", {
            **{k: control[k] for k in IDENTITY}, "rank": rank,
            "answer_position": control["num_tokens"] - 1, "vocab_size": len(logits),
            "dtype": str(logits.dtype), "logits": logits[control["scored_token_ids"]].tolist(),
            "log_normalizer": float(torch.logsumexp(logits.float(), -1)), "argmax": int(logits.argmax())})
    return collect_logits(control, response)


def test_logits_hook_nonmutation_identity_and_artifacts(tmp_path):
    root = tmp_path / "control"
    ids = [2, 7]
    control = dict(request_id="request", runtime_id="runtime", config_hash="hash", cell_id="target",
                   num_tokens=3, input_ids_hash=digest([1, 2, 3]), scored_token_ids=ids,
                   raw_logits=True, output_root=str(tmp_path / "pass"))
    logits = torch.arange(10, dtype=torch.float32).reshape(1, 10)
    output = SimpleNamespace(next_token_logits=logits, extra=object())
    before, extra = logits.clone(), output.extra
    hook = make_logits_hook({"control_root": str(root)})
    assert hook(SimpleNamespace(vocab_size=10), (torch.tensor([1, 2, 3]),), output) is output
    atomic_json(root / "NEXT.json", control)
    atomic_json(root / "acks/request.rank0.json", control)
    with pytest.raises(RuntimeError, match="input identity"):
        hook(SimpleNamespace(vocab_size=10), (torch.tensor([3, 2, 1]),), output)
    assert hook(SimpleNamespace(vocab_size=10), (torch.tensor([1, 2, 3]),), output) is output
    assert torch.equal(logits, before) and output.extra is extra
    with pytest.raises(RuntimeError, match="stale"):
        hook(SimpleNamespace(vocab_size=10), (torch.tensor([1, 2, 3]),), output)
    response = raw_response(logits[0], ids)
    saved = json.loads((tmp_path / "pass/logits.rank0.json").read_text())
    assert saved["logits"] == [2., 7.]
    response = write_raw(control, response, logits[0])
    assert validate_logits(response, control)["log_normalizer"] == pytest.approx(float(torch.logsumexp(logits[0], 0)))
    bad = copy.deepcopy(response)
    bad["meta_info"]["output_token_ids_logprobs"][0][0][0] += .01
    with pytest.raises(ValueError, match="inconsistency"):
        validate_logits(bad, control)
    with pytest.raises(ValueError, match="stale"):
        validate_logits(response, {**control, "request_id": "stale"})
    bad = copy.deepcopy(response)
    bad["raw_logits_refs"].pop()
    with pytest.raises(ValueError, match="missing"):
        validate_logits(bad, control)
    path = Path(response["raw_logits_refs"][3]["path"])
    row = json.loads(path.read_text()); row["logits"][0] += 1e-6
    atomic_json(path, row)
    with pytest.raises(ValueError, match="checksum"):
        validate_logits(response, control)
    response["raw_logits_refs"][3]["sha256"] = file_digest(path)
    with pytest.raises(ValueError, match="across ranks"):
        validate_logits(response, control)


def test_logits_aliases_and_raw_shift_gate(tmp_path):
    ids = [1, 2]
    control = dict(request_id="r", runtime_id="rt", config_hash="h", cell_id="c", num_tokens=3,
                   input_ids_hash="x", scored_token_ids=ids, output_root=str(tmp_path / "a"))
    logits = torch.tensor([0., 1., 2.])
    a = write_raw(control, raw_response(logits, ids), logits)
    b = write_raw({**control, "output_root": str(tmp_path / "b")}, raw_response(logits + 1, ids), logits + 1)
    gate = invariant(a, b, ids)
    assert gate["max_logprob_error"] < 1e-6 and not gate["passed"]
    mapping = [{"label": label, "token_id": t, "value": t} for label, t in
               [("target_sum", 1), ("donor_sum", 2), ("mixed_sum", 2)]]
    scores = score_candidates(mapping, a, b)
    assert all(s["delta_logit"] == 1 for s in scores)
    assert scores[1]["patched_logit"] == scores[2]["patched_logit"]


def test_every_requested_layer_and_downstream_response():
    from filler.dsv4.factorial import NativeResidualHooks
    # Small values avoid overflow after 43 causal layers.
    class Layer(CausalLayer):
        def forward(self, hidden):
            return hidden + hidden.cumsum(0) * .01, "aux"
    model = SimpleNamespace(layers=torch.nn.ModuleList([Layer(i) for i in range(43)]))
    clean = run_model(model, torch.ones(24, 2, 3))
    donor = run_model(model, torch.ones(24, 2, 3) * 2)
    hooks = NativeResidualHooks(model)
    for positions in ([0, 1, 2, 3, 4], [5, 8, 14]):
        with hooks.transplant_layers(list(range(43)), positions, donor):
            patched = run_model(model, torch.ones(24, 2, 3))
        for lid in range(43):
            assert torch.equal(patched[lid][positions], donor[lid][positions])
            if lid > 0:
                assert not torch.equal(patched[lid][20:], clean[lid][20:])  # suffix responds starting at the next block
        with hooks.transplant_layers([42], positions, donor):
            diagnostic = run_model(model, torch.ones(24, 2, 3))
        assert torch.equal(diagnostic[42][-1], clean[42][-1])


class RawTransport(FakeTransport):
    def run(self, control, input_ids, scored_ids):
        _, acks = super().run(control, input_ids, scored_ids)
        logits = torch.full((1200,), -10.)
        logits[1040] = 3.
        patch = bool(control["layers"]) and control["layers"] != [42]
        identity = patch and control["donor_capture"]["cell_id"] == control["cell_id"]
        scale = .1 if "z001" in control["cell_id"] else .2
        logits[scored_ids] += scale * len(control["positions"]) if patch and not identity else 0.
        response = write_raw(control, raw_response(logits, scored_ids), logits)
        return response, acks


def reduced_manifest():
    m = build_manifest(rendered(), Tokenizer(), "filler-redundancy", draws=2)
    m["panels"] = m["panels"][:2]
    pids = {p["panel_id"] for p in m["panels"]}
    for key in ("trials", "identity_controls"):
        m[key] = [s for s in m[key] if s["panel_id"] in pids]
    return m


def test_progression_resume_exports_and_panel_bootstrap(tmp_path):
    from filler.dsv4.patching_analysis import summarize
    from filler.dsv4.patching_campaign import verify_records
    m = reduced_manifest()
    first_trial = 4 + 44 + len(layer42_diagnostics(m, m["pilot_panel"]))
    first = make_campaign(m, tmp_path, "old", RawTransport(m, fail_at=first_trial + 2))
    with pytest.raises(InterruptedError):
        first.run()
    old = next(r for r in first.journal.records.values() if r["kind"] == "trial")
    transport = RawTransport(m)
    second = make_campaign(m, tmp_path, "new", transport)
    result = second.run()
    assert result["matched_baseline_cells"] == 8
    assert result["trials"] == 176 and result["identity"] == 88
    assert second.journal.records[old["record_id"]] == old
    assert second.journal.records[old["baseline_id"]]["runtime_id"] == "old"
    assert len(transport.controls) == second.planned_passes
    assert all(c["recomputation"] == "full_downstream" for c in transport.controls)
    report = summarize(m, second.journal, tmp_path / "analysis", resamples=100, seed=42)
    assert report["bootstrap_unit"] == "panel" and all(r["panels"] == 2 for r in report["conditions"])
    assert len(report["conditions"]) == 12 * 27
    assert len(report["paired_comparisons"]) == 16 * 27
    assert {r["contrast"] for r in report["paired_comparisons"]} == {"adjacent_size", "early_minus_later5", "donor_type"}
    scores = list(csv.DictReader((tmp_path / "analysis/trial_scores.csv").open()))
    assert len(scores) == 176 * 3
    panels = list(csv.DictReader((tmp_path / "analysis/panel_means.csv").open()))
    for r in panels:
        if r["metric"] != "delta_logit:target_sum":
            continue
        rows = [x for x in scores if all(x[k] == r[k] for k in ("panel_id", "family", "subset_size", "donor_role")) and x["label"] == "target_sum"]
        assert float(r["mean"]) == pytest.approx(np.mean([float(x["delta_logit"]) for x in rows]))
        assert len(rows) == 4 * (2 if r["family"] == "later_subset" else 1)
    # Independently reconstruct shared panel resamples and paired differences.
    draws = np.random.default_rng(42).integers(0, 2, (100, 2))
    for row in report["conditions"]:
        panel_values = np.array([float(p["mean"]) for p in panels if p["metric"] == row["metric"]
                                 and p["family"] == row["family"] and int(p["subset_size"]) == row["subset_size"]
                                 and p["donor_role"] == row["donor_role"]])
        boot = panel_values[draws].mean(1)
        assert row["bootstrap_se"] == pytest.approx(boot.std(ddof=1), abs=1e-12)
        assert [row["ci_low"], row["ci_high"]] == pytest.approx(np.quantile(boot, [.025, .975]), abs=1e-12)
    assert any(r["bootstrap_se"] > 0 for r in report["conditions"])
    for unit in ("logit", "logprob", "logit_gap"):
        for ext in ("png", "pdf"):
            assert (tmp_path / f"analysis/{unit}_changes.{ext}").is_file()
    # Removal cannot silently fall back to historical probability-only scoring.
    trial = next(r for r in second.journal.records.values() if r["kind"] == "trial")
    path = Path(json.loads(Path(trial["response"]["path"]).read_text())["raw_logits_refs"][0]["path"])
    path.rename(path.with_suffix(".missing"))
    with pytest.raises(FileNotFoundError):
        verify_records(m, second.journal)


def test_launch_has_both_hooks_and_frozen_cli_defaults():
    from scripts.dsv4.one_fact_patching import REDUNDANCY_ROOT, launch_command, parse_args
    args = parse_args(["prepare", "--design", "filler-redundancy"])
    assert args.root == REDUNDANCY_ROOT and (args.draws, args.seed) == (5, 42)
    command = launch_command(Path("/tmp/control"), raw_logits=True)
    hooks = json.loads(command[command.index("--forward-hooks") + 1])
    assert len(hooks[0]["target_modules"]) == 43
    assert hooks[1]["target_modules"] == ["logits_processor"]
    assert len(json.loads(launch_command(Path("/tmp/c"))[-1])) == 1


def test_native_equivalence_rejects_raw_shift_even_with_equal_probabilities(tmp_path, monkeypatch):
    from filler.dsv4.lens import DeepseekV4LensWeights, project_logits
    from filler.dsv4.patching_campaign import native_equivalence
    torch.manual_seed(4)
    weights = DeepseekV4LensWeights(torch.randn(2, 6), torch.randn(2), torch.randn(1),
                                   torch.ones(3), torch.randn(10, 3))
    monkeypatch.setattr("filler.dsv4.lens.load_checkpoint_readout", lambda *a, **kw: weights)
    hidden = torch.randn(2, 3)
    logits = project_logits(hidden, weights)
    raw = raw_response(logits, [0, 9])
    raw["meta_info"].update(hidden_states=[hidden.flatten().tolist()], output_top_logprobs=[[]])
    raw["raw_logits"] = {"scored_token_ids": [0, 9], "logits": logits[[0, 9]].tolist(),
                         "log_normalizer": float(torch.logsumexp(logits, -1))}
    refs = {}
    for rank in range(4):
        path = tmp_path / f"native{rank}.pt"
        torch.save({"states": {42: hidden.unsqueeze(0)}}, path)
        refs[str(rank)] = {"path": str(path), "sha256": file_digest(path)}
    assert native_equivalence({"ranks": refs}, raw, tmp_path, [0, 9])["passed"]
    raw["raw_logits"]["logits"] = [v + 1 for v in raw["raw_logits"]["logits"]]
    raw["raw_logits"]["log_normalizer"] += 1
    gate = native_equivalence({"ranks": refs}, raw, tmp_path, [0, 9])
    assert gate["max_logprob_error"] < 1e-6 and not gate["passed"]


def test_collector_waits_for_rank_publication_and_rejects_timeout(tmp_path):
    import threading
    import time
    control = dict(request_id="delayed", runtime_id="rt", config_hash="h", cell_id="c", num_tokens=3,
                   input_ids_hash="x", scored_token_ids=[1, 2], output_root=str(tmp_path))
    logits = torch.tensor([0., 1., 2.])
    response = raw_response(logits, [1, 2])
    enriched = write_raw(control, response, logits)
    rank_path = Path(enriched["raw_logits_refs"][1]["path"])
    pending = rank_path.with_suffix('.pending')
    rank_path.rename(pending)
    def publish():
        time.sleep(.08)
        pending.rename(rank_path)
    thread = threading.Thread(target=publish)
    thread.start()
    try:
        assert collect_logits(control, response, wait_timeout=1) == enriched
    finally:
        thread.join()
    rank_path.rename(pending)
    with pytest.raises(TimeoutError, match="missing logits rank"):
        collect_logits(control, response, wait_timeout=.02)
