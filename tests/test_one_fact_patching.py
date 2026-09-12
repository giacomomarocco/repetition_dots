import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from filler.dsv4.campaign_hook import make_campaign_hook, validate_ack_records, validate_acknowledgements
from filler.dsv4.factorial import NativeResidualHooks, replace_residual_rows
from filler.dsv4.patching import (
    Journal, PILOT_PANEL, atomic_json, build_manifest, candidates, digest, file_digest,
    invariant, requested_scores, score_candidates, layer42_diagnostics,
)
from filler.dsv4.patching_campaign import Campaign, WalltimeReached, read_response, verify_records


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return [1000 + int(text)] if text.isdigit() else [ord(c) for c in text]
    def decode(self, ids):
        return str(ids[0] - 1000)


def rendered():
    panels = []
    for i in range(24):
        pid = PILOT_PANEL if i == 0 else f"discovery-filler-z{i:03}"
        cells = []
        for row in range(2):
            for col in range(2):
                a, x = 10 + row * 7, 30 + col * 3
                prompt = "question-" + str(row) + str(col) + "." * 20 + "ans"
                cells.append({"cell_id": f"{pid}:{row}{col}", "panel_id": pid, "row": row,
                    "col": col, "left_value": a, "right_value": x, "target": a + x,
                    "rendered_prompt": prompt, "input_ids": Tokenizer().encode(prompt)})
        panels.append({"panel_id": pid, "cells": cells,
                       "positions": {"fillers": list(range(11, 31)), "answer_prompt": 33, "last_question": 10}})
    return {"filler_length": 20, "split": "discovery", "eligible_panels": panels}


def manifest(two_panels=False, design="original"):
    value = build_manifest(rendered(), Tokenizer(), design=design)
    if two_panels:
        value["panels"] = value["panels"][:2]
        pids = {p["panel_id"] for p in value["panels"]}
        for key in ("trials", "identity_controls"):
            value[key] = [s for s in value[key] if s["panel_id"] in pids]
    return value


def response(ids, output=1040, change=0.):
    return {"output_ids": [output], "meta_info": {"cached_tokens": 0,
        "output_token_ids_logprobs": [[[-2. - t / 1000. + change, t, None] for t in dict.fromkeys(ids)]]}}


def test_frozen_counts_alignment_candidates_and_phase_independent_ids():
    m = manifest()
    assert m["counts"] == {"panels": 24, "targets": 96, "trials": 1536, "identity": 768, "baselines": 192}
    pilot = [s for s in m["trials"] if s["panel_id"] == PILOT_PANEL]
    assert len(pilot) == 64
    assert len({s["record_id"] for s in m["trials"] + m["identity_controls"]}) == 2304
    assert all("pilot" not in s["record_id"] for s in pilot)
    assert {tuple(s["layers"]) for s in pilot} == {tuple(range(33, 43)), tuple(range(43))}
    assert {tuple(s["positions"]) for s in pilot} == {(16,), (21,)}
    assert m["panels"][0]["answer_position"] == len(m["panels"][0]["cells"][0]["input_ids"]) - 1
    assert all(len(s["candidates"]) == 3 for s in pilot)
    for s in pilot:
        if s["donor_role"] == "left":
            assert s["candidates"][1]["token_id"] == s["candidates"][2]["token_id"]
    bad = rendered()
    bad["eligible_panels"][0]["cells"][0]["input_ids"].pop()
    with pytest.raises(ValueError, match="alignment"):
        build_manifest(bad, Tokenizer())
    legacy = rendered()
    del legacy["split"]
    for panel in legacy["eligible_panels"]:
        for cell in panel["cells"]:
            cell["split"] = "discovery"
    assert build_manifest(legacy, Tokenizer())["counts"] == m["counts"]
    legacy["eligible_panels"][0]["cells"][0]["split"] = "confirmation"
    with pytest.raises(ValueError, match="discovery"):
        build_manifest(legacy, Tokenizer())


def test_aliases_and_full_vocabulary_scores_explicit_even_outside_topk():
    mapping = candidates({"left_value": 10, "right_value": 30}, {"left_value": 17, "right_value": 30}, Tokenizer())
    ids = [c["token_id"] for c in mapping]
    clean, patch = response(ids), response(ids, change=.7)
    rows = score_candidates(mapping, clean, patch)
    assert [r["label"] for r in rows] == ["target_sum", "donor_sum", "mixed_sum"]
    assert rows[1]["delta_logprob"] == rows[2]["delta_logprob"] == pytest.approx(.7)
    with pytest.raises(ValueError, match="missing"):
        requested_scores(response(ids[:1]), ids)
    with pytest.raises(ValueError, match="probability"):
        requested_scores(response(ids, change=float("nan")), ids)
    assert not invariant(clean, response(ids, output=0), ids)["passed"]
    assert not invariant(clean, response(ids, change=.151), ids)["passed"]


def test_native_final_layer_equivalence_checks_hidden_argmax_ranks_and_explicit_scores(tmp_path, monkeypatch):
    from filler.dsv4.lens import DeepseekV4LensWeights, project_logits
    from filler.dsv4.patching_campaign import native_equivalence
    torch.manual_seed(2)
    weights = DeepseekV4LensWeights(torch.randn(2, 6), torch.randn(2), torch.randn(1),
                                   torch.ones(3), torch.randn(10, 3))
    monkeypatch.setattr("filler.dsv4.lens.load_checkpoint_readout", lambda *a, **kw: weights)
    hidden = torch.randn(2, 3)
    lp = torch.log_softmax(project_logits(hidden, weights), -1)
    refs = {}
    for rank in range(4):
        path = tmp_path / f"rank{rank}.pt"
        torch.save({"states": {42: hidden.unsqueeze(0)}}, path)
        refs[str(rank)] = {"path": str(path), "sha256": file_digest(path)}
    top = int(lp.argmax())
    raw = {"output_ids": [top], "meta_info": {
        "hidden_states": [hidden.flatten().tolist()],
        "output_top_logprobs": [[[float(lp[top]), top, None]]],
        "output_token_ids_logprobs": [[[float(lp[t]), t, None] for t in (0, 9)]]}}
    capture = {"ranks": refs}
    assert native_equivalence(capture, raw, tmp_path, [0, 9])["passed"]
    bad = copy.deepcopy(raw)
    bad["meta_info"]["output_token_ids_logprobs"][0][1][0] -= .151
    assert not native_equivalence(capture, bad, tmp_path, [0, 9])["passed"]
    bad = copy.deepcopy(raw)
    bad["meta_info"]["hidden_states"][0][0] += 1
    assert not native_equivalence(capture, bad, tmp_path, [0, 9])["passed"]
    bad = copy.deepcopy(raw)
    bad["output_ids"] = [(top + 1) % 10]
    assert not native_equivalence(capture, bad, tmp_path, [0, 9])["passed"]
    path = Path(refs["3"]["path"])
    torch.save({"states": {42: (hidden + 1).unsqueeze(0)}}, path)
    refs["3"]["sha256"] = file_digest(path)
    assert not native_equivalence(capture, raw, tmp_path, [0, 9])["passed"]


class CausalLayer(torch.nn.Module):
    use_fused_mhc_post_pre = False
    def __init__(self, lid):
        super().__init__()
        self.layer_id = lid
    def forward(self, hidden):
        return (hidden + hidden.cumsum(0), "aux")


def run_model(model, hidden):
    results = {}
    for lid, layer in enumerate(model.layers):
        hidden, _ = layer(hidden)
        results[lid] = hidden.clone()
    return results


def test_native_simultaneous_replacement_causal_propagation_and_restoration():
    model = SimpleNamespace(layers=torch.nn.ModuleList([CausalLayer(i) for i in range(4)]))
    hooks = NativeResidualHooks(model)
    hidden = torch.ones(6, 2, 3)
    clean = run_model(model, hidden)
    donor = run_model(model, hidden * 2)
    with hooks.transplant_layers([0, 1], [1], donor):
        full = run_model(model, hidden)
    with hooks.transplant_layers([0, 1], [1], donor, clean=clean,
                                 recomputation="answer_only", answer_position=5):
        frozen = run_model(model, hidden)
    for lid in (0, 1):
        assert torch.equal(full[lid][1], donor[lid][1])
        assert torch.equal(frozen[lid][1], donor[lid][1])
    for lid in range(4):
        assert torch.equal(frozen[lid][[0, 2, 3, 4]], clean[lid][[0, 2, 3, 4]])
    assert not torch.equal(full[3][3], clean[3][3])
    assert not torch.equal(frozen[3][-1], clean[3][-1])
    assert not torch.equal(full[3][-1], frozen[3][-1])
    assert torch.equal(run_model(model, hidden)[3], clean[3])
    with hooks.transplant(1, [1], {1: donor[1][1]}):
        old = run_model(model, hidden)
    assert torch.equal(old[1][1], donor[1][1])
    with hooks.transplant_layers([3], [1], donor):
        last = run_model(model, hidden)
    assert torch.equal(last[3][-1], clean[3][-1])


@pytest.mark.parametrize("positions,answer", [([-1], 5), ([6], 5), ([1, 1], 5), ([1], 4)])
def test_invalid_absolute_alignment_rejected(positions, answer):
    with pytest.raises(ValueError):
        replace_residual_rows(torch.zeros(6, 2, 3), positions=positions,
                              clean=torch.ones(6, 2, 3), recomputation="answer_only", answer_position=answer)


def hook_pass(tmp_path, *, rid, clean=None, donor=None, mode="full_downstream", layers=(), positions=(), all_positions=True):
    root = tmp_path / "control"
    control = {"request_id": rid, "runtime_id": "r1", "config_hash": "hash", "cell_id": rid,
               "num_tokens": 6, "layers": list(layers), "positions": list(positions),
               "recomputation": mode, "capture_all": all_positions, "clean_capture": clean,
               "donor_capture": donor, "output_root": str(tmp_path / rid)}
    atomic_json(root / "NEXT.json", control)
    hook = make_campaign_hook({"control_root": str(root), "num_layers": 3})
    outputs = []
    for lid in range(3):
        hidden = torch.full((6, 2, 3), float(lid + (10 if rid == "donor" else 0)))
        outputs.append(hook(CausalLayer(lid), (), (hidden, "aux"))[0])
    acks = validate_acknowledgements(root, control, tp_size=1, num_layers=3)
    ref = {"cell_id": rid, "ranks": {"0": acks[0]["capture"]}}
    return ref, outputs, control, acks


def test_sglang_full_capture_simultaneous_patch_restoration_and_rank_acks(tmp_path):
    clean, _, _, _ = hook_pass(tmp_path, rid="clean")
    donor, _, _, _ = hook_pass(tmp_path, rid="donor")
    ref, outputs, control, acks = hook_pass(tmp_path, rid="patch", clean=clean, donor=donor,
        mode="answer_only", layers=(0, 2), positions=(1,), all_positions=False)
    assert outputs[0][1].eq(10).all() and outputs[2][1].eq(12).all()
    saved = torch.load(ref["ranks"]["0"]["path"], weights_only=True)
    assert saved["metadata"]["positions"] == [1, 5]
    assert all(x["restored_count"] == 4 and x["restoration_exact"] for x in saved["audits"])
    with pytest.raises(ValueError, match="missing rank"):
        validate_ack_records(acks, control, tp_size=4, num_layers=3)
    wrong = copy.deepcopy(acks)
    wrong[0]["rank"] = 1
    with pytest.raises(ValueError, match="rank"):
        validate_ack_records(wrong, control, tp_size=1, num_layers=3)
    wrong = copy.deepcopy(acks)
    wrong[0]["audits"][0]["patched_positions"] = []
    with pytest.raises(ValueError, match="replacement"):
        validate_ack_records(wrong, control, tp_size=1, num_layers=3)
    wrong = copy.deepcopy(acks)
    wrong[0]["runtime_id"] = "stale"
    with pytest.raises(ValueError, match="runtime_id"):
        validate_ack_records(wrong, control, tp_size=1, num_layers=3)


def test_hook_rejects_chunked_prompt_and_consumes_once(tmp_path):
    _, _, control, _ = hook_pass(tmp_path, rid="capture")
    hook = make_campaign_hook({"control_root": str(tmp_path / "control"), "num_layers": 3})
    with pytest.raises(RuntimeError, match="full-prompt"):
        hook(CausalLayer(0), (), torch.zeros(1, 2, 3))
    hook = make_campaign_hook({"control_root": str(tmp_path / "control"), "num_layers": 3})
    for lid in range(3):
        hook(CausalLayer(lid), (), torch.zeros(6, 2, 3))
    # No new trigger: decode shape is ignored, and no capture is overwritten.
    checksum = file_digest(Path(control["output_root"]) / "rank0.pt")
    for lid in range(3):
        hook(CausalLayer(lid), (), torch.ones(1, 2, 3))
    assert checksum == file_digest(Path(control["output_root"]) / "rank0.pt")


def test_journal_recovers_result_before_progress_and_incomplete_tail(tmp_path, monkeypatch):
    journal = Journal(tmp_path, "hash")
    def crash(**status):
        raise InterruptedError("between result fsync and progress replace")
    monkeypatch.setattr(journal, "progress", crash)
    with pytest.raises(InterruptedError):
        journal.append({"record_id": "a", "kind": "trial"})
    (tmp_path / "progress.json").write_text("broken")
    with (tmp_path / "results.jsonl").open("ab") as sink:
        sink.write(b'{"record": {"record_id": "unfinished"')
    recovered = Journal(tmp_path, "hash")
    assert set(recovered.records) == {"a"}
    assert list(tmp_path.glob("interrupted-tail-*.bin"))
    recovered.append({"record_id": "b", "kind": "trial"})
    assert set(Journal(tmp_path, "hash").records) == {"a", "b"}
    with pytest.raises(ValueError, match="config"):
        Journal(tmp_path, "other")


def test_journal_retains_complete_final_record_without_newline(tmp_path):
    j = Journal(tmp_path, "h")
    j.append({"record_id": "x", "kind": "trial"})
    path = tmp_path / "results.jsonl"
    path.write_bytes(path.read_bytes().rstrip(b"\n"))
    assert "x" in Journal(tmp_path, "h").records
    assert path.read_bytes().endswith(b"\n")
    path.write_bytes(path.read_bytes() + b'{broken}\n')
    with pytest.raises(ValueError, match="corrupt complete"):
        Journal(tmp_path, "h")


class FakeTransport:
    """Cheap deterministic HTTP/rank stand-in; actual tensor semantics tested above."""
    def __init__(self, m, *, fail_at=None, bad_identity=False):
        self.cells = {c["cell_id"]: c for p in m["panels"] for c in p["cells"]}
        self.controls = []
        self.fail_at, self.bad_identity = fail_at, bad_identity
    def run(self, control, input_ids, scored_ids):
        self.controls.append(control)
        if len(self.controls) == self.fail_at:
            raise InterruptedError("during trial before durable result")
        patch = bool(control["layers"])
        identity = patch and control["donor_capture"]["cell_id"] == control["cell_id"]
        effect = .5 if patch and not identity and control["layers"] != [42] else 0.
        if identity and self.bad_identity:
            effect = 1.
        raw = response(scored_ids, 1000 + self.cells[control["cell_id"]]["target"], effect)
        acks = []
        for rank in range(4):
            path = Path(control["output_root"]) / f"rank{rank}.pt"
            atomic_json(path, {"fake": True, "rank": rank, "request_id": control["request_id"]})
            n, pos = control["num_tokens"], control["positions"]
            acks.append({**{k: control[k] for k in ("request_id", "runtime_id", "config_hash", "num_tokens", "cell_id", "layers", "recomputation")},
                "rank": rank, "positions": list(range(n)) if control["capture_all"] else sorted(set([*pos, n - 1])),
                "patch_positions": pos, "fused_mhc_post_pre": False,
                "audits": [{"layer": lid, "patched_positions": pos if lid in control["layers"] else [],
                    "restored_count": n - len(pos) - 1 if control["recomputation"] == "answer_only" else 0,
                    "replacement_exact": True, "restoration_exact": True} for lid in range(43)],
                "capture": {"path": str(path), "sha256": file_digest(path)}})
        return raw, acks


def make_campaign(m, root, runtime, transport, **kwargs):
    return Campaign(m, root, runtime, transport, Path("unused"), deadline=time.time() + 3600,
                    validation=lambda *a: {"passed": True}, **kwargs)


@pytest.mark.parametrize("design,scale", [("original", 2), ("filler-coverage", 1), ("filler10-coverage", .5)])
def test_pilot_automatically_continues_full_and_all_three_deltas_are_verified(tmp_path, design, scale):
    m = manifest(two_panels=True, design=design)
    transport = FakeTransport(m)
    campaign = make_campaign(m, tmp_path, "runtime1", transport)
    result = campaign.run()
    assert result == {"trials": 64 * scale, "identity": 32 * scale, "matched_baseline_cells": 16, "passed": True}
    assert len(transport.controls) == campaign.planned_passes == 96 * scale + 16 + len(layer42_diagnostics(m, PILOT_PANEL))
    records = list(campaign.journal.records.values())
    pilot_index = next(i for i, r in enumerate(records) if r["kind"] == "pilot_passed")
    assert sum(r["kind"] == "trial" for r in records[:pilot_index]) == 32 * scale
    assert sum(r["kind"] == "identity" for r in records[:pilot_index]) == 16 * scale
    assert sum(r["kind"] == "layer42_validation" for r in records) == len(layer42_diagnostics(m, PILOT_PANEL))
    assert {c["runtime_id"] for c in transport.controls} == {"runtime1"}
    assert sum(r["kind"] == "native_validation" for r in records) == 1
    trial = next(r for r in records if r["kind"] == "trial")
    assert all(s["delta_logprob"] == pytest.approx(.5) for s in trial["scores"])
    from filler.dsv4.patching_analysis import summarize
    summary = summarize(m, campaign.journal, tmp_path / "analysis", resamples=100, seed=2)
    assert len(summary["conditions"]) == 24 * scale
    assert len(summary["paired_comparisons"]) == (96 if design == "original" else 24 * scale)
    if design == "filler-coverage":
        assert {r["factor"] for r in summary["paired_comparisons"]} == {"recomputation", "donor_role"}
        assert "two intervening non-filler tokens" in (tmp_path / "analysis/REPORT.md").read_text()
    import csv
    csv_scores = list(csv.DictReader((tmp_path / "analysis/trial_scores.csv").open()))
    assert len(csv_scores) == 64 * scale * 3
    assert all(float(r["patched_logprob"]) - float(r["clean_logprob"]) == pytest.approx(float(r["delta_logprob"])) for r in csv_scores)
    assert all(r["mean_paired_difference"] == pytest.approx(0) for r in summary["paired_comparisons"])
    assert (tmp_path / "analysis/candidate_changes.png").is_file()
    assert (tmp_path / "analysis/candidate_changes.pdf").is_file()
    trial["scores"][2]["delta_logprob"] += 1
    with pytest.raises(ValueError, match="deltas"):
        verify_records(m, campaign.journal)


@pytest.mark.parametrize("design,scale", [("original", 2), ("filler-coverage", 1), ("filler10-coverage", .5)])
@pytest.mark.parametrize("stage", ["identity", "trial", "second_panel"])
def test_resume_reuses_completed_trials_and_gets_new_runtime_baselines(tmp_path, design, scale, stage):
    m = manifest(two_panels=True, design=design)
    # Interrupt during controls, after one pilot trial, or after one second-panel trial.
    first_trial = 8 + 16 * scale + len(layer42_diagnostics(m, PILOT_PANEL))
    fail_at = {"identity": 13, "trial": first_trial + 2,
               "second_panel": first_trial + 32 * scale + 8 + 16 * scale + 2}[stage]
    first = make_campaign(m, tmp_path, "old", FakeTransport(m, fail_at=fail_at))
    with pytest.raises(InterruptedError):
        first.run()
    done = [r for r in first.journal.records.values() if r["kind"] == "trial"]
    assert len(done) == {"identity": 0, "trial": 1, "second_panel": 32 * scale + 1}[stage]
    old_identities = [r for r in first.journal.records.values() if r["kind"] == "identity"]
    assert (tmp_path / "runtimes/old/failure.json").is_file()
    new_transport = FakeTransport(m)
    second = make_campaign(m, tmp_path, "new", new_transport)
    second.run()
    old_record = (done or old_identities)[0]
    preserved = second.journal.records[old_record["record_id"]]
    assert preserved == old_record
    assert second.journal.records[preserved["baseline_id"]]["runtime_id"] == "old"
    rechecks = sum(r["panel_id"] != PILOT_PANEL if stage == "second_panel" else True for r in old_identities)
    assert len([r for r in second.journal.records.values() if r["kind"] == "identity_recheck"]) == rechecks
    assert len(new_transport.controls) == second.planned_passes
    assert len([r for r in second.journal.records.values() if r["kind"] == "trial"]) == 64 * scale
    assert len([r for r in second.journal.records.values() if r["kind"] == "native_validation"]) == 2
    diagnostics = [r for r in second.journal.records.values() if r["kind"] == "layer42_validation" and r["runtime_id"] == "new"]
    assert len(diagnostics) == len(layer42_diagnostics(m, PILOT_PANEL))
    assert {r["panel_id"] for r in diagnostics} == {m["panels"][1 if stage == "second_panel" else 0]["panel_id"]}


def test_failed_identity_stops_before_substantive_and_preserves_diagnostics(tmp_path):
    m = manifest(two_panels=True)
    transport = FakeTransport(m, bad_identity=True)
    campaign = make_campaign(m, tmp_path, "fail", transport)
    with pytest.raises(RuntimeError, match="invariance"):
        campaign.run()
    assert not any(r["kind"] in {"trial", "identity"} for r in campaign.journal.records.values())
    assert any(r["kind"] == "failed_validation" for r in campaign.journal.records.values())
    assert len(transport.controls) == 9


def test_walltime_checkpoint_avoids_new_inference(tmp_path):
    m = manifest(two_panels=True)
    transport = FakeTransport(m)
    campaign = make_campaign(m, tmp_path, "short", transport)
    campaign.deadline = time.time() + 1
    with pytest.raises(WalltimeReached):
        campaign.run()
    assert not transport.controls
    assert json.loads((tmp_path / "progress.json").read_text())["status"] == "checkpointed"


def test_launcher_instruments_before_load_and_has_no_phase_shutdown():
    from scripts.dsv4.one_fact_patching import launch_command
    command = launch_command(Path("/tmp/control"))
    assert command[command.index("--chunked-prefill-size") + 1] == "-1"
    for flag in ("--disable-radix-cache", "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
                 "--enable-return-hidden-states", "--disable-overlap-schedule"):
        assert flag in command
    spec = json.loads(command[command.index("--forward-hooks") + 1])[0]
    assert len(spec["target_modules"]) == 43
    assert "campaign_hook" in spec["hook_factory"]


@pytest.mark.parametrize("qos", ["interactive", "gpu_interactive"])
def test_allocation_accepts_nersc_gpu_qos_alias_and_reads_utc(qos):
    from datetime import datetime, timezone
    from scripts.dsv4.one_fact_patching import parse_allocation_deadline
    raw = (f"JobId=58128140 Account=m5258_g QOS={qos} JobState=RUNNING "
           "NumNodes=1 NumCPUs=128 TimeLimit=04:00:00 EndTime=2026-09-10T00:23:06 "
           "Features=gpu&a100&hbm80g AdminComment={\"qos\":\"gpu_interactive\"}")
    assert parse_allocation_deadline(raw) == datetime(2026, 9, 10, 0, 23, 6, tzinfo=timezone.utc).timestamp()
    with pytest.raises(RuntimeError, match="interactive GPU"):
        parse_allocation_deadline(raw.replace(f"QOS={qos}", "QOS=gpu_regular"))
    with pytest.raises(RuntimeError, match="four-hour"):
        parse_allocation_deadline(raw.replace("TimeLimit=04:00:00", "TimeLimit=01:00:00"))


def test_allocation_query_forces_scheduler_to_return_utc(monkeypatch):
    from scripts.dsv4.one_fact_patching import allocation_deadline
    monkeypatch.setenv("SLURM_JOB_ID", "58128140")
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    def query(command, **kwargs):
        assert command == ["scontrol", "show", "job", "58128140", "-o"]
        assert kwargs["env"]["TZ"] == "UTC"
        return "QOS=gpu_interactive NumNodes=1 TimeLimit=04:00:00 EndTime=2026-09-10T00:23:06"
    monkeypatch.setattr("scripts.dsv4.one_fact_patching.subprocess.check_output", query)
    assert allocation_deadline() > 0


def test_http_port_does_not_leak_into_sglang_internal_broadcast_allocation(monkeypatch):
    import ast
    import os
    from scripts.dsv4.one_fact_patching import WORKSPACE, launch_command, server_environment
    monkeypatch.setenv("SGLANG_PORT", "30002")
    command = launch_command(Path("/tmp/control"), port=30123)
    assert command[command.index("--port") + 1] == "30123"
    child_env = server_environment()
    # Exercise the actual pinned SGLang port allocator without importing its
    # full inference stack or binding a network socket in the CPU test suite.
    tree = ast.parse((WORKSPACE / "ports/sglang/python/sglang/srt/utils/network.py").read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "get_open_port")
    scope = {"os": os, "is_port_available": lambda p: True,
             "try_bind_socket": lambda: SimpleNamespace(getsockname=lambda: ("0.0.0.0", 47111), close=lambda: None)}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "pinned-sglang-get-open-port", "exec"), scope)
    assert scope["get_open_port"]() == 30002  # Reproduce the startup collision.
    monkeypatch.setattr(os, "environ", child_env)
    assert scope["get_open_port"]() == 47111  # Child uses an independent socket.


def test_filler_coverage_design_counts_diagnostics_and_saved_inputs():
    original, coverage = manifest(), manifest(design="filler-coverage")
    assert coverage["panels"] == original["panels"]
    assert coverage["counts"] == {"panels": 24, "targets": 96, "trials": 768, "identity": 384, "baselines": 192}
    assert coverage["pilot_counts"] == {"targets": 4, "trials": 32, "identity": 16, "baselines": 8}
    expected = {("all_fillers", tuple(range(43)), mode) for mode in ("full_downstream", "answer_only")}
    expected |= {("filler_5", tuple(range(32, 38)), mode) for mode in ("full_downstream", "answer_only")}
    assert {(s["site"], tuple(s["layers"]), s["recomputation"]) for s in coverage["trials"]} == expected
    specs = coverage["trials"] + coverage["identity_controls"]
    assert len({s["record_id"] for s in specs}) == 1152
    for s in specs:
        assert s["positions"] == (list(range(11, 31)) if s["site"] == "all_fillers" else [16])
    assert sum(coverage["counts"][k] for k in ("trials", "identity", "baselines")) + coverage["diagnostics_per_runtime"] == 1376
    for m in (original, coverage):
        diagnostics = layer42_diagnostics(m, PILOT_PANEL)
        assert len(diagnostics) == 32
        assert {tuple(s["layers"]) for s in diagnostics} == {(42,)}
        assert len({(s["target_id"], s["donor_id"], tuple(s["positions"]), s["recomputation"]) for s in diagnostics}) == 32
    assert "design" not in original and original["schema_version"] == 2
    assert build_manifest(rendered(), Tokenizer(), design="original") == original
    bad = rendered()
    for c in bad["eligible_panels"][0]["cells"]:
        c["rendered_prompt"] += "x"
        c["input_ids"] = Tokenizer().encode(c["rendered_prompt"])
    with pytest.raises(ValueError, match="two non-filler"):
        build_manifest(bad, Tokenizer(), design="filler-coverage")
    with pytest.raises(ValueError, match="unknown design"):
        build_manifest(rendered(), Tokenizer(), design="misspelled")


def test_design_cli_uses_separate_root_and_manifest_for_resume():
    from scripts.dsv4.one_fact_patching import COVERAGE_ROOT, DEFAULT_ROOT, parse_args
    assert parse_args(["prepare"]).root == DEFAULT_ROOT
    args = parse_args(["prepare", "--design", "filler-coverage"])
    assert args.root == COVERAGE_ROOT and args.design == "filler-coverage"
    assert parse_args(["run", "--root", str(COVERAGE_ROOT)]).root == COVERAGE_ROOT
    with pytest.raises(SystemExit):
        parse_args(["run", "--design", "filler-coverage"])


def test_reused_inputs_require_completed_campaign_and_exact_prompts_checkpoint_hashes():
    from scripts.dsv4.one_fact_patching import CHECKPOINT, RENDERED, WORKSPACE, verify_reused_inputs
    old = manifest()
    old["checkpoint_files"] = {"shard": {"size": 123, "mtime_ns": 42}}
    old["source_hashes"] = {str(p.relative_to(WORKSPACE)): "sha" for p in (
        RENDERED, CHECKPOINT / "config.json", CHECKPOINT / "tokenizer.json", CHECKPOINT / "tokenizer_config.json")}
    old["config_hash"] = digest({k: v for k, v in old.items() if k != "config_hash"})
    complete = {"config_hash": old["config_hash"], "status": "complete", "integrity": {"passed": True}}
    new = {**manifest(design="filler-coverage"), "checkpoint_files": old["checkpoint_files"], "source_hashes": old["source_hashes"]}
    verify_reused_inputs(new, old, complete)
    for key in ("panels", "checkpoint_files", "source_hashes"):
        bad = copy.deepcopy(new)
        if key == "panels":
            bad[key][0]["cells"][0]["rendered_prompt"] += "x"
        elif key == "checkpoint_files":
            bad[key]["shard"]["mtime_ns"] += 1
        else:
            bad[key][str(RENDERED.relative_to(WORKSPACE))] = "changed"
        with pytest.raises(ValueError, match="differ|changed"):
            verify_reused_inputs(bad, old, complete)
    with pytest.raises(ValueError, match="completed original"):
        verify_reused_inputs(new, old, {**complete, "status": "incomplete"})


class BoundedCausalLayer(CausalLayer):
    def forward(self, hidden):
        return (hidden + .05 * hidden.cumsum(0), "aux")


@pytest.mark.parametrize("site,layers", [("all_fillers", list(range(43))), ("filler_5", list(range(32, 38))), ("filler_10", list(range(32, 38)))])
def test_43_layer_native_coverage_restoration_and_downstream_propagation(site, layers):
    model = SimpleNamespace(layers=torch.nn.ModuleList([BoundedCausalLayer(i) for i in range(43)]))
    hooks = NativeResidualHooks(model)
    hidden = torch.arange(27 * 2 * 3, dtype=torch.float64).reshape(27, 2, 3) / 100 + 1
    clean, donor = run_model(model, hidden), run_model(model, hidden * 2)
    positions = list(range(4, 24)) if site == "all_fillers" else [9 if site == "filler_5" else 14]
    frozen_rows = [p for p in range(26) if p not in positions]
    with hooks.transplant_layers(layers, positions, donor):
        full = run_model(model, hidden)
    with hooks.transplant_layers(layers, positions, donor, clean=clean, recomputation="answer_only", answer_position=26):
        answer = run_model(model, hidden)
    for lid in range(43):
        assert torch.equal(answer[lid][frozen_rows], clean[lid][frozen_rows])
        if lid in layers:
            assert torch.equal(answer[lid][positions], donor[lid][positions])
            assert torch.equal(full[lid][positions], donor[lid][positions])
        if lid < layers[0]:
            assert torch.equal(answer[lid], clean[lid])
            assert torch.equal(full[lid], clean[lid])
    assert not torch.equal(full[42][-1], answer[42][-1])
    assert not torch.equal(answer[42][-1], clean[42][-1])
    if site == "all_fillers":
        # Causality keeps the prefix equal; only the two intervening rows and
        # answer row can differ between modes after each block's replacement.
        assert torch.equal(full[42][:4], answer[42][:4])
        assert torch.equal(full[42][positions], answer[42][positions])
        assert not torch.equal(full[42][24:26], answer[42][24:26])
    else:
        for lid in range(38, 43):
            assert not torch.equal(answer[lid][positions], donor[lid][positions])
            assert not torch.equal(answer[lid][positions], clean[lid][positions])
            assert not torch.equal(answer[lid][positions], answer[lid - 1][positions])
    for mode in ("answer_only", "full_downstream"):
        with hooks.transplant_layers([42], positions, donor, clean=clean, recomputation=mode, answer_position=26):
            diagnostic = run_model(model, hidden)
        assert torch.equal(diagnostic[42][-1], clean[42][-1])
        with hooks.transplant_layers(layers, positions, clean, clean=clean, recomputation=mode, answer_position=26):
            identity = run_model(model, hidden)
        assert torch.equal(identity[42], clean[42])


@pytest.mark.parametrize("positions,layers", [(list(range(4, 24)), list(range(43))), ([9], list(range(32, 38))), ([14], list(range(32, 38)))])
def test_campaign_hook_four_ranks_complete_replacement_and_exact_layer_acknowledgements(tmp_path, monkeypatch, positions, layers):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    root = tmp_path / "control"
    def execute(rid, *, clean=None, donor=None):
        control = {"request_id": rid, "runtime_id": "r1", "config_hash": "hash", "cell_id": rid,
                   "num_tokens": 27, "layers": layers if donor else [], "positions": positions if donor else [],
                   "recomputation": "answer_only" if donor else "full_downstream", "capture_all": donor is None,
                   "clean_capture": clean, "donor_capture": donor, "output_root": str(tmp_path / rid)}
        atomic_json(root / "NEXT.json", control)
        for rank in range(4):
            monkeypatch.setattr(torch.distributed, "get_rank", lambda rank=rank: rank)
            hook = make_campaign_hook({"control_root": str(root), "num_layers": 43})
            hidden = torch.ones(27, 2, 3, dtype=torch.float64) * (1 + rank) * (2 if rid == "donor" else 1)
            for lid in range(43):
                layer = BoundedCausalLayer(lid)
                hidden = hook(layer, (), layer(hidden))[0]
        acks = validate_acknowledgements(root, control)
        ref = {"cell_id": rid, "runtime_id": "r1", "ranks": {str(a["rank"]): a["capture"] for a in acks}}
        return ref, control, acks
    clean, _, _ = execute("clean")
    donor, _, _ = execute("donor")
    patched, control, acks = execute("patch", clean=clean, donor=donor)
    for rank in range(4):
        saved = torch.load(patched["ranks"][str(rank)]["path"], weights_only=True)
        donor_saved = torch.load(donor["ranks"][str(rank)]["path"], weights_only=True)
        for lid in layers:
            assert torch.equal(saved["states"][lid][:-1], donor_saved["states"][lid][positions])
        assert all(a["restored_count"] == 26 - len(positions) for a in saved["audits"])
    wrong = copy.deepcopy(acks)
    wrong[3]["audits"][layers[-1]]["patched_positions"] = positions[:-1]
    with pytest.raises(ValueError, match="replacement"):
        validate_ack_records(wrong, control)
    wrong = copy.deepcopy(acks)
    wrong[3]["audits"][38 if layers[-1] == 37 else 0]["patched_positions"] = positions if layers[-1] == 37 else []
    with pytest.raises(ValueError, match="replacement"):
        validate_ack_records(wrong, control)


def test_diagnostic_integrity_requires_exact_distinct_combinations(tmp_path):
    m = manifest(two_panels=True, design="filler-coverage")
    campaign = make_campaign(m, tmp_path, "r1", FakeTransport(m))
    campaign.run()
    diagnostic = next(r for r in campaign.journal.records.values() if r["kind"] == "layer42_validation")
    diagnostic["positions"] = diagnostic["positions"][:-1]
    with pytest.raises(ValueError, match="diagnostic differs"):
        verify_records(m, campaign.journal)


def test_filler_coverage_bootstrap_pairs_panels_and_conditions(tmp_path):
    import numpy as np
    from filler.dsv4.patching_analysis import summarize
    m = manifest(two_panels=True, design="filler-coverage")
    class VariedTransport(FakeTransport):
        def run(self, control, input_ids, scored_ids):
            raw, acks = super().run(control, input_ids, scored_ids)
            if control["layers"] and control["layers"] != [42] and control["donor_capture"]["cell_id"] != control["cell_id"]:
                panel = 1 if self.cells[control["cell_id"]]["panel_id"] == PILOT_PANEL else 2
                mode = 1 if control["recomputation"] == "full_downstream" else 3
                site = 1 if len(control["positions"]) == 20 else 2
                donor = 1 if self.cells[control["donor_capture"]["cell_id"]]["col"] == self.cells[control["cell_id"]]["col"] else 2
                for row in raw["meta_info"]["output_token_ids_logprobs"][0]:
                    row[0] += .01 * panel * mode * site * donor - .5
            return raw, acks
    campaign = make_campaign(m, tmp_path, "r1", VariedTransport(m))
    campaign.run()
    summary = summarize(m, campaign.journal, tmp_path / "analysis", resamples=20000, seed=42)
    draws = np.random.default_rng(42).integers(0, 2, (20000, 2))
    for r in summary["paired_comparisons"]:
        common = .01 * (1 if r["site"] == "all_fillers" else 2)
        if r["factor"] == "recomputation":
            panel_differences = np.array([1., 2.]) * common * -2 * (1 if r["donor_role"] == "left" else 2)
        else:
            panel_differences = np.array([1., 2.]) * common * -1 * (1 if r["recomputation"] == "full_downstream" else 3)
        expected = np.quantile(panel_differences[draws].mean(axis=1), [.025, .975])
        assert r["mean_paired_difference"] == pytest.approx(panel_differences.mean())
        assert [r["ci_low"], r["ci_high"]] == pytest.approx(expected)
