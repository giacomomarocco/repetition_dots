import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from filler.dsv4.campaign_hook import make_campaign_hook, validate_ack_records
from filler.dsv4.factorial import NativeResidualHooks, replace_residual_rows
from filler.dsv4.patching import atomic_json, build_manifest, campaign_counts, file_digest, layer42_diagnostics
from filler.dsv4.patching_campaign import verify_records
from test_one_fact_patching import FakeTransport, Tokenizer, make_campaign, rendered, run_model
from test_filler_redundancy import raw_response, write_raw


def test_design_counts_mappings_and_canonical_answers():
    m = build_manifest(rendered(), Tokenizer(), "filler-repeat")
    assert m["counts"] == dict(panels=24, targets=96, baselines=96, trials=192, identity=192)
    assert len(layer42_diagnostics(m, m["pilot_panel"])) == 8
    assert sum(m["counts"][k] for k in ("baselines", "trials", "identity")) + 8 == 488
    assert m["bootstrap"] == dict(resamples=2000, seed=42, unit="panel")
    assert m == build_manifest(rendered(), Tokenizer(), "filler-repeat")
    for s in m["trials"] + m["identity_controls"]:
        i = s["source_filler_index"]
        assert s["donor_id"] == s["target_id"]
        assert s["positions"] == list(range(11 + i + 1, 31))
        assert s["source_positions"] == (s["positions"] if s["kind"] == "identity" else [11 + i] * (19 - i))
        assert s["layers"] == list(range(43)) and s["recomputation"] == "full_downstream"
        assert len(s["candidates"]) == 1 and s["candidates"][0]["label"] == "target_sum"
    class BadTokenizer(Tokenizer):
        def encode(self, text, **kwargs):
            return [1, 2] if text.isdigit() else super().encode(text, **kwargs)
    with pytest.raises(ValueError, match="canonical single token"):
        build_manifest(rendered(), BadTokenizer(), "filler-repeat")


@pytest.mark.parametrize("positions,sources", [([1, 2], [0]), ([1], [-1]), ([1], [6]),
    ([1, 1], [0, 0]), ([1], [True]), ([1], [1.5]), ([True], [0])])
def test_invalid_mappings(positions, sources):
    with pytest.raises(ValueError):
        replace_residual_rows(torch.zeros(6, 2, 3), positions=positions,
                              source_positions=sources, donor=torch.ones(6, 2, 3))


def test_exact_repeated_sources_unselected_rows_and_aliasing():
    original = torch.arange(36.).reshape(6, 2, 3)
    clean = original.clone()
    out = replace_residual_rows(original, positions=[2, 3, 4], source_positions=[1, 1, 1], donor=original)
    assert torch.equal(out[[2, 3, 4]], clean[[1, 1, 1]])
    assert torch.equal(out[[0, 1, 5]], clean[[0, 1, 5]])
    assert torch.equal(original, clean)
    out = replace_residual_rows(original, positions=[1, 2], source_positions=[2, 1], donor=original)
    assert torch.equal(out[[1, 2]], original[[2, 1]])
    assert torch.equal(replace_residual_rows(original, positions=[2], donor=original), original)
    with pytest.raises(ValueError, match="shape"):
        replace_residual_rows(original, positions=[2], source_positions=[1], donor={1: torch.ones(2)})


class Layer(torch.nn.Module):
    use_fused_mhc_post_pre = False
    def __init__(self, i):
        super().__init__()
        self.layer_id = i
    def forward(self, x):
        return x + .01 * x.cumsum(0), "aux"


def test_native_all_43_layers_and_final_layer_causality():
    model = SimpleNamespace(layers=torch.nn.ModuleList([Layer(i) for i in range(43)]))
    x = torch.arange(204.).reshape(34, 2, 3)
    clean = run_model(model, x)
    hooks = NativeResidualHooks(model)
    for source in range(6):
        positions = list(range(11 + source + 1, 31))
        sources = [11 + source] * len(positions)
        with hooks.transplant_layers(range(43), positions, clean, source_positions=sources):
            patched = run_model(model, x)
        for lid in range(43):
            assert torch.equal(patched[lid][positions], clean[lid][sources])
            assert torch.equal(patched[lid][sources], clean[lid][sources])
        assert torch.equal(patched[0][31:], clean[0][31:])
        assert not torch.equal(patched[1][31:], clean[1][31:])
        assert not torch.equal(patched[42][-1], clean[42][-1])
        with hooks.transplant_layers([42], positions, clean, source_positions=sources):
            last = run_model(model, x)
        assert torch.equal(last[42][31:], clean[42][31:])
    assert torch.equal(run_model(model, x)[42], clean[42])


def test_real_hook_saved_source_destination_and_ack_integrity(tmp_path):
    root = tmp_path / "control"
    common = dict(runtime_id="r", config_hash="h", cell_id="c", num_tokens=6,
                  recomputation="full_downstream", clean_capture=None, donor_capture=None)
    clean = dict(common, request_id="clean", layers=[], positions=[], capture_all=True, output_root=str(tmp_path / "clean"))
    atomic_json(root / "NEXT.json", clean)
    hook = make_campaign_hook(dict(control_root=str(root), num_layers=43))
    x = torch.arange(36.).reshape(6, 2, 3)
    for lid in range(43):
        hook(Layer(lid), (), (x + lid, "aux"))
    ack = json.loads((root / "acks/clean.rank0.json").read_text())
    donor = dict(cell_id="c", ranks={"0": ack["capture"]})
    control = dict(common, request_id="repeat", layers=list(range(43)), positions=[2, 3, 4],
                   source_positions=[1, 1, 1], capture_all=False,
                   donor_capture=donor, output_root=str(tmp_path / "repeat"))
    atomic_json(root / "NEXT.json", control)
    for lid in range(43):
        out = hook(Layer(lid), (), (x + lid, "aux"))
        assert torch.equal(out[0][[2, 3, 4]], (x + lid)[[1, 1, 1]])
        assert out[1] == "aux"
    ack = json.loads((root / "acks/repeat.rank0.json").read_text())
    validate_ack_records([ack], control, tp_size=1)
    saved = torch.load(ack["capture"]["path"], weights_only=True)
    assert saved["metadata"]["positions"] == [1, 2, 3, 4, 5]
    assert saved["metadata"]["source_positions"] == [1, 1, 1]
    assert len(saved["states"]) == 43
    bad = copy.deepcopy(ack)
    bad["audits"][20]["source_positions"] = [0, 0, 0]
    with pytest.raises(ValueError, match="source mapping"):
        validate_ack_records([bad], control, tp_size=1)
    bad = copy.deepcopy(ack)
    bad["audits"][42]["unselected_exact"] = False
    with pytest.raises(ValueError, match="unselected"):
        validate_ack_records([bad], control, tp_size=1)


class RepeatTransport(FakeTransport):
    def run(self, control, input_ids, scored_ids):
        _, acks = super().run(control, input_ids, scored_ids)
        for ack in acks:
            src = control.get("source_positions")
            if src is not None:
                ack["source_positions"] = src
                ack["positions"] = sorted(set([*control["positions"], *src, control["num_tokens"] - 1]))
                for audit in ack["audits"]:
                    audit["source_positions"] = src if audit["patched_positions"] else []
                    audit["unselected_exact"] = True
            path = Path(ack["capture"]["path"])
            # Real serialized tensors allow independent artifact integrity checks;
            # causal semantics are exercised separately with the real native hook.
            torch.save({"metadata": {k: v for k, v in ack.items() if k not in {"capture", "audits"}},
                        "states": {i: torch.zeros(len(ack["positions"]), 2, 3) for i in range(43)},
                        "audits": ack["audits"]}, path)
            ack["capture"]["sha256"] = file_digest(path)
        cell = self.cells[control["cell_id"]]
        correct = 1000 + cell["target"]
        logits = torch.full((1200,), -10.)
        logits[1199] = 0.
        logits[correct] = 2. if cell["col"] == 0 else -2.
        if control["layers"] and control["layers"] != [42] and control.get("source_positions") != control["positions"]:
            if len(control["positions"]) == 14:
                logits[correct] -= 3. if "z001" not in cell["cell_id"] else 1.
            else:
                logits[correct] += 3.
        return write_raw(control, raw_response(logits, scored_ids), logits), acks


def reduced():
    m = build_manifest(rendered(), Tokenizer(), "filler-repeat")
    m["panels"] = m["panels"][:2]
    pids = {p["panel_id"] for p in m["panels"]}
    for key in ("trials", "identity_controls"):
        m[key] = [s for s in m[key] if s["panel_id"] in pids]
    return m


def test_restart_report_accuracy_shared_bootstrap_and_tamper(tmp_path):
    from filler.dsv4.patching_analysis import summarize
    m = reduced()
    first = make_campaign(m, tmp_path, "old", RepeatTransport(m, fail_at=22))
    with pytest.raises(InterruptedError):
        first.run()
    old = next(r for r in first.journal.records.values() if r["kind"] == "trial")
    second = make_campaign(m, tmp_path, "new", RepeatTransport(m))
    result = second.run()
    assert result == dict(trials=16, identity=16, matched_baseline_cells=8, passed=True)
    assert second.journal.records[old["record_id"]] == old
    assert any(r["kind"] == "identity_recheck" for r in second.journal.records.values())
    report = summarize(m, second.journal, tmp_path / "analysis")
    assert report["resamples"] == 2000 and report["bootstrap_unit"] == "panel"
    counts = report["accuracy_counts"]
    assert counts[0] == dict(site="repeat_filler_5", n=8, clean_correct=4, patched_correct=2, correct_to_incorrect=2, incorrect_to_correct=0)
    assert counts[1] == dict(site="repeat_filler_0", n=8, clean_correct=4, patched_correct=8, correct_to_incorrect=0, incorrect_to_correct=4)
    rows = list(csv.DictReader((tmp_path / "analysis/per_example.csv").open()))
    assert len(rows) == 16
    indices = np.random.default_rng(42).integers(0, 2, (2000, 2))
    for e in report["conditions"]:
        v = np.array([np.mean([float(r[e["metric"]]) for r in rows if r["site"] == e["site"] and r["panel_id"] == p["panel_id"]]) for p in m["panels"]])
        b = v[indices].mean(1)
        assert e["mean"] == pytest.approx(v.mean())
        assert e["bootstrap_se"] == pytest.approx(b.std(ddof=1))
        assert [e["ci_low"], e["ci_high"]] == pytest.approx(np.quantile(b, [.025, .975]))
    for e in report["paired_comparisons"]:
        vals = []
        for p in m["panels"]:
            vals.append(np.mean([float(r[e["metric"]]) for r in rows if r["site"] == "repeat_filler_0" and r["panel_id"] == p["panel_id"]]) - np.mean([float(r[e["metric"]]) for r in rows if r["site"] == "repeat_filler_5" and r["panel_id"] == p["panel_id"]]))
        b = np.array(vals)[indices].mean(1)
        assert e["bootstrap_se"] == pytest.approx(b.std(ddof=1))
        assert [e["ci_low"], e["ci_high"]] == pytest.approx(np.quantile(b, [.025, .975]))
    for name in ("accuracy", "logprob", "logit"):
        for ext in ("png", "pdf"):
            assert (tmp_path / f"analysis/{name}.{ext}").is_file()
    r = second.journal.records[old["record_id"]]
    acks = json.loads(Path(r["ack_path"]).read_text())
    path = Path(acks[3]["capture"]["path"])
    original = path.read_bytes()
    saved = torch.load(path, weights_only=True)
    saved["states"][42][saved["metadata"]["positions"].index(r["positions"][0])] += 1
    torch.save(saved, path)
    checksum = file_digest(path)
    acks[3]["capture"]["sha256"] = checksum
    r["capture"]["ranks"]["3"]["sha256"] = checksum
    atomic_json(Path(r["ack_path"]), acks)
    r["ack_sha256"] = file_digest(Path(r["ack_path"]))
    with pytest.raises(ValueError, match="destination tensor"):
        verify_records(m, second.journal)
    path.write_bytes(original)
    checksum = file_digest(path)
    acks[3]["capture"]["sha256"] = checksum
    r["capture"]["ranks"]["3"]["sha256"] = checksum
    atomic_json(Path(r["ack_path"]), acks)
    r["ack_sha256"] = file_digest(Path(r["ack_path"]))
    control = json.loads(Path(r["control_path"]).read_text())
    control["source_positions"][0] += 1
    atomic_json(Path(r["control_path"]), control)
    r["control_sha256"] = file_digest(Path(r["control_path"]))
    with pytest.raises(ValueError, match="source"):
        verify_records(m, second.journal)


def test_cli_launch_and_storage_capture_union():
    from scripts.dsv4.one_fact_patching import REPEAT_ROOT, parse_args, launch_command
    args = parse_args(["prepare", "--design", "filler-repeat"])
    assert args.root == REPEAT_ROOT.resolve()
    command = launch_command(Path("/tmp/control"), raw_logits=True)
    hooks = json.loads(command[command.index("--forward-hooks") + 1])
    assert len(hooks) == 2 and len(hooks[0]["target_modules"]) == 43
    assert "--enable-return-hidden-states" in command
