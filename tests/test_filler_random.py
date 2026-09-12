import copy
import json
import random
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from filler.dsv4.campaign_hook import make_campaign_hook, validate_ack_records
from filler.dsv4.filler_random import norm_error, random_row, stream_seed, validate_bank
from filler.dsv4.patching import atomic_json, build_manifest, file_digest, layer42_diagnostics
from filler.dsv4.patching_campaign import verify_records
from test_one_fact_patching import Tokenizer, rendered, make_campaign
from test_filler_repeat import Layer
from test_filler_redundancy import raw_response, write_raw


class RandomTokenizer(Tokenizer):
    def decode(self, ids):
        return {ord('a'): 'Answer', ord('n'): ':', ord('s'): ' '}.get(ids[0], super().decode(ids))


def manifest(reduced=False):
    m = build_manifest(rendered(), RandomTokenizer(), "filler-random")
    if reduced:
        m["panels"] = m["panels"][:1]
        pid = m["panels"][0]["panel_id"]
        for key in ("trials", "identity_controls"):
            m[key] = [s for s in m[key] if s["panel_id"] == pid]
    return m


def test_design_coverage_counts_and_defaults():
    from scripts.dsv4.one_fact_patching import parse_args, RANDOM_ROOT, storage_estimate
    m = manifest()
    assert m["counts"] == dict(panels=24, targets=96, trials=576, identity=576, baselines=96)
    assert len(layer42_diagnostics(m, m["pilot_panel"])) == 24
    assert 96 + 576 + 576 + 24 == 1272
    assert m == manifest()
    for s in m["trials"] + m["identity_controls"]:
        assert s["positions"] == list(range(12 + s["cutoff"], 31))
        assert s["source_positions"] == s["positions"]
        assert s["target_id"] == s["donor_id"]
        assert s["layers"] == list(range(43))
    for coverage in m["position_coverage"].values():
        assert len(coverage) == 24
        assert [p["token"] for p in coverage[-3:]] == ["Answer", ":", " "]
    seeds = [s for layers in m["random_seeds"].values() for positions in layers.values() for s in positions.values()]
    assert len(set(seeds)) == 96 * 43 * 19
    args = parse_args(["prepare", "--design", "filler-random"])
    assert args.root == RANDOM_ROOT and args.draws == 1 and args.seed == 42
    assert parse_args(["prepare"]).draws == 5
    with pytest.raises(SystemExit):
        parse_args(["prepare", "--design", "filler-random", "--draws", "5"])
    with pytest.raises(ValueError, match="Answer"):
        build_manifest(rendered(), Tokenizer(), "filler-random")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_stream_independence_reproducibility_norms_and_global_rng(dtype):
    clean = torch.arange(1, 8193).reshape(4, 2048).to(dtype)
    before_torch = torch.random.get_rng_state().clone()
    before_random = random.getstate()
    before_numpy = np.random.get_state()
    rows = []
    for cid, lid, pos in [("a", 0, 1), ("b", 0, 1), ("a", 1, 1), ("a", 0, 2)]:
        seed = stream_seed(42, cid, lid, pos)
        r = random_row(clean, seed)
        assert torch.equal(r, random_row(clean, seed))
        assert norm_error(r, clean) <= .005
        rows.append(r.float().flatten())
    for i in range(len(rows)):
        for j in range(i):
            assert abs(float(torch.nn.functional.cosine_similarity(rows[i], rows[j], dim=0))) < .05
    assert torch.equal(before_torch, torch.random.get_rng_state())
    assert before_random == random.getstate()
    after_numpy = np.random.get_state()
    assert before_numpy[0] == after_numpy[0] and np.array_equal(before_numpy[1], after_numpy[1]) and before_numpy[2:] == after_numpy[2:]
    assert random_row(torch.zeros(4, 3, dtype=dtype), 42).eq(0).all()
    for value in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="nonfinite"):
            random_row(torch.full((4, 3), value), 42)


class RandomTransport:
    """Actual 43-layer hooks, causal toy blocks and four simulated rank processes."""
    def __init__(self, m, fail_at=None):
        self.cells = {c["cell_id"]: c for p in m["panels"] for c in p["cells"]}
        self.controls = []
        self.fail_at = fail_at
        self.hooks = {}

    def run(self, control, input_ids, scored_ids):
        self.controls.append(control)
        if len(self.controls) == self.fail_at:
            raise InterruptedError("simulated restart")
        root = Path(control["output_root"]).parents[1] / "control"
        atomic_json(root / "NEXT.json", control)
        acks, finals = [], []
        for rank in range(4):
            hook = self.hooks.setdefault(rank, make_campaign_hook(dict(control_root=str(root))))
            x = torch.arange(len(input_ids) * 6, dtype=torch.float32).reshape(len(input_ids), 2, 3) / 100
            with patch("torch.distributed.is_initialized", return_value=True), patch("torch.distributed.get_rank", return_value=rank):
                for lid in range(43):
                    block = Layer(lid)
                    x = hook(block, (), block(x))[0]
            finals.append(x[-1].clone())
            acks.append(json.loads((root / f"acks/{control['request_id']}.rank{rank}.json").read_text()))
        assert all(torch.equal(finals[0], x) for x in finals)
        validate_ack_records(acks, control)
        logits = torch.full((1200,), -10.)
        logits[1199] = 0.
        cell = self.cells[control["cell_id"]]
        logits[1000 + cell["target"]] = finals[0].mean() / 100 - (10 if cell["col"] else 0)
        return write_raw(control, raw_response(logits, scored_ids), logits), acks


def test_real_hooks_all_layers_ranks_propagation_restart_and_artifact_tamper(tmp_path):
    m = manifest(reduced=True)
    first = make_campaign(m, tmp_path, "old", RandomTransport(m, fail_at=54))
    with pytest.raises(InterruptedError):
        first.run()
    old = next(r for r in first.journal.records.values() if r["kind"] == "trial")
    second = make_campaign(m, tmp_path, "new", RandomTransport(m))
    assert second.run()["passed"]
    assert second.journal.records[old["record_id"]] == old
    assert any(r["kind"] == "identity_recheck" for r in second.journal.records.values())
    r = next(r for r in second.journal.records.values() if r["kind"] == "trial" and r["runtime_id"] == "new")
    baseline = second.journal.records[r["baseline_id"]]
    bank = validate_bank(r["replacement_bank"], m, baseline)
    saved = torch.load(r["capture"]["ranks"]["0"]["path"], weights_only=True)
    clean = torch.load(r["clean_capture"]["ranks"]["0"]["path"], weights_only=True)
    positions = saved["metadata"]["positions"]
    assert positions == list(range(10, 34))
    for lid in range(43):
        for p in r["positions"]:
            assert torch.equal(saved["states"][lid][positions.index(p)], bank["states"][lid][bank["metadata"]["positions"].index(p)])
    assert torch.equal(saved["states"][0][-3:], clean["states"][0][-3:])
    assert not torch.equal(saved["states"][1][-3:], clean["states"][1][-3:])
    assert not torch.equal(saved["states"][42][-1], clean["states"][42][-1])
    assert all(r["gate"]["max_logit_error"] == 0 for r in second.journal.records.values() if r["kind"] == "layer42_validation")
    from filler.dsv4.random_analysis import summarize, audit_exports
    report = summarize(m, second.journal, tmp_path / "analysis")
    assert report["resamples"] == 2000 and audit_exports(tmp_path / "analysis")["passed"]
    for name in ("accuracy", "logprob", "logit"):
        for ext in ("png", "pdf"):
            assert (tmp_path / f"analysis/{name}.{ext}").is_file()
    control = json.loads(Path(r["control_path"]).read_text())
    acks = json.loads(Path(r["ack_path"]).read_text())
    wrong = copy.deepcopy(acks)
    wrong[2]["audits"][0]["max_norm_error"] = float("nan")
    with pytest.raises(ValueError, match="norm"):
        validate_ack_records(wrong, control)
    wrong = copy.deepcopy(acks)
    wrong[2]["positions"].pop(-2)
    with pytest.raises(ValueError, match="position"):
        validate_ack_records(wrong, control)
    path = Path(r["replacement_bank"]["path"])
    original = path.read_bytes()
    bank["states"][0][0] *= -1  # preserves norm, but violates seeded direction
    torch.save(bank, path)
    with pytest.raises(ValueError, match="checksum"):
        verify_records(m, second.journal)
    new_ref = dict(r["replacement_bank"], sha256=file_digest(path))
    with pytest.raises(ValueError, match="seeded reconstruction"):
        validate_bank(new_ref, m, baseline)
    path.write_bytes(original)
    path = Path(r["capture"]["ranks"]["0"]["path"])
    saved["states"][42][positions.index(r["positions"][0])] += 1
    torch.save(saved, path)
    r["capture"]["ranks"]["0"]["sha256"] = file_digest(path)
    acks[0]["capture"]["sha256"] = file_digest(path)
    atomic_json(Path(r["ack_path"]), acks)
    r["ack_sha256"] = file_digest(Path(r["ack_path"]))
    with pytest.raises(ValueError, match="destination tensor"):
        verify_records(m, second.journal)
