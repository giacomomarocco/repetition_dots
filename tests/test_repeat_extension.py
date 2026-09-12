import copy
import json
import time
from pathlib import Path

import numpy as np
import pytest

from filler.dsv4.patching import build_manifest, campaign_counts, layer42_diagnostics, digest, atomic_json, file_digest
from test_one_fact_patching import rendered, Tokenizer, make_campaign
from test_filler_repeat import RepeatTransport


def test_four_sources_counts_and_controls():
    m = build_manifest(rendered(), Tokenizer(), "filler-repeat", repeat_sources=[1, 2, 3, 4])
    assert m["counts"] == dict(panels=24, targets=96, baselines=96, trials=384, identity=384)
    assert len(layer42_diagnostics(m, m["pilot_panel"])) == 16
    assert m["repeat_sources"] == [1, 2, 3, 4]
    for s in m["trials"] + m["identity_controls"]:
        i = s["source_filler_index"]
        assert s["positions"] == list(range(12+i, 31))
        assert s["source_positions"] == (s["positions"] if s["kind"] == "identity" else [11+i]*(19-i))
        assert s["layers"] == list(range(43))
    assert sum(m["counts"][k] for k in ("baselines", "trials", "identity")) + 16 == 880


@pytest.mark.parametrize("sources", [[], [1, 1], [-1], [19], [True], [1.5]])
def test_invalid_sources(sources):
    with pytest.raises(ValueError, match="repeat_sources"):
        build_manifest(rendered(), Tokenizer(), "filler-repeat", repeat_sources=sources)


def test_cli_prepared_extension_requires_new_root():
    from scripts.dsv4.one_fact_patching import parse_args
    args = parse_args(["prepare", "--design", "filler-repeat", "--root", "/tmp/new-repeat",
                       "--repeat-sources", "1", "2", "3", "4", "--five-shot-k5"])
    assert args.repeat_sources == [1, 2, 3, 4] and args.five_shot_k5
    for args in (["run", "--repeat-sources", "1"], ["prepare", "--design", "filler-repeat", "--repeat-sources", "1"]):
        with pytest.raises(SystemExit):
            parse_args(args)


def test_four_source_restart_and_independent_pair_bootstrap(tmp_path):
    from filler.dsv4.repeat_analysis import summarize, result_rows, export_rows
    m = build_manifest(rendered(), Tokenizer(), "filler-repeat", repeat_sources=[1, 2, 3, 4])
    m["panels"] = m["panels"][:2]
    pids = {p["panel_id"] for p in m["panels"]}
    for key in ("trials", "identity_controls"):
        m[key] = [s for s in m[key] if s["panel_id"] in pids]
    m["counts"] = campaign_counts(m)
    first = make_campaign(m, tmp_path, "old", RepeatTransport(m, fail_at=50))
    with pytest.raises(InterruptedError):
        first.run()
    saved = copy.deepcopy([r for r in first.journal.records.values() if r["kind"] == "trial"])
    assert saved
    second = make_campaign(m, tmp_path, "new", RepeatTransport(m))
    integrity = second.run()
    assert integrity == dict(trials=32, identity=32, matched_baseline_cells=8, passed=True)
    assert all(second.journal.records[r["record_id"]] == r for r in saved)
    report = summarize(m, second.journal, tmp_path / "analysis", integrity=integrity)
    assert len(report["accuracy_counts"]) == 4
    assert len(report["paired_comparisons"]) == 6*11
    # A deliberately nonconstant paired dataset catches accidental unpaired draws.
    rows = result_rows(m, second.journal)
    for i, row in enumerate(rows):
        row["delta_logprob"] = (i % 7) * .13
    sites = [f"repeat_filler_{i}" for i in range(1, 5)]
    report = export_rows(m["panels"], sites, rows, tmp_path / "synthetic", {})
    inds = np.random.default_rng(42).integers(0, 2, (2000, 2))
    for e in report["paired_comparisons"]:
        a, b = e["contrast"].split("_minus_")
        vals = np.array([np.mean([r[e["metric"]] for r in rows if r["site"] == a and r["panel_id"] == p]) -
                         np.mean([r[e["metric"]] for r in rows if r["site"] == b and r["panel_id"] == p]) for p in pids])
        # Panel order is irrelevant to this two-panel percentile/SE comparison.
        draws = vals[inds].mean(1)
        assert e["mean"] == pytest.approx(vals.mean())
        # Match the actual ordered panels for finite Monte Carlo SE.
        ordered = [p["panel_id"] for p in m["panels"]]
        vals = np.array([np.mean([r[e["metric"]] for r in rows if r["site"] == a and r["panel_id"] == p]) -
                         np.mean([r[e["metric"]] for r in rows if r["site"] == b and r["panel_id"] == p]) for p in ordered])
        draws = vals[inds].mean(1)
        assert e["bootstrap_se"] == pytest.approx(draws.std(ddof=1))
        assert [e["ci_low"], e["ci_high"]] == pytest.approx(np.quantile(draws, [.025, .975]))


def k5_fixture(tmp_path):
    from filler.addition import one_fact as of
    from test_one_fact_addition_sglang import fake_encoder
    history, output, combined = [tmp_path/p for p in ("history", "k5", "combined")]
    output.mkdir()
    facts = [dict(fact_id=f"fact{i}", question=f"Fact {i}?", answer=2) for i in range(262)]
    old = []
    for t in of.make_tasks(facts, 42, 1, [0, 10, 20, 50, 100]):
        old.append({**t, "rendered_prompt": of.render_prompt(fake_encoder, t), "response": str(t["target"]),
                    "correct": True, "target_log_probability": -.1, "target_top_rank": 1})
    atomic_json(history / "results.json", old)
    config = {"source": {"selected_facts": 262}, "mode": "generation", "filler_lengths": [0,10,20,50,100],
              "demonstrations": [1]*5}
    atomic_json(history / "run_config.json", config)
    prompts = [{**t, "rendered_prompt": of.render_prompt(fake_encoder, t), "target_token_id": t["target"]}
               for t in of.make_tasks(facts, 42, 1, [5])]
    path = output / "prepared-prompts.json"
    atomic_json(path, prompts)
    spec = {"config": {**config, "filler_lengths": [5]}, "output": str(output), "history": str(history),
            "combined_output": str(combined), "prompts": str(path),
            "input_hashes": {str(p): file_digest(p) for p in (history / "results.json", path)}}
    spec["config_hash"] = digest(spec)
    return spec, prompts


def test_k5_restart_and_preserved_historical_cohort(tmp_path, monkeypatch):
    from filler.addition import k5_extension as ext
    from filler.addition import one_fact as of
    from filler.dsv4.patching_campaign import WalltimeReached
    spec, prompts = k5_fixture(tmp_path)
    old_bytes = (Path(spec["history"])/"results.json").read_bytes()
    calls = []
    monkeypatch.setattr(of, "validate_one_token_target", lambda endpoint,prompt,target,timeout: target)
    def generate(endpoint, prompt, timeout, limit, token_id, top):
        assert limit == 8 and top == 20 and timeout == 600
        calls.append(prompt)
        meta = {"output_token_ids_logprobs": [[[-.1, token_id]]], "output_top_logprobs": [[[-.1, token_id]]]}
        return str(token_id), meta
    monkeypatch.setattr(of, "request_generation_with_limit", generate)
    with pytest.raises(WalltimeReached):
        ext.run(spec, "http://server/generate", {"runtime_id": "old", "job_id": "1"},
                control_root=tmp_path/"control", deadline=time.time()+10000, should_stop=lambda: len(calls)==7)
    assert len(calls) == 7
    ext.run(spec, "http://server/generate", {"runtime_id": "new", "job_id": "2"},
            control_root=tmp_path/"control2", deadline=time.time()+10000)
    assert len(calls) == 262 and len(set(calls)) == 262
    assert (Path(spec["history"])/"results.json").read_bytes() == old_bytes
    rows = json.loads((Path(spec["output"])/"results.json").read_text())
    assert sum(r["runtime_id"] == "old" for r in rows) == 7
    summary = json.loads((Path(spec["combined_output"])/"summary.json").read_text())
    assert summary["result_count"] == 1572 and len(summary["conditions"]) == 6
    assert all(v["count"] == 262 for v in summary["conditions"].values())
    assert all(r["rendered_prompt"].count(of.answer_slot(5)) == 6 for r in rows)
    atomic_json(tmp_path/"armed/NEXT.json", {"request_id": "stale"})
    with pytest.raises(RuntimeError, match="precede arming"):
        ext.run(spec, "http://server/generate", {}, control_root=tmp_path/"armed", deadline=time.time()+10000)
    path = Path(spec["output"])/"results_progress.jsonl"
    data = [json.loads(line) for line in path.read_text().splitlines()]
    data[0]["target_log_probability"] += 1
    path.write_text("\n".join(map(json.dumps, data))+"\n")
    with pytest.raises(ValueError, match="checksum|score changed"):
        ext.load_results(spec, "http://server/generate")


def test_k5_partial_tail_quarantine_and_complete_corruption(tmp_path):
    from filler.addition.k5_extension import recover_progress
    p = tmp_path / "results_progress.jsonl"
    p.write_bytes(b'{"completed": 1}\n{"interrupted":')
    recover_progress(p)
    assert p.read_bytes() == b'{"completed": 1}\n'
    assert len(list(tmp_path.glob("interrupted-tail-*.bin"))) == 1
    p.write_bytes(b'{"completed": 2}')
    recover_progress(p)
    assert p.read_bytes() == b'{"completed": 2}\n'
    p.write_bytes(b'invalid complete record\n')
    recover_progress(p)
    assert p.read_bytes() == b'invalid complete record\n'


def test_hooks_pass_through_unarmed_prefill_and_decode(tmp_path):
    import torch
    from types import SimpleNamespace
    from filler.dsv4.campaign_hook import make_campaign_hook
    from filler.dsv4.patching_logits import make_logits_hook
    residual = make_campaign_hook(dict(control_root=str(tmp_path), num_layers=43))
    logits = make_logits_hook(dict(control_root=str(tmp_path)))
    for tokens in (71, 1, 1, 1):
        x = torch.arange(tokens*6).reshape(tokens, 2, 3).float()
        for lid in range(43):
            module = SimpleNamespace(layer_id=lid, use_fused_mhc_post_pre=False)
            assert residual(module, (), x) is x
        out = SimpleNamespace(next_token_logits=torch.zeros(1, 100))
        assert logits(None, (), out) is out
    assert not list(tmp_path.iterdir())


def test_notebook_export_has_two_figures_and_six_lengths(tmp_path):
    from filler.dsv4.repeat_extension import refresh_notebook
    import shutil
    p = tmp_path / "notebooks/addition_accuracy.ipynb"
    p.parent.mkdir()
    shutil.copy2(Path(__file__).resolve().parents[1]/"notebooks/addition_accuracy.ipynb", p)
    base = tmp_path / "runs/deepseek-v4-flash"
    for dirname, ks, n in (("one-fact-addition-5shot-with-k5", [0,5,10,20,50,100], 262),
                           ("two-fact-addition-5shot-cyclic", [0,10,20,50,100], 260)):
        atomic_json(base/dirname/"run_config.json", {"mode":"generation", "filler_lengths":ks,
            "demonstrations":[1]*5, "source":{"selected_facts":n}, "created_at_utc":"fixture"})
        atomic_json(base/dirname/"summary.json", {"result_count":n*len(ks), "complete":True,
            "conditions":{"baseline" if k==0 else f"dots_{k}":{"count":n,"correct":n//2} for k in ks}})
    result = refresh_notebook(tmp_path)
    assert result["passed"] and result["embedded_pngs"] == 2
    assert (base/"plots/one-fact-addition-5shot-accuracy-vs-filler.png").is_file()
    cells = json.loads(p.read_text())["cells"]
    assert any('dots_5' in str(c.get('outputs')) or '    5 ' in str(c.get('outputs')) for c in cells)
