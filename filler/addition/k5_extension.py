"""Frozen five-shot k=5 supplement, resumable on the campaign-owned server."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import time

from filler.addition import one_fact as of
from filler.dsv4.patching import atomic_bytes, atomic_json, digest, file_digest


def verify(spec):
    if digest({k: v for k, v in spec.items() if k != "config_hash"}) != spec["config_hash"]:
        raise ValueError("k5 configuration hash mismatch")
    for name, checksum in spec["input_hashes"].items():
        if file_digest(Path(name)) != checksum:
            raise ValueError(f"k5 input changed: {name}")


def prepare(workspace: Path, tokenizer):
    base = (workspace / "runs/deepseek-v4-flash").resolve()
    history, output = base / "one-fact-addition-5shot-full", base / "one-fact-addition-5shot-k5"
    config = json.loads((history / "run_config.json").read_text())
    old = json.loads((history / "results.json").read_text())
    if (config["filler_lengths"] != [0, 10, 20, 50, 100]
            or config["source"]["selected_facts"] != 262 or len(old) != 1310
            or config["seed"] != 42 or config["addends_per_fact"] != 1
            or config["decoding"] != {"temperature": 0, "max_new_tokens": 8}
            or config["system_prompt"] != of.SYSTEM_PROMPT
            or config.get("prompt_variant", "local") != "local"
            or not json.loads((history / "summary.json").read_text())["complete"]):
        raise ValueError("historical five-shot evaluation protocol changed")
    if Path(config["model_id"]).resolve() != (workspace / "model/DeepSeek-V4-Flash-0731-MoE-MXFP4-BF16").resolve():
        raise ValueError("k5 checkpoint differs from campaign")
    facts, sha = of.load_facts(Path(config["source"]["path"]))
    if sha != config["source"]["sha256"] or len(facts) != 262:
        raise ValueError("historical fact cohort changed")
    encoder = of.load_encoder(Path(config["encoder"]))
    expected = of.make_tasks(facts, 42, 1, config["filler_lengths"])
    lookup = {r["prompt_id"]: r for r in old}
    if len(lookup) != 1310 or set(lookup) != {r["prompt_id"] for r in expected}:
        raise ValueError("historical prompt membership changed")
    for task in expected:
        row = lookup[task["prompt_id"]]
        if (any(row.get(k) != v for k, v in task.items())
                or row["rendered_prompt"] != of.render_prompt(encoder, task)
                or row["correct"] != (of.parse_answer(row["response"]) == row["target"])):
            raise ValueError("historical prompts, pair IDs or correctness labels differ")
    prompts = []
    for task in of.make_tasks(facts, 42, 1, [5]):
        prompt = of.render_prompt(encoder, task)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        together = tokenizer.encode(prompt + str(task["target"]), add_special_tokens=False)
        canonical = tokenizer.encode(str(task["target"]), add_special_tokens=False)
        if (together != ids + canonical or len(canonical) != 1
                or tokenizer.decode(canonical) != str(task["target"])
                or prompt.count(of.answer_slot(5)) != 6):
            raise ValueError("k5 target is not a canonical single continuation token")
        prompts.append({**task, "rendered_prompt": prompt, "target_token_id": canonical[0]})
    output.mkdir(parents=True, exist_ok=True)
    prompt_path = output / "prepared-prompts.json"
    if prompt_path.exists() and json.loads(prompt_path.read_text()) != prompts:
        raise ValueError("prepared k5 prompts changed; use a new root")
    atomic_json(prompt_path, prompts)
    config = {**config, "filler_lengths": [5], "prompt_variant": "local", "mode": "generation"}
    config.pop("created_at_utc", None)
    config.pop("repository_revision", None)
    paths = [history / f for f in ("run_config.json", "results.json", "summary.json")]
    paths += [Path(config["source"]["path"]), Path(config["encoder"]), prompt_path]
    spec = {"config": config, "output": str(output), "history": str(history),
            "combined_output": str(base / "one-fact-addition-5shot-with-k5"),
            "prompts": str(prompt_path), "requests": 262, "request_timeout": 600,
            "input_hashes": {str(p): file_digest(p) for p in paths}}
    spec["config_hash"] = digest(spec)
    previous = output / "prepared.json"
    if previous.exists() and (output / "results_progress.jsonl").exists() and json.loads(previous.read_text()) != spec:
        raise ValueError("cannot change k5 configuration after results exist")
    atomic_json(previous, spec)
    return spec


def load_results(spec, endpoint):
    root = Path(spec["output"])
    config = {**spec["config"], "endpoint": endpoint, "extension_config_hash": spec["config_hash"]}
    previous = json.loads((root / "run_config.json").read_text()) if (root / "run_config.json").exists() else None
    if previous is not None and previous.get("extension_config_hash") != spec["config_hash"]:
        raise ValueError("k5 resume configuration changed")
    prompts = json.loads(Path(spec["prompts"]).read_text())
    results = of.load_resume_results(root / "results_progress.jsonl", prompts, previous, config)
    expected = {p["prompt_id"]: p for p in prompts}
    for r in results:
        if r.get("record_sha256") != digest({k: v for k, v in r.items() if k != "record_sha256"}):
            raise ValueError("k5 result checksum mismatch")
        if (any(r.get(k) != v for k, v in expected[r["prompt_id"]].items())
                or r["correct"] != (of.parse_answer(r["response"]) == r["target"])):
            raise ValueError("k5 resume prompt or correctness changed")
        score = of.extract_target_score(r["server_metadata"], r["target_token_id"], 20)
        if any(r.get(k) != v for k, v in score.items()):
            raise ValueError("k5 resume score changed")
    return prompts, config, results


def recover_progress(path):
    """Quarantine only an interrupted final line; never repair corrupt full rows."""
    if not path.exists():
        return
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        prefix, _, tail = data.rpartition(b"\n")
        try:
            json.loads(tail)
        except (ValueError, UnicodeDecodeError):
            atomic_bytes(path.with_name(f"interrupted-tail-{time.time_ns()}.bin"), tail)
            atomic_bytes(path, prefix + b"\n" if prefix else b"")
        else:
            atomic_bytes(path, data + b"\n")


def run(spec, endpoint, runtime, *, control_root, deadline, should_stop=lambda: False):
    from filler.dsv4.patching_campaign import WalltimeReached
    verify(spec)
    if (Path(control_root) / "NEXT.json").exists():
        raise RuntimeError("ordinary k5 requests must precede arming intervention controls")
    root = Path(spec["output"])
    with (root / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        recover_progress(root / "results_progress.jsonl")
        prompts, config, results = load_results(spec, endpoint)
        atomic_json(root / "run_config.json", config)
        atomic_json(root / "prompts.json", prompts)
        done = {r["prompt_id"] for r in results}
        try:
            with (root / "results_progress.jsonl").open("a") as sink:
                for row in prompts:
                    if row["prompt_id"] in done:
                        continue
                    if should_stop() or time.time() >= deadline - 780:
                        raise WalltimeReached("checkpointed during k5 evaluation")
                    if (Path(control_root) / "NEXT.json").exists():
                        raise RuntimeError("intervention control armed during ordinary k5 evaluation")
                    started = time.perf_counter()
                    token_id = of.validate_one_token_target(endpoint, row["rendered_prompt"], row["target"], 600)
                    if token_id != row["target_token_id"]:
                        raise ValueError("server k5 continuation token differs from CPU preflight")
                    response, metadata = of.request_generation_with_limit(endpoint, row["rendered_prompt"], 600, 8, token_id, 20)
                    parsed = of.parse_answer(response)
                    result = {**row, **of.extract_target_score(metadata, token_id, 20),
                        "response": response, "parsed_answer": parsed, "correct": parsed == row["target"],
                        "generation_seconds": time.perf_counter() - started, "server_metadata": metadata,
                        "runtime_id": runtime["runtime_id"], "job_id": runtime["job_id"]}
                    result["record_sha256"] = digest(result)
                    sink.write(json.dumps(result, ensure_ascii=False) + "\n")
                    sink.flush()
                    os.fsync(sink.fileno())
                    results.append(result)
                    print(f"k5 {len(results)}/262: target={row['target']} response={response!r}", flush=True)
        finally:
            atomic_json(root / "results.json", results)
            summary = of.summarize(results) if results else {"result_count": 0, "conditions": {}}
            summary.update(complete=len(results) == len(prompts), prompt_count=len(prompts))
            atomic_json(root / "summary.json", summary)
            runtime_path = root / "runtimes" / f"{runtime['runtime_id']}.json"
            atomic_json(runtime_path, {**runtime, "extension_config_hash": spec["config_hash"]})
    verify(spec)
    return merge(spec)


def merge(spec):
    """Build a derived view; historical artifacts are never rewritten."""
    verify(spec)
    old_root, new_root, out = map(Path, (spec["history"], spec["output"], spec["combined_output"]))
    new_config = json.loads((new_root / "run_config.json").read_text())
    _, _, new = load_results(spec, new_config["endpoint"])
    if len(new) != 262:
        raise ValueError("cannot merge incomplete k5 evaluation")
    old = json.loads((old_root / "results.json").read_text())
    rows = [{**r, "origin": "historical", "source_run": str(old_root)} for r in old]
    rows += [{**r, "origin": "k5_extension", "source_run": str(new_root)} for r in new]
    cohorts = {k: {r["pair_id"] for r in rows if r["k"] == k} for k in (0, 5, 10, 20, 50, 100)}
    if len(rows) != 1572 or any(len(c) != 262 or c != cohorts[0] for c in cohorts.values()):
        raise ValueError("six-length cohort identity/count mismatch")
    summary = of.summarize(rows)
    summary.update(complete=True, prompt_count=1572, source_runs=[str(old_root), str(new_root)],
        caveat="k5 is a new runtime; other lengths are historical. Pointwise intervals do not estimate runtime variance.")
    atomic_json(out / "results.json", rows)
    atomic_json(out / "summary.json", summary)
    historical_config = json.loads((old_root / "run_config.json").read_text())
    atomic_json(out / "run_config.json", {**historical_config, "filler_lengths": [0, 5, 10, 20, 50, 100],
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "derived_view": True,
        "source_runs": [str(old_root), str(new_root)]})
    provenance = {"prepared": spec, "result_sha256": {str(p): file_digest(p) for p in
                  (old_root / "results.json", new_root / "results.json")},
                  "runtime_ids": sorted({r["runtime_id"] for r in new}),
                  "historical_records_preserved": True}
    atomic_json(out / "reproducibility.json", provenance)
    from filler.addition.accuracy_plot import load_accuracy
    points = load_accuracy(out / "summary.json")
    with (out / "accuracy.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, list(points[0])); w.writeheader(); w.writerows(points)
    keys = ["pair_id", "fact_id", "k", "target", "response", "correct", "target_log_probability", "origin", "source_run"]
    with (out / "per_example.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    lines = ["# Five-shot one-fact addition with k=5", "", summary["caveat"], "",
             "Same 262 facts and addends, five fixed demonstrations, identical k in every demonstration and target. Greedy decoding, up to 8 tokens; exact integer completion scoring. Intervals are pointwise 95% Wilson intervals.", "",
             "| k | Correct / 262 | Accuracy | 95% CI |", "|---:|---:|---:|---:|"]
    for p in points:
        lines.append(f"| {p['filler_length']} | {p['correct']}/262 | {100*p['accuracy']:.2f}% | [{100*p['ci_low']:.2f}, {100*p['ci_high']:.2f}]% |")
    lines += ["", "[Per-example CSV](per_example.csv) · [Summary](summary.json) · [Reproducibility](reproducibility.json)", ""]
    atomic_bytes(out / "REPORT.md", "\n".join(lines).encode())
    atomic_json(new_root / "COMPLETE.json", {"config_hash": spec["config_hash"], "complete": True, "requests": 262})
    entry = f"\n## {datetime.now(timezone.utc).isoformat()} — k=5 completed\n\n262 requests; historical 1,310 records retained in the derived six-length view. See [report]({out / 'REPORT.md'}). Config {spec['config_hash']}.\n"
    lab = new_root / "LAB_LOG.md"
    existing = lab.read_text() if lab.exists() else "# Five-shot k=5 lab log\n"
    if spec["config_hash"] not in existing:
        atomic_bytes(lab, (existing + entry).encode())
    return summary
