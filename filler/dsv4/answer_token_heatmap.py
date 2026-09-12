"""Reuse k=20 residuals and capture k=0, including all Answer: tokens."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time
import uuid

from filler.dsv4.patching import atomic_json, digest, file_digest

WORKSPACE = Path(__file__).resolve().parents[2]
CHECKPOINT = WORKSPACE / "model/DeepSeek-V4-Flash-0731-MoE-MXFP4-BF16"
DEFAULT_ROOT = WORKSPACE / "runs/deepseek-v4-flash/answer-token-heatmap"
HISTORY_ROOT = WORKSPACE / "runs/deepseek-v4-flash/one-fact-patching-discovery"
SOURCE_FILES = (
    "filler/dsv4/answer_token_heatmap.py", "filler/dsv4/top_object_heatmap.py",
    "filler/dsv4/factorial.py", "filler/dsv4/lens.py", "filler/dsv4/campaign_hook.py",
    "filler/dsv4/patching.py", "filler/dsv4/patching_campaign.py",
    "scripts/dsv4/answer_token_heatmap.py", "scripts/dsv4/one_fact_patching.py",
    "scripts/dsv4/run_activation_patching.py", "run_deepseek_v4_a100.sh",
    "run_answer_token_heatmap_allocation.sh", "tests/test_answer_token_heatmap.py",
    "ports/sglang/python/sglang/srt/models/deepseek_v4.py",
    "ports/sglang/python/sglang/srt/layers/mhc_head.py",
    "ports/sglang/python/sglang/srt/layers/layernorm.py",
    "ports/sglang/python/sglang/srt/layers/sampler.py",
    "ports/deepseek-v4-a100-sglang/dsv4_a100_patch/patch.py",
)


def select_positions(tokenizer, ids: list[int], positions: dict, k: int) -> list[dict]:
    """Verify the actual three-token suffix, including the k=0 boundary."""
    end = positions["answer_prompt"]
    if end != len(ids) - 1 or end < 2:
        raise ValueError("answer_prompt must be the final prompt token")
    if [tokenizer.decode([t]) for t in ids[end - 2:end + 1]] != ["Answer", ":", " "]:
        raise ValueError("expected the three distinct tokens Answer, colon, space")
    fillers = positions["fillers"]
    expected = list(range(positions["last_question"] + 1, end - 2))
    if fillers != expected or len(fillers) != k:
        raise ValueError("filler positions do not span the question-to-Answer boundary")
    pairs = [("last_question", positions["last_question"])]
    pairs += [(f"filler_{i}", p) for i, p in enumerate(fillers)]
    pairs += [("answer_word", end - 2), ("answer_colon", end - 1), ("answer_prompt", end)]
    return [{"label": label, "absolute_position": p, "token_id": ids[p],
             "token": tokenizer.decode([ids[p]])} for label, p in pairs]


def checkpoint_files() -> dict:
    return {p.name: {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            for p in sorted(CHECKPOINT.glob("*.safetensors"))}


def reuse_k20(cells: list[dict], hashes: dict, shards: dict) -> tuple[list[dict], dict]:
    """Read only complete, unpatched full-prompt baselines from the later campaign."""
    history = json.loads((HISTORY_ROOT / "manifest.json").read_text())
    complete = json.loads((HISTORY_ROOT / "COMPLETE.json").read_text())
    config_hash = history["config_hash"]
    if (digest({k: v for k, v in history.items() if k != "config_hash"}) != config_hash
            or complete["config_hash"] != config_hash or complete["status"] != "complete"
            or complete["integrity"]["passed"] is not True):
        raise ValueError("historical campaign is not complete and validated")
    if history["checkpoint_files"] != shards:
        raise ValueError("historical checkpoint files differ")
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        key = str((CHECKPOINT / name).relative_to(WORKSPACE))
        if history["source_hashes"][key] != hashes[key]:
            raise ValueError("historical tokenizer/model configuration differs")
    originals = {c["cell_id"]: c for p in history["panels"] for c in p["cells"]}
    baselines, validations = {}, set()
    with (HISTORY_ROOT / "results.jsonl").open() as source:
        for line in source:
            envelope = json.loads(line)
            record = envelope["record"]
            if envelope["sha256"] != digest(record) or record["config_hash"] != config_hash:
                raise ValueError("historical journal checksum mismatch")
            if record["kind"] == "native_validation" and record["report"]["passed"]:
                validations.add(record["runtime_id"])
            if record["kind"] == "baseline" and record["recomputation"] == "full_downstream":
                if record["target_id"] in baselines:
                    raise ValueError("ambiguous repeated historical baseline")
                baselines[record["target_id"]] = record
    for name in ("manifest.json", "COMPLETE.json", "results.jsonl"):
        path = HISTORY_ROOT / name
        hashes[str(path.relative_to(WORKSPACE))] = file_digest(path)
    reused, files = [], {}
    for cell in cells:
        if cell["filler_length"] != 20:
            continue
        old, record = originals[cell["cell_id"]], baselines[cell["cell_id"]]
        if old["input_ids"] != cell["input_ids"] or record["runtime_id"] not in validations:
            raise ValueError("historical input IDs or native validation differ")
        for key, checksum in (("control_path", "control_sha256"), ("ack_path", "ack_sha256")):
            path = Path(record[key])
            if file_digest(path) != record[checksum]:
                raise ValueError("historical control/acknowledgement checksum mismatch")
            hashes[str(path.relative_to(WORKSPACE))] = record[checksum]
        control = json.loads(Path(record["control_path"]).read_text())
        if (control["layers"] or control["clean_capture"] or control["donor_capture"]
                or not control["capture_all"] or control["num_tokens"] != len(cell["input_ids"])):
            raise ValueError("historical capture is not an unpatched complete prompt")
        acks = json.loads(Path(record["ack_path"]).read_text())
        from filler.dsv4.campaign_hook import validate_ack_records
        validate_ack_records(acks, control)
        for rank in range(4):
            ref = record["capture"]["ranks"][str(rank)]
            if acks[rank]["capture"] != ref:
                raise ValueError("historical capture/acknowledgement reference mismatch")
            path = Path(ref["path"])
            files[str(path)] = {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        response = record["response"]
        if file_digest(Path(response["path"])) != response["sha256"]:
            raise ValueError("historical response checksum mismatch")
        hashes[str(Path(response["path"]).relative_to(WORKSPACE))] = response["sha256"]
        reused.append({"cell": cell, "capture": record["capture"], "response": response["path"],
                       "scored_token_ids": record["scored_token_ids"]})
    return reused, files


def prepare(root: Path, readout_probe: Path | None = None) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True, trust_remote_code=True)
    cells, hashes = [], {p: file_digest(WORKSPACE / p) for p in SOURCE_FILES}
    for k in (0, 20):
        path = WORKSPACE / f"runs/deepseek-v4-flash/factorial/filler-discovery-rendered-k{k}.json"
        hashes[str(path.relative_to(WORKSPACE))] = file_digest(path)
        rendered = json.loads(path.read_text())
        for panel in rendered["eligible_panels"]:
            for cell in panel["cells"]:
                ids = cell["input_ids"]
                if tokenizer.encode(cell["rendered_prompt"], add_special_tokens=False) != ids:
                    raise ValueError("current tokenizer differs from saved prompt tokens")
                values = {"A": cell["left_value"], "X": cell["right_value"], "A+X": cell["target"]}
                targets = {}
                for label, value in values.items():
                    encoded = tokenizer.encode(str(value), add_special_tokens=False)
                    if len(encoded) != 1:
                        raise ValueError(f"target is not one token: {label}={value}")
                    targets[label] = encoded[0]
                cells.append({**{key: cell[key] for key in (
                    "cell_id", "panel_id", "split", "row", "col", "left_value", "right_value", "target")},
                    "filler_length": k, "input_ids": ids, "target_token_ids": targets,
                    "positions": select_positions(tokenizer, ids, panel["positions"], k)})
    keys = {(c["filler_length"], c["cell_id"]) for c in cells}
    if len(keys) != len(cells) or any(sum(c["filler_length"] == k for c in cells) != 96 for k in (0, 20)):
        raise ValueError("expected 96 unique examples at each of k=0 and k=20")
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        path = CHECKPOINT / name
        hashes[str(path.relative_to(WORKSPACE))] = file_digest(path)
    shards = checkpoint_files()
    if len(shards) != 48:
        raise ValueError("expected 48 checkpoint shards")
    reused, reused_files = reuse_k20(cells, hashes, shards)
    plan = {"created_at": datetime.now(timezone.utc).isoformat(), "cells": cells,
            "source_hashes": hashes, "checkpoint_files": shards,
            "reused_captures": reused, "reused_capture_files": reused_files,
            "positions": sum(len(c["positions"]) for c in cells),
            "rows": 43 * sum(len(c["positions"]) for c in cells),
            "forward_passes": sum(c["filler_length"] == 0 for c in cells),
            "cohort_source": "complete-prompt outputs paired with each capture: historical k20, fresh k0"}
    if readout_probe is not None:
        readout_probe = readout_probe.resolve()
        control = json.loads((readout_probe / "control.json").read_text())
        acks = json.loads((readout_probe / "acks.json").read_text())
        from filler.dsv4.campaign_hook import validate_ack_records
        validate_ack_records(acks, control)
        if control["layers"] or not control["capture_all"] or control["clean_capture"] or control["donor_capture"]:
            raise ValueError("readout probe must be an unpatched complete prompt")
        cell = next(c for c in cells if c["cell_id"] == control["cell_id"] and c["filler_length"] == 0)
        if digest(cell["input_ids"]) != control["input_ids_hash"]:
            raise ValueError("readout probe prompt differs")
        for name in ("control.json", "acks.json", "response.json"):
            path = readout_probe / name
            hashes[str(path.relative_to(WORKSPACE))] = file_digest(path)
        for ack in acks:
            path = Path(ack["capture"]["path"])
            reused_files[str(path)] = {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        plan["readout_probe"] = {"response": str(readout_probe / "response.json"),
            "capture": {"cell_id": cell["cell_id"], "ranks": {str(a["rank"]): a["capture"] for a in acks}},
            "token_ids": list(set(cell["target_token_ids"].values()))}
    plan["readout"] = "SGLang fused CUDA mHC + RMSNorm + TP=4 vocabulary GEMMs"
    plan["config_hash"] = digest(plan)
    atomic_json(root / "preflight.json", plan)
    print(json.dumps({key: plan[key] for key in ("forward_passes", "positions", "rows", "config_hash")}, indent=2))
    return plan


def verify_plan(plan: dict) -> None:
    if digest({k: v for k, v in plan.items() if k != "config_hash"}) != plan["config_hash"]:
        raise ValueError("preflight checksum mismatch")
    for p, expected in plan["source_hashes"].items():
        if file_digest(WORKSPACE / p) != expected:
            raise ValueError(f"input changed since preflight: {p}")
    if checkpoint_files() != plan["checkpoint_files"]:
        raise ValueError("checkpoint changed since preflight")
    for name, expected in plan["reused_capture_files"].items():
        stat = Path(name).stat()
        if {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns} != expected:
            raise ValueError(f"historical capture file changed since preflight: {name}")


def score_capture(cell: dict, saved: dict, response: dict, weights, numeric_ids, *, projector=None):
    import torch
    from filler.dsv4.lens import project_logits
    projector = project_logits if projector is None else projector

    if set(saved["states"]) != set(range(43)):
        raise ValueError("missing captured layers")
    meta = saved["metadata"]
    if meta["cell_id"] != cell["cell_id"] or meta["num_tokens"] != len(cell["input_ids"]):
        raise ValueError("capture belongs to a different example")
    offsets = {p: i for i, p in enumerate(meta["positions"])}
    correct = response["output_ids"][0] == cell["target_token_ids"]["A+X"]
    for pos in cell["positions"]:
        offset = offsets[pos["absolute_position"]]
        states = torch.stack([saved["states"][layer][offset] for layer in range(43)])
        with torch.inference_mode():
            logits = projector(states.to(weights.lm_head_weight.device), weights)
        if not torch.isfinite(logits).all():
            raise ValueError("nonfinite projected logits")
        top = logits.argmax(-1).tolist()
        numeric = numeric_ids[logits.index_select(-1, numeric_ids).argmax(-1)].tolist()
        if pos["label"] == "answer_prompt" and top[-1] != response["output_ids"][0]:
            raise ValueError("final-layer argmax differs from native generated answer")
        for layer in range(43):
            yield {**{key: cell[key] for key in (
                "panel_id", "cell_id", "split", "row", "col", "left_value", "right_value", "target", "filler_length")},
                "clean_correct": correct, "position_label": pos["label"],
                "absolute_position": pos["absolute_position"], "layer": layer,
                "top_token_id": top[layer], "top_numeric_token_id": numeric[layer],
                "targets": {key: {"token_id": value} for key, value in cell["target_token_ids"].items()}}


def run(root: Path, port: int) -> None:
    import torch
    from transformers import AutoTokenizer
    from filler.dsv4.campaign_hook import campaign_hook_spec
    from filler.dsv4.lens import load_checkpoint_readout, project_sglang_logits
    from filler.dsv4.patching_campaign import HTTPTransport, native_equivalence
    from filler.dsv4.top_object_heatmap import aggregate_rows, canonical_numeric_token_ids, plot_stats, read_jsonl
    from scripts.dsv4.one_fact_patching import get_json, launch_command, server_environment

    plan = json.loads((root / "preflight.json").read_text())
    verify_plan(plan)
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id or not socket.gethostname().startswith("nid"):
        raise RuntimeError("run through srun in an explicitly approved GPU allocation")
    raw = subprocess.check_output(["scontrol", "show", "job", job_id, "-o"], text=True,
                                  env={**os.environ, "TZ": "UTC"})
    allocation = dict(x.split("=", 1) for x in shlex.split(raw) if "=" in x)
    if (allocation.get("Account") != "m5258_g" or allocation.get("NumNodes") != "1"
            or allocation.get("QOS") not in {"interactive", "gpu_interactive"}):
        raise RuntimeError("requires one interactive node with account m5258_g")
    if torch.cuda.device_count() != 4:
        raise RuntimeError("requires four visible GPUs")
    devices = [torch.cuda.get_device_properties(i) for i in range(4)]
    if any("A100" not in d.name or d.total_memory < 79 * 1024**3 for d in devices):
        raise RuntimeError("requires four A100 80 GB GPUs")
    deadline = datetime.fromisoformat(allocation["EndTime"]).replace(tzinfo=timezone.utc).timestamp() - 180
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))
    runtime_id = f"{job_id}-{uuid.uuid4().hex[:8]}"
    output = root / "runtimes" / runtime_id
    output.mkdir(parents=True)
    control_root = output / "control"
    command = launch_command(control_root, port)
    atomic_json(output / "runtime.json", {
        "runtime_id": runtime_id, "job_id": job_id, "hostname": socket.gethostname(),
        "allocation": {k: allocation.get(k) for k in (
            "Account", "QOS", "NodeList", "StartTime", "EndTime", "Features", "TimeLimit")},
        "gpus": [{"name": d.name, "memory_bytes": d.total_memory} for d in devices],
        "command": command, "config_hash": plan["config_hash"], "torch_version": str(torch.__version__),
    })
    if "readout_probe" in plan:
        probe = plan["readout_probe"]
        report = native_equivalence(probe["capture"], json.loads(Path(probe["response"]).read_text()),
            CHECKPOINT, probe["token_ids"], device="cuda:0", projector=project_sglang_logits)
        atomic_json(output / "preload_readout_validation.json", report)
        if not report["passed"]:
            raise RuntimeError("saved-capture readout validation failed BEFORE model load")
        torch.cuda.empty_cache()
        print("Saved-capture native readout validation passed before model load", flush=True)
    def stop(*_args):
        raise KeyboardInterrupt("allocation interrupted")
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    transport = HTTPTransport(f"http://127.0.0.1:{port}", control_root)
    records, validated = list(plan["reused_captures"]), set()
    fresh_cells = [c for c in plan["cells"] if c["filler_length"] == 0]
    with (output / "server.log").open("w") as log:
        server = subprocess.Popen(command, cwd=WORKSPACE, env=server_environment(),
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            startup_deadline = min(deadline, time.time() + 2400)
            while True:
                if server.poll() is not None or time.time() > startup_deadline:
                    raise RuntimeError("model startup failed or timed out; see server.log")
                try:
                    info = get_json(f"http://127.0.0.1:{port}/server_info")
                    break
                except (OSError, ValueError):
                    print(f"Waiting for model; {deadline - time.time():.0f}s remaining", flush=True)
                    time.sleep(15)
            required = {"disable_radix_cache": True, "chunked_prefill_size": -1,
                        "disable_cuda_graph": True, "disable_piecewise_cuda_graph": True,
                        "disable_overlap_schedule": True, "max_running_requests": 1,
                        "enable_return_hidden_states": True}
            if (any(info.get(k) != v for k, v in required.items())
                    or info.get("forward_hooks") != campaign_hook_spec(control_root)
                    or Path(info["model_path"]).resolve() != CHECKPOINT):
                raise RuntimeError("server instrument/configuration mismatch")
            atomic_json(output / "server_config.json", {k: info.get(k) for k in (
                *required, "version", "forward_hooks", "model_path", "dtype", "quantization")})
            for i, cell in enumerate(fresh_cells):
                if time.time() > deadline:
                    raise RuntimeError("allocation deadline reached; partial captures retained")
                k = cell["filler_length"]
                request_id = f"k{k}-{i:03d}"
                control = {"request_id": request_id, "runtime_id": runtime_id,
                    "config_hash": plan["config_hash"], "cell_id": cell["cell_id"],
                    "input_ids_hash": digest(cell["input_ids"]), "num_tokens": len(cell["input_ids"]),
                    "layers": [], "positions": [p["absolute_position"] for p in cell["positions"]],
                    "recomputation": "full_downstream", "capture_all": k not in validated,
                    "clean_capture": None, "donor_capture": None,
                    "output_root": str(output / "passes" / request_id)}
                atomic_json(Path(control["output_root"]) / "control.json", control)
                target_ids = list(set(cell["target_token_ids"].values()))
                response, acks = transport.run(control, cell["input_ids"], target_ids)
                atomic_json(Path(control["output_root"]) / "acks.json", acks)
                capture = {"cell_id": cell["cell_id"], "ranks": {str(a["rank"]): a["capture"] for a in acks}}
                if k not in validated:
                    report = native_equivalence(capture, response, CHECKPOINT, target_ids,
                                                device="cuda:0", projector=project_sglang_logits)
                    atomic_json(output / f"validation_k{k}.json", report)
                    if not report["passed"]:
                        raise RuntimeError("native final-layer equivalence failed")
                    validated.add(k)
                records.append({"cell": cell, "capture": capture,
                                "response": str(Path(control["output_root"]) / "response.json")})
                atomic_json(output / "captures.json", records)
                print(f"Captured {i + 1}/{len(fresh_cells)}: k={k} {cell['cell_id']}", flush=True)
        except BaseException as exc:
            atomic_json(output / "FAILED.json", {"error": str(exc)})
            raise
        finally:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait(timeout=10)

    # Free the model before GPU readout; no second model load is needed.
    historical = plan["reused_captures"][0]
    report = native_equivalence(historical["capture"], json.loads(Path(historical["response"]).read_text()),
                                CHECKPOINT, historical["scored_token_ids"],
                                device="cuda:0", projector=project_sglang_logits)
    atomic_json(output / "validation_k20.json", report)
    if not report["passed"]:
        raise ValueError("reused k20 capture failed native final-layer equivalence")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True, trust_remote_code=True)
    numeric_ids = torch.tensor(canonical_numeric_token_ids(tokenizer), device="cuda:0")
    weights = load_checkpoint_readout(CHECKPOINT, device="cuda:0")
    rows_path = output / "lens_rows_with_numeric_argmax.jsonl"
    count = 0
    with rows_path.open("w") as sink:
        for i, record in enumerate(records):
            if time.time() > deadline:
                raise RuntimeError("allocation deadline reached during projection")
            ref = record["capture"]["ranks"]["0"]
            if file_digest(Path(ref["path"])) != ref["sha256"]:
                raise ValueError("capture checksum mismatch")
            saved = torch.load(ref["path"], map_location="cpu", weights_only=True)
            response = json.loads(Path(record["response"]).read_text())
            for row in score_capture(record["cell"], saved, response, weights, numeric_ids,
                                     projector=project_sglang_logits):
                sink.write(json.dumps(row, separators=(",", ":")) + "\n")
                count += 1
            print(f"Scored {i + 1}/{len(records)}; rows={count}", flush=True)
    if count != plan["rows"]:
        raise ValueError("incomplete output rows")
    cohorts = {}
    for k in (0, 20):
        for cohort, correct in (("all", None), ("correct", True), ("wrong", False)):
            stats = aggregate_rows(read_jsonl(rows_path), filler_length=k, correct=correct)
            denominators = set(stats["example_counts"].values())
            if len(denominators) != 1 or len(stats["positions"]) != k + 4 or len(stats["layers"]) != 43:
                raise ValueError("incomplete plot grid or inconsistent cohorts")
            cohorts[f"k{k}_{cohort}"] = denominators.pop()
            atomic_json(output / f"top_object_heatmap_k{k}_{cohort}.json", stats)
            plot_stats(stats, output / f"top_object_heatmap_k{k}_{cohort}.png")
    verify_plan(plan)
    complete = {"status": "complete", "config_hash": plan["config_hash"],
                "finished_at": datetime.now(timezone.utc).isoformat(), "runtime_id": runtime_id,
                "rows_path": str(rows_path), "rows": count, "cohorts": cohorts,
                "rows_sha256": file_digest(rows_path), "filler_lengths": [0, 20],
                "readout": plan["readout"],
                "validation": "native equivalence per length; final-layer argmax checked for all 192 examples"}
    atomic_json(output / "COMPLETE.json", complete)
    atomic_json(root / "COMPLETE.json", complete)
    print(json.dumps(complete, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--port", type=int, default=30002)
    parser.add_argument("--readout-probe", type=Path,
                        help="during prepare, require a saved capture to pass GPU readout before loading a model")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root.resolve(), args.readout_probe)
    else:
        run(args.root.resolve(), args.port)
