"""Three-phase startup comparison, inside separately approved Slurm allocations.

Never submits/cancels jobs. Owns and stops only the server process it launches.
Use sequential for cold, same-node restart, and same-node bundle restoration.
Fresh-node validation remains a separate phase on a different approved node.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import time

from filler.dsv4.startup_cache import WORKSPACE, atomic_json, digest, file_hash, publish
from filler.dsv4.startup_timing import emit, span

CHECKS = ["tests/test_dsv4_startup.py", "tests/test_dsv4_factorial.py",
          "tests/test_deepseek_v4_logit_lens.py", "tests/test_one_fact_patching.py"]


def compare(baseline, result):
    tokens_equal = result["tokens"] == baseline["tokens"]
    ratio = result["warm_tokens_per_second"] / baseline["warm_tokens_per_second"]
    return {"tokens_equal": tokens_equal, "throughput_ratio": ratio,
            "native_passed": result["native"]["passed"],
            "passed": tokens_equal and ratio >= 0.95 and result["native"]["passed"]}


def phase_sequence(phase):
    return {"cold-same": ("cold", "same"),
            "sequential": ("cold", "same", "restored")}.get(phase, (phase,))


def source_changes(baseline, current, allow_driver_update=False):
    changed = sorted(k for k in baseline.keys() | current.keys() if baseline.get(k) != current.get(k))
    permitted = {"scripts/dsv4/benchmark_startup.py", "tests/test_dsv4_startup.py"}
    if changed and (not allow_driver_update or not set(changed) <= permitted):
        raise ValueError("workflow sources changed since cold baseline")
    return changed


def timing_summary(root):
    events = [json.loads(line) for path in sorted(root.glob("*.jsonl"))
              for line in path.read_text().splitlines()]
    return {"compile_count": sum(e.get("compile_count", 0) for e in events),
            "build_count": sum(e.get("build_count", 0) for e in events),
            "instrumented_ranks": sorted({e["rank"] for e in events
                if e["event"] == "weights.assignment_and_transfer.end"}, key=str),
            "spans": [e for e in events if e["event"].endswith(".end")]}


def run(args):
    from scripts.dsv4.one_fact_patching import (
        CHECKPOINT, DEFAULT_ROOT, SOURCES, allocation_deadline, checkpoint_files,
        get_json, launch_command, server_environment)
    from filler.dsv4.campaign_hook import campaign_hook_spec
    from filler.dsv4.patching_campaign import HTTPTransport, native_equivalence
    from scripts.dsv4.run_activation_patching import post_json
    from filler.dsv4.factorial import NativeResidualHooks  # preflight the required integration

    root = args.root.resolve()
    run_root = root / args.phase
    sources = {name: file_hash(WORKSPACE / name) for name in (*SOURCES, *CHECKS,
        "filler/dsv4/startup_cache.py", "filler/dsv4/startup_timing.py",
        "filler/dsv4/startup_bootstrap/sitecustomize.py", "scripts/dsv4/startup_cache.py",
        "scripts/dsv4/benchmark_startup.py", "run_deepseek_v4_startup.sh")}
    checkpoint = {"shards": checkpoint_files(), "config": file_hash(CHECKPOINT / "config.json")}
    if run_root.exists():
        raise ValueError(f"phase output already exists: {run_root}; use a new comparison root")
    if args.phase != "cold":
        baseline = json.loads((root / "cold/result.json").read_text())
        if baseline.get("passed") is not True:
            raise ValueError("a validated cold phase is required")
        changes = source_changes(baseline["source_hashes"], sources, args.allow_driver_update)
        if checkpoint != baseline["checkpoint"]:
            raise ValueError("checkpoint changed since cold baseline")
        if (socket.gethostname() == baseline["hostname"]) != (args.phase in {"same", "restored", "rebuild"}):
            raise ValueError("same/restored require the original node; fresh requires a different node")
        if args.phase in {"fresh", "restored"}:
            same = json.loads((root / "same/result.json").read_text())
            if not same.get("native", {}).get("passed") or not same.get("comparison", {}).get("tokens_equal"):
                raise ValueError("same-node correctness must pass before bundle restoration comparison")
    deadline = allocation_deadline()
    if deadline - time.time() < 2400:
        raise RuntimeError("need at least 40 minutes left in the approved allocation")
    import torch
    devices = [torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]
    if len(devices) != 4 or any("A100" not in d.name or d.total_memory < 79 * 1024**3 for d in devices):
        raise RuntimeError("requires four A100 80 GB GPUs on the compute node")
    allocation_raw = subprocess.check_output(["scontrol", "show", "job", os.environ["SLURM_JOB_ID"], "-o"],
        text=True, env={**os.environ, "TZ": "UTC"})
    import shlex
    allocation = dict(p.split("=", 1) for p in shlex.split(allocation_raw) if "=" in p)
    if allocation.get("Account") != "m5258_g":
        raise RuntimeError("requires account m5258_g")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    run_root.mkdir(parents=True)
    if args.phase == "cold":
        manifest = json.loads((DEFAULT_ROOT / "manifest.json").read_text())
        cells = manifest["panels"][0]["cells"][:3]
        prompts = [{"cell_id": c["cell_id"], "input_ids": c["input_ids"]} for c in cells]
        atomic_json(root / "prompts.json", prompts)
    else:
        prompts = json.loads((root / "prompts.json").read_text())
        if digest(prompts) != baseline["prompts_hash"]:
            raise ValueError("prompt fixtures changed")
    command = launch_command(run_root / "control", args.port)
    command[1] = str(WORKSPACE / "run_deepseek_v4_startup.sh")
    started = time.time()
    env = {**server_environment(), "DSV4_KERNEL_CACHE_MODE": "reuse",
           "DSV4_KERNEL_CACHE_ROOT": str(root / "bundles"), "DSV4_STARTUP_TIMING": "1",
           "DSV4_STARTUP_TIMING_DIR": str(run_root / "timing"), "DSV4_STARTUP_RUN_DIR": str(run_root),
           "DSV4_STARTUP_EPOCH": str(started), "DSV4_KERNEL_CACHE_FRESH": "1" if args.phase in {"cold", "rebuild"} else "0",
           "DSV4_KERNEL_CACHE_RESTORE": "1" if args.phase == "restored" else "0"}
    os.environ.update({k: env[k] for k in ("DSV4_STARTUP_TIMING", "DSV4_STARTUP_TIMING_DIR", "DSV4_STARTUP_EPOCH")})
    result = {"phase": args.phase, "hostname": socket.gethostname(), "job_id": os.environ["SLURM_JOB_ID"],
              "allocation": {k: allocation.get(k) for k in ("Account", "QOS", "NodeList", "StartTime", "EndTime", "Features")},
              "gpus": [{"name": d.name, "memory_bytes": d.total_memory} for d in devices],
              "prompts_hash": digest(prompts), "command": command, "start_time": started, "passed": False,
              "source_hashes": sources, "checkpoint": checkpoint, "warm_cycles": args.warm_cycles,
              "driver_changes_from_baseline": changes if args.phase != "cold" else []}
    atomic_json(run_root / "runtime.json", result)
    base_url = f"http://127.0.0.1:{args.port}"

    def interrupted(signum, frame):
        raise InterruptedError(f"received signal {signum}")

    previous_handlers = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)}
    server = None
    try:
        with (run_root / "server.log").open("w") as log:
            server = subprocess.Popen(command, cwd=WORKSPACE, env=env, stdout=log,
                                      stderr=subprocess.STDOUT, start_new_session=True)
            with span("server.readiness"):
                while True:
                    if server.poll() is not None:
                        raise RuntimeError(f"server exited {server.returncode}; see {run_root / 'server.log'}")
                    if time.time() > min(started + 2100, deadline - 300):
                        raise TimeoutError("server readiness deadline")
                    try:
                        info = get_json(base_url + "/server_info")
                        break
                    except (OSError, ValueError):
                        time.sleep(5)
            result["readiness_s"] = time.time() - started
            requirements = {"tp_size": 4, "disable_cuda_graph": True, "disable_radix_cache": True,
                "disable_piecewise_cuda_graph": True, "disable_overlap_schedule": True,
                "chunked_prefill_size": -1, "max_running_requests": 1, "enable_return_hidden_states": True,
                "forward_hooks": campaign_hook_spec(run_root / "control"), "model_path": str(CHECKPOINT)}
            for key, value in requirements.items():
                if info.get(key) != value:
                    raise RuntimeError(f"server requirement differs: {key}={info.get(key)!r}")
            result["server_settings"] = {k: info.get(k) for k in (
                *requirements, "dtype", "quantization", "kv_cache_dtype", "attention_backend", "mem_fraction_static")}
            first = prompts[0]
            output = run_root / "native"
            control = {"request_id": "startup-native", "runtime_id": run_root.name, "config_hash": digest(prompts),
                "cell_id": first["cell_id"], "input_ids_hash": digest(first["input_ids"]),
                "num_tokens": len(first["input_ids"]), "layers": [], "positions": [],
                "recomputation": "full_downstream", "capture_all": True, "clean_capture": None,
                "donor_capture": None, "output_root": str(output)}
            with span("response.first_validated", gpu=False):
                response, acks = HTTPTransport(base_url, run_root / "control").run(control, first["input_ids"], [0, 1])
                result["first_response_s"] = time.time() - started
                capture = {"ranks": {str(a["rank"]): a["capture"] for a in acks}}
                result["native"] = native_equivalence(capture, response, CHECKPOINT, [0, 1])
                atomic_json(output / "equivalence.json", result["native"])
                if not result["native"]["passed"]:
                    raise RuntimeError("native final-layer equivalence failed")
            result["first_validated_s"] = time.time() - started
            emit("response.validated", elapsed_s=result["first_validated_s"])
            tokens, rates = None, []
            # Discard the first full cycle; keep every subsequent cycle, including
            # slow ones. Follow-up runs can collect a longer sustained sample.
            for cycle in range(args.warm_cycles + 1):
                batch, token_count, elapsed = [], 0, 0.0
                for prompt in prompts:
                    if deadline - time.time() < 180:
                        raise TimeoutError("validation walltime reserve reached")
                    request_start = time.perf_counter()
                    answer = post_json(base_url + "/generate", {"input_ids": prompt["input_ids"],
                        "sampling_params": {"temperature": 0, "max_new_tokens": 16}}, timeout=120)
                    elapsed += time.perf_counter() - request_start
                    batch.append(answer["output_ids"])
                    token_count += len(answer["output_ids"])
                    atomic_json(run_root / "responses" / f"{cycle}-{len(batch)}.json", answer)
                if tokens is not None and batch != tokens:
                    raise RuntimeError("greedy tokens changed between repetitions")
                tokens = batch
                if cycle:
                    rates.append(token_count / elapsed)
            result.update(tokens=tokens, warm_cycle_tokens_per_second=rates,
                          warm_tokens_per_second=statistics.median(rates))
            if args.phase == "cold":
                result["comparison"] = {"passed": True, "baseline": True}
            else:
                result["comparison"] = compare(baseline, result)
                for key in result["server_settings"]:
                    if key != "forward_hooks" and result["server_settings"][key] != baseline["server_settings"][key]:
                        raise RuntimeError(f"serving setting changed: {key}")
                if not result["comparison"]["tokens_equal"]:
                    raise RuntimeError("token equivalence failed")
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        if server is not None:
            try:
                os.killpg(server.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait(timeout=30)
        atomic_json(run_root / "result.json", result)
    summary = timing_summary(run_root / "timing")
    atomic_json(run_root / "timing-summary.json", summary)
    if summary["instrumented_ranks"] != [0, 1, 2, 3]:
        raise RuntimeError("missing timing from one or more TP ranks")
    if any(file_hash(WORKSPACE / name) != value for name, value in sources.items()):
        raise RuntimeError("workflow sources changed during the benchmark")
    if checkpoint != {"shards": checkpoint_files(), "config": file_hash(CHECKPOINT / "config.json")}:
        raise RuntimeError("checkpoint changed during the benchmark")
    cache = json.loads((run_root / "cache.json").read_text())
    expected = {"cold": "empty", "same": "same_node", "fresh": "restored", "restored": "restored", "rebuild": "empty"}[args.phase]
    if cache["restore_status"] != expected:
        raise RuntimeError(f"expected {expected} cache, got {cache['restore_status']}")
    if args.phase != "cold":
        old_cache = json.loads((root / "cold/cache.json").read_text())
        if cache["fingerprint"] != old_cache["fingerprint"]:
            raise RuntimeError("compiler/runtime/source fingerprint differs from baseline")
    result["compile_count"] = summary["compile_count"]
    result["build_count"] = summary["build_count"]
    result["passed"] = result["comparison"]["passed"]
    if result["passed"]:
        bundle = publish(cache["identity"], cache["root"], cache["local"], {
            "passed": True, "result": str(run_root / "result.json"), "native": result["native"],
            "comparison": result["comparison"]})
        result["bundle"] = str(bundle)
    if args.phase != "cold":
        result["readiness_saved_s"] = baseline["readiness_s"] - result["readiness_s"]
        result["validated_saved_s"] = baseline["first_validated_s"] - result["first_validated_s"]
        result["eligible_for_default_reuse"] = (
            result["passed"] and args.phase == "fresh" and result["build_count"] == 0
            and result["readiness_saved_s"] > 0 and result["validated_saved_s"] > 0)
    atomic_json(run_root / "result.json", result)
    print(json.dumps({k: result[k] for k in ("phase", "hostname", "passed", "readiness_s",
        "first_validated_s", "warm_tokens_per_second", "compile_count", "build_count", "comparison")}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("cold", "same", "fresh", "restored", "rebuild", "cold-same", "sequential"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=30012)
    parser.add_argument("--preflight", action="store_true", help="CPU tests only; no allocation/model load")
    parser.add_argument("--warm-cycles", type=int, default=5)
    parser.add_argument("--allow-driver-update", action="store_true",
                        help="record changes limited to this controller and its tests; model/cache/hook sources must match")
    args = parser.parse_args()
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "ENABLE_SGLANG_DSV4_A100_PATCH": "0"}
    subprocess.run([sys.executable, "-m", "pytest", "-q", *CHECKS], cwd=WORKSPACE, env=env, check=True)
    if args.preflight:
        return
    if not args.phase:
        parser.error("--phase is required unless --preflight")
    if args.warm_cycles < 5:
        parser.error("--warm-cycles must be at least five")
    for phase in phase_sequence(args.phase):
        run(argparse.Namespace(**{**vars(args), "phase": phase}))


if __name__ == "__main__":
    main()
