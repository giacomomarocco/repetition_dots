#!/usr/bin/env python3
"""Prepare, preflight, run and resume a one-fact pilot + full-split campaign."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import uuid

from filler.dsv4.patching import (
    DESIGNS, Journal, atomic_bytes, atomic_json, build_manifest, campaign_counts,
    digest, file_digest, layer42_diagnostics,
)

WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = (WORKSPACE / "runs/deepseek-v4-flash/one-fact-patching-discovery").resolve()
COVERAGE_ROOT = (WORKSPACE / "runs/deepseek-v4-flash/one-fact-patching-filler-coverage").resolve()
FILLER10_ROOT = (WORKSPACE / "runs/deepseek-v4-flash/one-fact-patching-filler10-coverage").resolve()
REPEAT_ROOT = (WORKSPACE / "runs/deepseek-v4-flash/one-fact-patching-filler-repeat").resolve()
RANDOM_ROOT = (WORKSPACE / "runs/deepseek-v4-flash/one-fact-patching-filler-random").resolve()
REDUNDANCY_ROOT = (WORKSPACE / "runs/deepseek-v4-flash/one-fact-patching-filler-redundancy").resolve()
CONFIRMATION_ROOT = (WORKSPACE / "runs/deepseek-v4-flash/one-fact-patching-filler10-confirmation").resolve()
CONFIRMATION_RENDERED = WORKSPACE / "runs/deepseek-v4-flash/factorial/filler-confirmation-rendered-k20.json"
FACTORIAL_DESIGN = WORKSPACE / "runs/deepseek-v4-flash/factorial/filler-design-expanded.json"
RENDERED = WORKSPACE / "runs/deepseek-v4-flash/factorial/filler-discovery-rendered-k20.json"
CHECKPOINT = WORKSPACE / "model/DeepSeek-V4-Flash-0731-MoE-MXFP4-BF16"
SOURCES = (
    "filler/dsv4/factorial.py", "filler/dsv4/lens.py", "filler/dsv4/campaign_hook.py",
    "filler/dsv4/patching.py", "filler/dsv4/patching_campaign.py", "filler/dsv4/patching_analysis.py",
    "filler/dsv4/repeat_analysis.py", "filler/dsv4/repeat_extension.py",
    "filler/dsv4/filler_random.py", "filler/dsv4/random_analysis.py", "filler/dsv4/lens_positions.py",
    "filler/addition/k5_extension.py", "filler/addition/one_fact.py", "filler/addition/accuracy_plot.py",
    "filler/dsv4/patching_comparison.py", "filler/dsv4/patching_logits.py", "filler/dsv4/redundancy_analysis.py",
    "scripts/dsv4/one_fact_patching.py", "scripts/dsv4/run_activation_patching.py",
    "run_deepseek_v4_a100.sh", "run_one_fact_patching_allocation.sh",
    "ports/sglang/python/sglang/srt/server_args.py",
    "ports/sglang/python/sglang/srt/model_executor/hook_manager.py",
    "ports/sglang/python/sglang/srt/models/deepseek_v4.py",
    "ports/sglang/python/sglang/srt/layers/sampler.py",
    "ports/sglang/python/sglang/srt/layers/logits_processor.py",
    "ports/sglang/python/sglang/srt/utils/network.py",
    "ports/sglang/python/sglang/srt/distributed/device_communicators/shm_broadcast.py",
    "ports/deepseek-v4-a100-sglang/dsv4_a100_patch/patch.py",
)
CHECKS = (
    "tests/test_dsv4_factorial.py", "tests/test_deepseek_v4_logit_lens.py", "tests/test_dsv4_transplant_hook.py",
    "tests/test_activation_patching_runner.py", "tests/test_one_fact_patching.py",
    "tests/test_filler_repeat.py", "tests/test_repeat_extension.py", "tests/test_one_fact_addition_sglang.py", "tests/test_two_fact_addition_sglang.py",
    "tests/test_filler_random.py",
    "tests/test_filler_redundancy.py", "tests/test_patching_comparison.py", "tests/test_patching_confirmation.py",
)


def launch_command(control_root: Path, port: int = 30002, *, raw_logits: bool = False) -> list[str]:
    from filler.dsv4.campaign_hook import campaign_hook_spec
    return ["bash", str(WORKSPACE / "run_deepseek_v4_a100.sh"),
            "--port", str(port),
            "--disable-radix-cache", "--chunked-prefill-size", "-1",
            "--disable-cuda-graph", "--disable-piecewise-cuda-graph", "--disable-overlap-schedule",
            "--max-running-requests", "1", "--enable-return-hidden-states",
            "--forward-hooks", json.dumps(campaign_hook_spec(control_root, raw_logits=raw_logits), separators=(",", ":"))]


def server_environment() -> dict[str, str]:
    env = {**os.environ, "DEEPSEEK_STARTUP_PROFILE": "fast",
           "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "4",
           "TOKENIZERS_PARALLELISM": "false"}
    # SGLang's shared-memory broadcaster also interprets SGLANG_PORT. Exporting
    # the HTTP port lets it claim that socket during initialization, long before
    # uvicorn binds. Keep the HTTP port exclusively in the command-line args.
    env.pop("SGLANG_PORT", None)
    return env


def verify_manifest(manifest: dict) -> None:
    expected = {k: v for k, v in manifest.items() if k != "config_hash"}
    if digest(expected) != manifest["config_hash"]:
        raise ValueError("manifest configuration hash mismatch")
    for path, checksum in manifest["source_hashes"].items():
        if file_digest(WORKSPACE / path) != checksum:
            raise ValueError(f"source changed after preflight: {path}")
    if "five_shot_k5" in manifest:
        from filler.addition.k5_extension import verify
        verify(manifest["five_shot_k5"])
        sources = [c["source"] for c in json.loads((WORKSPACE / "notebooks/addition_accuracy.ipynb").read_text())["cells"]]
        if sources != manifest["notebook_sources"]:
            raise ValueError("notebook source changed after preflight")
    if manifest["checkpoint_files"] != checkpoint_files():
        raise ValueError("checkpoint files changed after preflight")


def checkpoint_files() -> dict:
    return {path.name: {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for path in sorted(CHECKPOINT.glob("*.safetensors"))}


def verify_reused_inputs(manifest: dict, previous: dict, complete: dict) -> None:
    """Compare scientific inputs without requiring historical code to be current."""
    if digest({k: v for k, v in previous.items() if k != "config_hash"}) != previous["config_hash"]:
        raise ValueError("previous manifest configuration hash mismatch")
    if (complete["config_hash"] != previous["config_hash"] or complete["status"] != "complete"
            or not complete["integrity"]["passed"]):
        raise ValueError("requires the completed original campaign")
    if manifest["panels"] != previous["panels"] or manifest["checkpoint_files"] != previous["checkpoint_files"]:
        raise ValueError("saved panels/prompts or checkpoint differ from the original campaign")
    for path in (RENDERED, CHECKPOINT / "config.json", CHECKPOINT / "tokenizer.json", CHECKPOINT / "tokenizer_config.json"):
        key = str(path.relative_to(WORKSPACE))
        if manifest["source_hashes"][key] != previous["source_hashes"][key]:
            raise ValueError(f"reused input changed: {key}")


def validate_confirmation_inputs(rendered: dict, factorial: dict, discovery: dict) -> dict:
    """Require exact full-set membership and fact disjointness before any inference."""
    expected = [c for c in factorial["cells"] if c["split"] == "confirmation" and c["kind"] == "one_fact"]
    actual = [c for p in rendered["eligible_panels"] for c in p["cells"]]
    if (len(expected) != 240 or len(actual) != 240
            or len({c["cell_id"] for c in actual}) != 240):
        raise ValueError("confirmation requires all 240 distinct canonical cells")
    by_id = {c["cell_id"]: c for c in actual}
    if set(by_id) != {c["cell_id"] for c in expected}:
        raise ValueError("confirmation cell membership differs from the factorial design")
    for c in expected:
        if any(by_id[c["cell_id"]].get(k) != v for k, v in c.items()):
            raise ValueError("confirmation cell metadata differs from the factorial design")
    expected_panels = {c["panel_id"] for c in expected}
    if len(expected_panels) != 60 or {p["panel_id"] for p in rendered["eligible_panels"]} != expected_panels:
        raise ValueError("confirmation panel membership differs from the factorial design")
    for panel in rendered["eligible_panels"]:
        if any(c["panel_id"] != panel["panel_id"] for c in panel["cells"]):
            raise ValueError("confirmation cell attached to the wrong panel")
    confirmation_facts = {c["left_id"] for c in actual}
    discovery_facts = {c["left_id"] for p in discovery["eligible_panels"] for c in p["cells"]}
    if len(confirmation_facts) != 120 or confirmation_facts & discovery_facts:
        raise ValueError("requires 120 confirmation facts disjoint from discovery")
    if rendered.get("rejected_panel_count", 0) or rendered.get("rejected_panels"):
        raise ValueError("confirmation panels must not be excluded")
    return {"passed": True, "panels": 60, "targets": 240, "distinct_facts": 120,
            "discovery_fact_overlap": 0, "canonical_cell_metadata_exact": True,
            "clean_correctness_filter": False}


def prepare(root: Path, design: str = "original", *, draws: int | None = None, seed: int = 42, repeat_sources=None, five_shot_k5=False) -> None:
    from transformers import AutoTokenizer
    root.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True, trust_remote_code=True)
    rendered_path = CONFIRMATION_RENDERED if design == "filler10-confirmation" else RENDERED
    rendered = json.loads(rendered_path.read_text())
    dataset_validation = None
    if design == "filler10-confirmation":
        dataset_validation = validate_confirmation_inputs(rendered, json.loads(FACTORIAL_DESIGN.read_text()),
                                                         json.loads(RENDERED.read_text()))
    manifest = build_manifest(rendered, tokenizer, design=design, draws=draws, seed=seed, repeat_sources=repeat_sources)
    if five_shot_k5:
        if design != "filler-repeat":
            raise ValueError("five-shot-k5 requires filler-repeat")
        from filler.addition.k5_extension import prepare as prepare_k5
        manifest["five_shot_k5"] = prepare_k5(WORKSPACE, tokenizer)
    manifest["rendered"] = str(rendered_path)
    if dataset_validation is not None:
        manifest["dataset_validation"] = dataset_validation
    manifest["checkpoint_files"] = checkpoint_files()
    if len(manifest["checkpoint_files"]) != 48:
        raise ValueError("expected all 48 local checkpoint shards")
    manifest["source_hashes"] = {path: file_digest(WORKSPACE / path) for path in (*SOURCES, *CHECKS)}
    input_paths = [rendered_path, CHECKPOINT / "config.json", CHECKPOINT / "tokenizer.json", CHECKPOINT / "tokenizer_config.json"]
    if design == "filler10-confirmation":
        input_paths.extend([FACTORIAL_DESIGN, RENDERED])
    for path in input_paths:
        manifest["source_hashes"][str(path.relative_to(WORKSPACE)) if path.is_relative_to(WORKSPACE) else str(path)] = file_digest(path)
    if design in {"filler-coverage", "filler-redundancy", "filler-repeat", "filler-random"}:
        previous_path, complete_path = DEFAULT_ROOT / "manifest.json", DEFAULT_ROOT / "COMPLETE.json"
        previous = json.loads(previous_path.read_text())
        verify_reused_inputs(manifest, previous, json.loads(complete_path.read_text()))
        manifest["reused_campaign"] = {"manifest": str(previous_path), "config_hash": previous["config_hash"]}
        for path in (previous_path, complete_path):
            manifest["source_hashes"][str(path.relative_to(WORKSPACE)) if path.is_relative_to(WORKSPACE) else str(path)] = file_digest(path)
    if design == "filler10-coverage":
        from filler.dsv4.patching_comparison import historical_preflight
        compatibility = historical_preflight(manifest, [DEFAULT_ROOT, COVERAGE_ROOT])
        atomic_json(root / "compatibility.json", compatibility)
        manifest["reused_campaigns"] = [
            {"root": c["root"], "config_hash": c["config_hash"]} for c in compatibility["campaigns"]]
        for campaign in compatibility["campaigns"]:
            for path, checksum in campaign["input_sha256"].items():
                manifest["source_hashes"][str(Path(path).relative_to(WORKSPACE))] = checksum
    if design == "filler-repeat" and manifest.get("repeat_sources") == [1, 2, 3, 4]:
        from filler.dsv4.repeat_extension import freeze_history
        manifest["historical_repeat"] = freeze_history(WORKSPACE, manifest)
        manifest["source_hashes"].update(manifest["historical_repeat"]["input_hashes"])
    if five_shot_k5:
        manifest["source_hashes"].update(manifest["five_shot_k5"]["input_hashes"])
        manifest["notebook_sources"] = [c["source"] for c in json.loads((WORKSPACE / "notebooks/addition_accuracy.ipynb").read_text())["cells"]]
    if design == "filler-random":
        from filler.dsv4.random_analysis import freeze_history
        manifest["historical_repeats"] = freeze_history(WORKSPACE, manifest)
        for history in manifest["historical_repeats"]:
            manifest["source_hashes"].update(history["hashes"])
    manifest["revisions"] = {name: subprocess.check_output(["git", "-C", str(WORKSPACE / path), "rev-parse", "HEAD"], text=True).strip()
                             for name, path in (("workspace", "."), ("sglang", "ports/sglang"), ("a100_port", "ports/deepseek-v4-a100-sglang"))}
    manifest["config_hash"] = digest({k: v for k, v in manifest.items() if k != "config_hash"})
    path = root / "manifest.json"
    if path.exists() and (root / "results.jsonl").exists() and json.loads(path.read_text())["config_hash"] != manifest["config_hash"]:
        raise ValueError("cannot change a campaign configuration after results exist; choose a new output root")
    atomic_json(path, manifest)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "TOKENIZERS_PARALLELISM": "false", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    test = subprocess.run([sys.executable, "-m", "pytest", "-q", *CHECKS], cwd=WORKSPACE,
                          env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    atomic_bytes(root / "preflight-tests.log", test.stdout.encode())
    print(test.stdout, flush=True)
    if test.returncode:
        raise RuntimeError("preflight tests failed")
    subprocess.run(["bash", "-n", str(WORKSPACE / "run_one_fact_patching_allocation.sh"),
                    str(WORKSPACE / "run_deepseek_v4_a100.sh")], check=True)
    verify_manifest(manifest)
    if design == "filler-random":
        for name, checksum in manifest["source_hashes"].items():
            source = WORKSPACE / name
            relative = source.relative_to(WORKSPACE)
            destination = root / "source_snapshot" / relative
            atomic_bytes(destination, source.read_bytes())
            if file_digest(destination) != checksum:
                raise ValueError("source snapshot checksum mismatch")
    report = {"passed": True, "config_hash": manifest["config_hash"], "counts": manifest["counts"],
              "design": design,
              "diagnostics_per_runtime": len(layer42_diagnostics(manifest, manifest["pilot_panel"])),
              "fresh_runtime_forward_passes": sum(manifest["counts"][k] for k in ("trials", "identity", "baselines"))
                  + len(layer42_diagnostics(manifest, manifest["pilot_panel"])),
              "pilot_counts": manifest["pilot_counts"], "tests": list(CHECKS),
              "prepared_at": datetime.now(timezone.utc).isoformat(),
              "server_command_template": launch_command(root / "runtimes/RUNTIME_ID/control", raw_logits=manifest.get("raw_logits", False)),
              "allocation_command": ["salloc", "--account=m5258_g", "--qos=interactive", "--nodes=1", "--gpus=4",
                                     "--constraint=gpu&hbm80g", "--time=04:00:00"],
              "execution": f"srun --ntasks=1 --cpus-per-task=128 --cpu-bind=none --gpus=4 bash {WORKSPACE / 'run_one_fact_patching_allocation.sh'} {root}",
              "gpu_validation": "pending approved allocation; required before substantive trials"}
    report["ordinary_generation_requests"] = 262 if five_shot_k5 else 0
    report["storage"] = storage_estimate(manifest, root)
    if report["storage"]["filesystem_available_bytes"] < report["storage"]["recommended_free_bytes"]:
        raise RuntimeError("insufficient free space for prepared capture estimate")
    atomic_json(root / "preflight.json", report)
    print(json.dumps(report, indent=2), flush=True)


def storage_estimate(manifest: dict, root: Path) -> dict:
    from filler.dsv4.patching import baseline_modes
    config = json.loads((CHECKPOINT / "config.json").read_text())
    row_bytes = 2 * config["hc_mult"] * config["hidden_size"]  # BF16 complete residual
    layers, ranks = config["num_hidden_layers"], manifest["runtime_requirements"]["tp_size"]
    baseline_rows = sum(len(c["input_ids"]) if mode == "full_downstream" else 1
                        for p in manifest["panels"] for c in p["cells"] for mode in baseline_modes(manifest))
    specs = manifest["trials"] + manifest["identity_controls"] + layer42_diagnostics(manifest, manifest["pilot_panel"])
    selected_rows = sum(len(set([*s["positions"], *s.get("source_positions", []), next(p["answer_position"] for p in manifest["panels"] if p["panel_id"] == s["panel_id"])])) for s in specs)
    if manifest.get("design") == "filler-random":
        selected_rows = sum(len(manifest["position_coverage"][s["target_id"]]) for s in specs)
    payload = (baseline_rows + selected_rows) * layers * ranks * row_bytes
    bank_bytes = (manifest["counts"]["targets"] * 19 * layers * row_bytes
                  if manifest.get("design") == "filler-random" else 0)
    passes = len(specs) + manifest["counts"]["baselines"]
    # Tensor serialization, raw responses, rank logits/acks, controls and journal.
    overhead = passes * 512 * 1024
    estimate = payload + overhead + bank_bytes
    stat = os.statvfs(root)
    return {"residual_payload_bytes": payload, "metadata_allowance_bytes": overhead,
            "replacement_bank_bytes": bank_bytes,
            "estimated_total_bytes": estimate, "recommended_free_bytes": int(estimate * 1.25),
            "filesystem_available_bytes": stat.f_bavail * stat.f_frsize,
            "scope": "one uninterrupted runtime; restarts add fresh captures, baselines and controls",
            "filesystem_free_is_not_user_quota": True, "assumed_dtype": "BF16"}


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.load(response)


def parse_allocation_deadline(raw: str) -> float:
    """Read UTC scontrol output, including NERSC's public-to-internal QOS alias."""
    fields = dict(part.split("=", 1) for part in shlex.split(raw) if "=" in part)
    if fields.get("QOS") not in {"interactive", "gpu_interactive"} or fields.get("NumNodes") != "1":
        raise RuntimeError(f"campaign requires one interactive GPU node; received QOS={fields.get('QOS')}, NumNodes={fields.get('NumNodes')}")
    if fields.get("TimeLimit") != "04:00:00":
        raise RuntimeError("campaign requires the approved four-hour allocation")
    return datetime.fromisoformat(fields["EndTime"]).replace(tzinfo=timezone.utc).timestamp()


def allocation_deadline() -> float:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError("run requires an explicitly approved Slurm GPU allocation")
    # Slurm normally formats Pacific wall time. Force UTC at the source so the
    # controller's TZ cannot silently shift the allocation deadline by hours.
    raw = subprocess.check_output(["scontrol", "show", "job", job_id, "-o"],
                                  text=True, env={**os.environ, "TZ": "UTC"})
    return parse_allocation_deadline(raw)


def run(root: Path, port: int, *, campaign_class=None, finish_callback=None) -> None:
    from filler.dsv4.patching_campaign import Campaign, HTTPTransport, WalltimeReached
    Campaign = Campaign if campaign_class is None else campaign_class
    complete_campaign = finish if finish_callback is None else finish_callback
    manifest = json.loads((root / "manifest.json").read_text())
    verify_manifest(manifest)
    preflight = json.loads((root / "preflight.json").read_text())
    if not preflight["passed"] or preflight["config_hash"] != manifest["config_hash"]:
        raise RuntimeError("matching successful preflight required before model load")
    # Only one controller can ever arm this campaign's hook or append its journal.
    with (root / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        journal = Journal(root, manifest["config_hash"])
        expected = {s["record_id"] for key in ("trials", "identity_controls") for s in manifest[key]}
        k5 = manifest.get("five_shot_k5")
        k5_complete = k5 is None or (Path(k5["output"]) / "COMPLETE.json").is_file()
        if expected <= journal.records.keys() and k5_complete:
            complete_campaign(manifest, journal, root)
            return
        deadline = allocation_deadline()
        raw_allocation = subprocess.check_output(["scontrol", "show", "job", os.environ["SLURM_JOB_ID"], "-o"],
                                                text=True, env={**os.environ, "TZ": "UTC"})
        allocation = dict(part.split("=", 1) for part in shlex.split(raw_allocation) if "=" in part)
        if allocation.get("Account") != "m5258_g":
            raise RuntimeError("campaign requires the approved GPU account m5258_g")
        import torch
        if torch.cuda.device_count() != 4:
            raise RuntimeError("exactly four visible GPUs required")
        devices = [torch.cuda.get_device_properties(i) for i in range(4)]
        if any("A100" not in d.name or d.total_memory < 79 * 1024**3 for d in devices):
            raise RuntimeError("four A100 80 GB GPUs required")
        # Refuse to attach to an unrelated warmed server.
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError as exc:
                raise RuntimeError(f"port {port} already occupied; controller must own its server") from exc
        runtime_id = f"{os.environ['SLURM_JOB_ID']}-{uuid.uuid4().hex[:12]}"
        runtime_root = root / "runtimes" / runtime_id
        runtime_root.mkdir(parents=True)
        command = launch_command(runtime_root / "control", port, raw_logits=manifest.get("raw_logits", False))
        env = server_environment()
        runtime = {"runtime_id": runtime_id, "job_id": os.environ["SLURM_JOB_ID"],
                   "hostname": socket.gethostname(), "deadline": deadline, "command": command,
                   "created_at": datetime.now(timezone.utc).isoformat(),
                   "allocation": {key: allocation.get(key) for key in (
                       "Account", "QOS", "NodeList", "StartTime", "EndTime", "TimeLimit", "NumNodes", "NumCPUs", "TresPerNode", "Features")},
                   "config_hash": manifest["config_hash"], "source_hashes": manifest["source_hashes"],
                   "gpus": [{"name": d.name, "memory_bytes": d.total_memory} for d in devices]}
        atomic_json(runtime_root / "runtime.json", runtime)
        campaign = Campaign(manifest, root, runtime_id, HTTPTransport(f"http://127.0.0.1:{port}", runtime_root / "control"),
                            CHECKPOINT, deadline=deadline)
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
            signal.signal(signum, campaign.request_stop)
        with (runtime_root / "server.log").open("w") as log:
            server = subprocess.Popen(command, cwd=WORKSPACE, env=env, stdout=log,
                                      stderr=subprocess.STDOUT, start_new_session=True)
            try:
                startup_deadline = min(time.time() + 1800, deadline - 180)
                while True:
                    if server.poll() is not None:
                        raise RuntimeError(f"server exited {server.returncode}; see {runtime_root / 'server.log'}")
                    if campaign.stop_requested or time.time() > startup_deadline:
                        raise WalltimeReached("stopped during server startup")
                    try:
                        info = get_json(f"http://127.0.0.1:{port}/server_info")
                        break
                    except (OSError, ValueError):
                        print(f"Waiting for model server; {deadline - time.time():.0f}s allocation remaining", flush=True)
                        time.sleep(15)
                requirements = {**manifest["runtime_requirements"], "disable_piecewise_cuda_graph": True,
                                "disable_overlap_schedule": True}
                for key, value in requirements.items():
                    if key != "fused_mhc_post_pre" and info.get(key) != value:
                        raise RuntimeError(f"actual server configuration differs: {key}={info.get(key)!r}")
                from filler.dsv4.campaign_hook import campaign_hook_spec
                if info.get("forward_hooks") != campaign_hook_spec(runtime_root / "control", raw_logits=manifest.get("raw_logits", False)):
                    raise RuntimeError("actual server hooks differ from campaign")
                if Path(info["model_path"]).resolve() != CHECKPOINT.resolve():
                    raise RuntimeError("actual model path differs from frozen checkpoint")
                # Whitelist runtime metadata; never persist arbitrary server settings.
                runtime["server"] = {key: info.get(key) for key in (
                    *requirements, "version", "model_path", "dtype", "kv_cache_dtype", "attention_backend",
                    "quantization", "mem_fraction_static", "forward_hooks")}
                atomic_json(runtime_root / "runtime.json", runtime)
                if k5 is not None:
                    from filler.addition.k5_extension import run as run_k5
                    run_k5(k5, f"http://127.0.0.1:{port}/generate", runtime,
                           control_root=runtime_root / "control", deadline=deadline,
                           should_stop=lambda: campaign.stop_requested)
                    verify_manifest(manifest)
                integrity = campaign.run()
                verify_manifest(manifest)
                complete_campaign(manifest, campaign.journal, root, integrity=integrity)
            except WalltimeReached as exc:
                campaign.journal.progress(status="checkpointed", reason=str(exc), runtime_id=runtime_id)
                print(f"{exc}. Resume with the same campaign root in a new approved allocation.", flush=True)
            except BaseException as exc:
                atomic_json(runtime_root / "controller_failure.json", {"error": f"{type(exc).__name__}: {exc}"})
                raise
            finally:
                # Own only this server process group; no Slurm cancellation/modification.
                if server.poll() is None:
                    os.killpg(server.pid, signal.SIGTERM)
                    try:
                        server.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(server.pid, signal.SIGKILL)
                        server.wait(timeout=10)


def finish(manifest: dict, journal: Journal, root: Path, *, integrity: dict | None = None):
    from filler.dsv4.patching_analysis import summarize
    report = summarize(manifest, journal, root / "analysis", integrity=integrity)
    comparison = None
    if manifest.get("design") == "filler10-coverage":
        from filler.dsv4.patching_comparison import compare
        comparison = compare(root, integrity=report["integrity"])
    marker = {"status": "complete", "config_hash": manifest["config_hash"],
              "finished_at": datetime.now(timezone.utc).isoformat(), "integrity": report["integrity"]}
    if manifest.get("design") == "filler-random":
        lab = root / "LAB_LOG.md"
        entry = (f"\n### {datetime.now(timezone.utc).isoformat()} — completed\n\n"
                 f"Configuration `{manifest['config_hash']}`; integrity and independent export reconstruction passed. "
                 "All 96 prompts retained. See [results](analysis/REPORT.md), [comparison](comparison/REPORT.md), "
                 "and [manifest](manifest.json). Uncertainty is conditional on one noise realization; "
                 "cutoff comparisons also change replacement count.\n")
        atomic_bytes(lab, ((lab.read_text() if lab.exists() else "# Random filler lab log\n") + entry).encode())
        atomic_json(root / "COMPLETE.json", marker)
        journal.progress(status="complete", integrity=report["integrity"])
        return
    if comparison is not None:
        marker["comparison"] = {"passed": comparison["passed"], "validation": str(root / "comparison/validation.json")}
    if manifest.get("design") == "filler-repeat":
        if "historical_repeat" in manifest:
            from filler.dsv4.repeat_extension import combined_report
            combined_report(manifest, journal, root, report["integrity"])
        if "five_shot_k5" in manifest:
            from filler.addition.k5_extension import merge
            from filler.dsv4.repeat_extension import refresh_notebook
            merge(manifest["five_shot_k5"])
            validation = refresh_notebook(WORKSPACE)
            if validation["embedded_pngs"] != 2:
                raise RuntimeError("notebook must contain both accuracy figures")
            atomic_json(root / "notebook-validation.json", validation)
        lab = root / "LAB_LOG.md"
        entry = (f"\n### {datetime.now(timezone.utc).isoformat()} — campaign completed\n\n"
                 f"Configuration `{manifest["config_hash"]}`. All 96 targets retained; native and rank integrity gates passed. "
                 "Results use 2,000 shared whole-panel bootstrap resamples, seed 42. "
                 "Source comparisons combine source-position and replacement-count effects. "
                 "See [report](analysis/REPORT.md) and [manifest](manifest.json).\n")
        atomic_bytes(lab, ((lab.read_text() if lab.exists() else "# Filler-repeat lab log\n") + entry).encode())
        atomic_json(root / "COMPLETE.json", marker)
        journal.progress(status="complete", integrity=report["integrity"])
        return
    lab = WORKSPACE / "DEEPSEEK_V4_LOGIT_LENS.md"
    tag = f"Campaign `{manifest['config_hash']}`"
    existing = lab.read_text()
    if tag not in existing:
        counts = campaign_counts(manifest)
        diagnostics = len(layer42_diagnostics(manifest, manifest["pilot_panel"]))
        entry = (f"\n### {datetime.now(timezone.utc).date()}: one-fact simultaneous patching {manifest.get('split', 'discovery')} completed\n\n"
                 f"{tag}, design `{manifest.get('design', 'original')}`: all {counts['trials']:,} substantive trials, "
                 f"{counts['identity']:,} identity controls and {counts['baselines']} target/mode baseline cells completed. "
                 "Pilot results were retained in full-split totals. Each runtime passed native final-layer equivalence; "
                 "each panel passed identity controls before substantive trials. The first active panel per runtime "
                 f"also passed all {diagnostics} layer-42 filler-only controls. Every stored candidate delta was recomputed from "
                 "its original raw response and matched runtime/mode baseline.\n\n"
                 f"Manifest, journal and raw rank captures: `{root}`. "
                 f"[Results and plots]({root / 'analysis/REPORT.md'}) include all three sums and paired panel "
                 "bootstrap intervals (20,000 resamples, seed 42); no clean-correctness filtering. "
                 "Interrupted runtimes retain their original baselines; unfinished work uses fresh captures. "
                 "Reported intervals are descriptive and are not adjusted for multiple comparisons.\n")
        if comparison is not None:
            entry += (f"\n[Three-range filler_10/filler_5 comparison]({root / 'comparison/REPORT.md'}) "
                      "includes clean/patched candidate scores, target-relative and donor-minus-mixed logit gaps, "
                      "and matched mode/layer/donor/site contrasts. Uses 20,000 shared panel resamples, seed 42, "
                      "±1σ standard errors (ddof=1) and descriptive 95% intervals. Historical tables reproduce "
                      "to 1e-12. Errors do not separately estimate between-runtime variability.\n")
        atomic_bytes(lab, (existing + entry).encode())
        handoff = WORKSPACE / "ONE_FACT_PATCHING_HANDOFF.md"
        atomic_bytes(handoff, (handoff.read_text() + entry).encode())
    atomic_json(root / "COMPLETE.json", marker)
    journal.progress(status="complete", integrity=report["integrity"])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "analyze", "compare"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--design", choices=DESIGNS, default=None,
                        help="preparation design; defaults to original (run/analyze use the frozen manifest)")
    parser.add_argument("--draws", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--port", type=int, default=30002)
    parser.add_argument("--repeat-sources", type=int, nargs="+", default=None,
                        help="filler-repeat sources; default 5 0, each copied into every later filler")
    parser.add_argument("--five-shot-k5", action="store_true",
                        help="prepare the 262-example k=5 supplement before repeat requests")
    args = parser.parse_args(argv)
    if args.design is not None and args.action != "prepare":
        parser.error("--design is a preparation option; run/analyze use --root and its frozen manifest")
    args.design = args.design or "original"
    args.draws = (1 if args.design == "filler-random" else 5) if args.draws is None else args.draws
    if args.design == "filler-random" and (args.draws != 1 or args.seed != 42):
        parser.error("filler-random requires --draws 1 --seed 42")
    if args.repeat_sources is not None or args.five_shot_k5:
        if args.action != "prepare" or args.design != "filler-repeat":
            parser.error("--repeat-sources and --five-shot-k5 require prepare --design filler-repeat")
        if args.root is None:
            parser.error("extensions require an explicit new --root to preserve the completed campaign")
        from filler.dsv4.patching import design_interventions
        try:
            design_interventions(args.design, repeat_sources=args.repeat_sources)
        except ValueError as exc:
            parser.error(str(exc))
    if args.action == "compare" and args.root is None:
        parser.error("compare requires --root for the new filler10-coverage campaign")
    roots = {"original": DEFAULT_ROOT, "filler-coverage": COVERAGE_ROOT, "filler10-coverage": FILLER10_ROOT, "filler10-confirmation": CONFIRMATION_ROOT, "filler-redundancy": REDUNDANCY_ROOT, "filler-repeat": REPEAT_ROOT, "filler-random": RANDOM_ROOT}
    args.root = (args.root or roots[args.design]).resolve()
    return args


def main():
    args = parse_args()
    root = args.root
    if args.action == "prepare":
        prepare(root, design=args.design, draws=args.draws, seed=args.seed,
                repeat_sources=args.repeat_sources, five_shot_k5=args.five_shot_k5)
    elif args.action == "run":
        run(root, args.port)
    elif args.action == "compare":
        from filler.dsv4.patching_comparison import compare
        report = compare(root)
        print(json.dumps({"passed": report["passed"], "output": str(root / "comparison"),
                          "historical_reproduction": report["historical_reproduction"]}, indent=2))
    else:
        manifest = json.loads((root / "manifest.json").read_text())
        finish(manifest, Journal(root, manifest["config_hash"]), root)


if __name__ == "__main__":
    main()
