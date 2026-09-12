#!/usr/bin/env python3
"""Execute replay-correct one-fact residual transplants through SGLang."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import urllib.request
from pathlib import Path
from typing import Any


def post_json(url: str, payload: dict[str, Any], timeout: float = 300) -> dict[str, Any]:
    request = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def flush_cache(base_url: str, timeout: float = 300) -> None:
    request = urllib.request.Request(base_url.rstrip("/") + "/flush_cache", data=b"",
                                     method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"cache flush failed with HTTP {response.status}")


def arm_transplant(control_file: Path, payload: dict[str, Any]) -> None:
    """Atomically publish a control with an mtime distinct from its predecessor."""
    control_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        previous = control_file.stat().st_mtime_ns
    except FileNotFoundError:
        previous = 0
    stamp = max(time.time_ns(), previous + 1_000_000_000)
    temporary = control_file.with_suffix(f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
    os.utime(temporary, ns=(stamp, stamp))
    os.replace(temporary, control_file)


def wait_for_acks(ack_dir: Path, request_id: str, tp_size: int, timeout: float = 60) -> None:
    deadline = time.monotonic() + timeout
    expected = [ack_dir / f"{request_id}.rank{rank}.json" for rank in range(tp_size)]
    while time.monotonic() < deadline:
        if all(path.is_file() for path in expected):
            return
        time.sleep(0.05)
    missing = [path.name for path in expected if not path.is_file()]
    raise RuntimeError(f"missing transplant acknowledgements: {missing}")


def generate(
    url: str,
    input_ids: list[int],
    scored_ids: list[int],
    *,
    max_new_tokens: int = 1,
    session_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "input_ids": input_ids,
        "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens},
        "return_logprob": True, "top_logprobs_num": 20,
        "token_ids_logprob": list(dict.fromkeys(scored_ids)),
    }
    if session_params is not None:
        payload["session_params"] = session_params
    return post_json(url, payload)


def open_streaming_session(base_url: str, capacity: int = 4096) -> str:
    return str(post_json(base_url.rstrip("/") + "/open_session", {
        "capacity_of_str_len": capacity, "streaming": True,
    }))


def close_session(base_url: str, session_id: str) -> None:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/close_session",
        data=json.dumps({"session_id": session_id}).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        if response.status != 200:
            raise RuntimeError(f"session close failed with HTTP {response.status}")


def compact_response(response: dict[str, Any]) -> dict[str, Any]:
    meta = response["meta_info"]
    return {"output_ids": response["output_ids"], "text": response.get("text"),
            "cached_tokens": meta.get("cached_tokens"),
            "output_token_logprobs": meta.get("output_token_logprobs"),
            "output_top_logprobs": meta.get("output_top_logprobs"),
            "output_token_ids_logprobs": meta.get("output_token_ids_logprobs"),
            "e2e_latency": meta.get("e2e_latency")}


def requested_logprobs(response: dict[str, Any]) -> dict[int, float]:
    """Return requested-token logprobs for the first generated-token step."""
    rows = response["meta_info"]["output_token_ids_logprobs"][0]
    return {int(row[1]): float(row[0]) for row in rows}


def load_records(path: Path) -> dict[str, dict[str, Any]]:
    with path.open() as source:
        return {record.get("baseline_id", record["target_id"]): record
                for line in source if (record := json.loads(line))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:30002/generate")
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--target-id", action="append",
                        help="restrict execution to one or more target cell IDs")
    parser.add_argument("--site", action="append",
                        help="restrict patch execution to one or more sites")
    parser.add_argument("--layer", action="append", type=int,
                        help="restrict patch execution to one or more layers")
    parser.add_argument("--donor-role", action="append",
                        help="restrict patch execution to one or more donor roles")
    parser.add_argument("--clean", action="store_true",
                        help="run one fresh unpatched baseline per selected target")
    parser.add_argument("--chunked-clean", action="store_true",
                        help="run unpatched streaming baselines matched to each patch site")
    parser.add_argument("--identity", action="store_true",
                        help="run identity controls instead of substantive runs")
    parser.add_argument("--clean-output", type=Path,
                        help="fresh clean JSONL used for identity-invariance checks")
    parser.add_argument("--identity-logprob-tolerance", type=float, default=0.15)
    args = parser.parse_args()

    if sum((args.clean, args.chunked_clean, args.identity)) > 1:
        parser.error("--clean, --chunked-clean, and --identity are mutually exclusive")
    if args.clean_output is not None and not args.identity:
        parser.error("--clean-output is only valid with --identity")

    manifest = json.loads(args.manifest.read_text())
    rendered = json.loads(Path(manifest["rendered"]).read_text())
    cells = {cell["cell_id"]: cell for panel in rendered["eligible_panels"] for cell in panel["cells"]}
    panel_positions = {panel["panel_id"]: panel["positions"] for panel in rendered["eligible_panels"]}
    runs = manifest["identity_controls"] if args.identity else manifest["runs"]
    if args.target_id:
        wanted = set(args.target_id)
        runs = [run for run in runs if run["target_id"] in wanted]
    if args.site:
        wanted = set(args.site)
        runs = [run for run in runs if run["site"] in wanted]
    if args.layer:
        wanted = set(args.layer)
        runs = [run for run in runs if run["layer"] in wanted]
    if args.donor_role:
        wanted = set(args.donor_role)
        runs = [run for run in runs if run["donor_role"] in wanted]
    if args.max_runs is not None:
        runs = runs[: args.max_runs]
    control_file = args.control_root / "TRANSPLANT_NEXT.json"
    ack_dir = args.control_root / "acks"
    base_url = args.url.rsplit("/", 1)[0]
    capture_root = manifest["donor_capture_root"]

    if args.clean or args.chunked_clean:
        target_ids = args.target_id or sorted(cells)
        missing = sorted(set(target_ids) - set(cells))
        if missing:
            raise RuntimeError(f"unknown clean target IDs: {missing}")
        completed = set()
        if args.output.is_file():
            completed = set(load_records(args.output))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a") as sink:
            baseline_specs = [(cell_id, None) for cell_id in target_ids]
            if args.chunked_clean:
                baseline_specs = [(cell_id, site) for cell_id in target_ids
                                  for site in manifest["sites"]]
            for index, (cell_id, site) in enumerate(baseline_specs):
                baseline_id = cell_id if site is None else f"{cell_id}|{site}"
                if baseline_id in completed:
                    continue
                cell = cells[cell_id]
                capture_manifest = (Path(manifest["capture_manifests"]) /
                                    cell["panel_id"] / "manifest.json")
                captured_cell = next(x for x in json.loads(capture_manifest.read_text())["cells"]
                                     if x["cell_id"] == cell_id)
                answer_id = int(captured_cell["target_token_ids"]["A+X"])
                flush_cache(base_url)
                if site is None:
                    response = generate(args.url, cell["input_ids"], [answer_id])
                else:
                    positions = panel_positions[cell["panel_id"]]
                    absolute = {f"filler_{i}": value for i, value in enumerate(positions["fillers"])}
                    session_id = open_streaming_session(base_url)
                    session_rid = None
                    consumed = 0
                    try:
                        for label in manifest["sites"][site]:
                            prefix_len = absolute[label] + 1
                            stage = generate(args.url, cell["input_ids"][consumed:prefix_len],
                                             [answer_id], max_new_tokens=0,
                                             session_params={"id": session_id, "rid": session_rid})
                            session_rid = stage["meta_info"]["id"]
                            consumed = prefix_len
                        response = generate(args.url, cell["input_ids"][consumed:], [answer_id],
                                            session_params={"id": session_id, "rid": session_rid})
                    finally:
                        close_session(base_url, session_id)
                record = {"baseline_id": baseline_id, "target_id": cell_id,
                          "site": site, "target_value": int(cell["target"]),
                          "target_token_id": answer_id,
                          "response": compact_response(response),
                          "target_logprob": requested_logprobs(response)[answer_id],
                          "correct": response["output_ids"][0] == answer_id}
                sink.write(json.dumps(record, separators=(",", ":")) + "\n")
                sink.flush()
                print(f"completed clean {index + 1}/{len(baseline_specs)} {baseline_id}", flush=True)
        return

    clean = load_records(args.clean_output) if args.clean_output is not None else None

    completed = set()
    if args.output.is_file():
        with args.output.open() as source:
            completed = {json.loads(line)["run_id"] for line in source}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a") as sink:
        for index, run in enumerate(runs):
            run_id = f"{run['target_id']}|{run['donor_role']}|{run['site']}|L{run['layer']}"
            if run_id in completed:
                continue
            cell = cells[run["target_id"]]
            positions = panel_positions[cell["panel_id"]]
            absolute = {f"filler_{i}": value for i, value in enumerate(positions["fillers"])}
            target_id = int(cell["target"])
            # The manifest guarantees a one-token target; recover its saved ID
            # from the capture manifest rather than assuming token ID == value.
            capture_manifest = (Path(manifest["capture_manifests"]) /
                                cell["panel_id"] / "manifest.json")
            captured_cell = next(x for x in json.loads(capture_manifest.read_text())["cells"]
                                 if x["cell_id"] == run["target_id"])
            target_token_id = int(captured_cell["target_token_ids"]["A+X"])
            donor_cell = next(x for x in json.loads(capture_manifest.read_text())["cells"]
                              if x["cell_id"] == run["donor_id"])
            donor_token_id = int(donor_cell["target_token_ids"]["A+X"])
            scored_ids = [target_token_id, donor_token_id]
            flush_cache(base_url)
            replay = []
            previous_prefix_len = 0
            session_id = open_streaming_session(base_url)
            session_rid = None
            try:
                for stage, label in enumerate(run["positions"]):
                    request_id = f"{os.getpid()}-{index}-{stage}-{time.time_ns()}"
                    control = {"request_id": request_id, "capture_root": capture_root,
                               "pass_id": run["donor_pass_ids"][label], "layer": run["layer"]}
                    arm_transplant(control_file, control)
                    prefix_len = absolute[label] + 1
                    chunk = cell["input_ids"][previous_prefix_len:prefix_len]
                    response = generate(
                        args.url, chunk, scored_ids, max_new_tokens=0,
                        session_params={"id": session_id, "rid": session_rid},
                    )
                    wait_for_acks(ack_dir, request_id, args.tp_size)
                    session_rid = response["meta_info"]["id"]
                    replay.append({"position": label, "prefix_tokens": prefix_len,
                                   "appended_tokens": len(chunk),
                                   "response": compact_response(response)})
                    previous_prefix_len = prefix_len
                final_chunk = cell["input_ids"][previous_prefix_len:]
                final = generate(
                    args.url, final_chunk, scored_ids,
                    session_params={"id": session_id, "rid": session_rid},
                )
            finally:
                close_session(base_url, session_id)
            record = {**run, "run_id": run_id, "target_value": target_id,
                      "target_token_id": target_token_id,
                      "donor_value": int(donor_cell["target"]),
                      "donor_token_id": donor_token_id, "replay": replay,
                      "final": compact_response(final),
                      "final_answer_logprobs": {
                          "target": requested_logprobs(final)[target_token_id],
                          "donor": requested_logprobs(final)[donor_token_id],
                      },
                      "patched_correct": final["output_ids"][0] == target_token_id}
            if clean is not None:
                baseline = clean.get(f"{run['target_id']}|{run['site']}")
                if baseline is None:
                    raise RuntimeError(f"no fresh clean baseline for {run['target_id']}")
                clean_ids = baseline["response"]["output_ids"]
                patched_logprob = requested_logprobs(final)[target_token_id]
                logprob_error = abs(patched_logprob - float(baseline["target_logprob"]))
                invariant = (final["output_ids"] == clean_ids and
                             math.isfinite(logprob_error) and
                             logprob_error <= args.identity_logprob_tolerance)
                record["identity_invariance"] = {
                    "passed": invariant, "clean_output_ids": clean_ids,
                    "patched_target_logprob": patched_logprob,
                    "clean_target_logprob": baseline["target_logprob"],
                    "target_logprob_abs_error": logprob_error,
                    "max_allowed_logprob_error": args.identity_logprob_tolerance,
                }
            sink.write(json.dumps(record, separators=(",", ":")) + "\n")
            sink.flush()
            if clean is not None and not record["identity_invariance"]["passed"]:
                raise RuntimeError(f"identity-patch invariance failed: {run_id} "
                                   f"{record['identity_invariance']}")
            print(f"completed {index + 1}/{len(runs)} {run_id}", flush=True)


if __name__ == "__main__":
    main()
