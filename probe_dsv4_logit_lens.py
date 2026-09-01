#!/usr/bin/env python3
"""Send the deterministic first request used for DSV4 lens validation."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path


def capture_names(capture_root: Path, rank: int = 0) -> set[str]:
    return {path.name for path in (capture_root / f"rank{rank}").glob("pass*.pt")}


def arm_capture(trigger: Path) -> int:
    """Arm one capture with an mtime distinct from any previous trigger."""
    trigger.parent.mkdir(parents=True, exist_ok=True)
    try:
        previous = trigger.stat().st_mtime_ns
    except FileNotFoundError:
        previous = 0
    stamp = max(time.time_ns(), previous + 1_000_000_000)
    trigger.write_text(f"validation {stamp}\n")
    os.utime(trigger, ns=(stamp, stamp))
    return stamp


def discover_capture(capture_root: Path, before: set[str], tp_size: int = 4) -> int:
    created = capture_names(capture_root) - before
    if len(created) != 1:
        raise RuntimeError(f"validation request created captures={sorted(created)}")
    name = created.pop()
    if not (name.startswith("pass") and name.endswith(".pt")):
        raise RuntimeError(f"unexpected capture filename: {name}")
    missing = [
        rank for rank in range(tp_size)
        if not (capture_root / f"rank{rank}" / name).is_file()
    ]
    if missing:
        raise RuntimeError(f"incomplete TP capture {name}; missing ranks={missing}")
    return int(name.removeprefix("pass").removesuffix(".pt"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:30002/generate")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--capture-root", type=Path)
    parser.add_argument("--pass-id-output", type=Path)
    args = parser.parse_args()
    if (args.capture_root is None) != (args.pass_id_output is None):
        parser.error("--capture-root and --pass-id-output must be provided together")
    before = None
    if args.capture_root is not None:
        before = capture_names(args.capture_root)
        arm_capture(args.capture_root / "CAPTURE_NEXT")
    payload = {
        "text": args.prompt,
        "sampling_params": {"temperature": 0, "max_new_tokens": 1},
        "return_logprob": True,
        "top_logprobs_num": 10,
        "return_hidden_states": True,
    }
    request = urllib.request.Request(
        args.url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        result = json.load(response)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    pass_id = None
    if args.capture_root is not None:
        pass_id = discover_capture(args.capture_root, before)
        args.pass_id_output.parent.mkdir(parents=True, exist_ok=True)
        args.pass_id_output.write_text(json.dumps({"pass_id": pass_id}) + "\n")
    print(
        json.dumps(
            {
                "output_ids": result["output_ids"],
                "text": result["text"],
                "e2e_latency": result["meta_info"].get("e2e_latency"),
                "hidden_state_steps": len(result["meta_info"]["hidden_states"]),
                "capture_pass_id": pass_id,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
