#!/usr/bin/env python3
"""Record clean one-token correctness for rendered discovery panels."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer


def post(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rendered", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:30002/generate")
    args = parser.parse_args()
    rendered = json.loads(args.rendered.read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    results = []
    for panel in rendered["eligible_panels"]:
        cells = []
        for cell in panel["cells"]:
            target_ids = tokenizer.encode(str(cell["target"]), add_special_tokens=False)
            if len(target_ids) != 1:
                raise RuntimeError(f"renderer invariant failed for {cell['cell_id']}")
            response = post(args.url, {
                "input_ids": cell["input_ids"],
                "sampling_params": {"temperature": 0, "max_new_tokens": 1},
            })
            generated = response["output_ids"][0]
            cells.append({
                "cell_id": cell["cell_id"], "target": cell["target"],
                "generated_token_id": generated,
                "target_token_id": target_ids[0], "correct": generated == target_ids[0],
            })
        results.append({"panel_id": panel["panel_id"], "cells": cells})
        print(f"screened {panel['panel_id']} correct={sum(c['correct'] for c in cells)}/4", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"rendered": str(args.rendered), "panels": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
