"""Non-mutating pre-sampler logits capture and strict durable rank validation."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

from filler.dsv4.patching import atomic_json, digest, file_digest, requested_scores

IDENTITY = ("request_id", "runtime_id", "config_hash", "cell_id", "num_tokens", "input_ids_hash", "scored_token_ids")


def first_prediction(response: dict, max_new_tokens: int = 1) -> dict:
    """Validate completion length and expose step zero, preserving raw data."""
    ids = response["output_ids"]
    rows = response["meta_info"]["output_token_ids_logprobs"]
    if not 1 <= len(ids) <= max_new_tokens or len(rows) != len(ids):
        raise ValueError("invalid completion length or missing decoding scores")
    return {**response, "output_ids": ids[:1], "meta_info": {
        **response["meta_info"], "output_token_ids_logprobs": rows[:1]}}


def make_logits_hook(config: dict):
    import torch
    root = Path(config["control_root"])
    seen = None
    decode_steps = 0

    def hook(module, args, output):
        nonlocal seen, decode_steps
        path = root / "NEXT.json"
        if not path.is_file():
            return output  # Startup/warmup, before any request is armed.
        control = json.loads(path.read_text())
        if not control.get("raw_logits"):
            return output
        if control["request_id"] == seen:
            mode = getattr(args[3], "forward_mode", None) if len(args) > 3 else None
            if (control.get("max_new_tokens", 1) > 1 and mode is not None
                    and mode.is_decode() and args[0].numel() == 1
                    and decode_steps < control["max_new_tokens"] - 1):
                decode_steps += 1
                return output
            raise RuntimeError("duplicate/stale logits request or invalid decode")
        decode_steps = 0
        if digest(args[0].detach().cpu().tolist()) != control["input_ids_hash"]:
            raise RuntimeError("logits input identity mismatch")
        logits = output.next_token_logits
        if logits is None or logits.ndim != 2 or logits.shape[0] != 1 or logits.shape[1] != module.vocab_size:
            raise RuntimeError("expected one full-vocabulary answer-prediction row")
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        # Layer hook must have completed the exact same full-prompt request first.
        ack = json.loads((root / "acks" / f"{control['request_id']}.rank{rank}.json").read_text())
        if any(ack[k] != control[k] for k in IDENTITY[:5]):
            raise RuntimeError("logits/residual request identity mismatch")
        values = logits[0].detach().float()
        ids = control["scored_token_ids"]
        record = {**{k: control[k] for k in IDENTITY}, "rank": rank,
                  "answer_position": control["num_tokens"] - 1,
                  "vocab_size": module.vocab_size, "dtype": str(logits.dtype),
                  "logits": values[ids].cpu().tolist(),
                  "log_normalizer": float(torch.logsumexp(values, dim=-1)),
                  "argmax": int(values.argmax())}
        atomic_json(Path(control["output_root"]) / f"logits.rank{rank}.json", record)
        seen = control["request_id"]
        return output  # No in-place operations or replacement of any output field.
    return hook


def collect_logits(control: dict, response: dict, tp_size: int = 4, *, wait_timeout: float = 30.0) -> dict:
    # Rank 0 may return HTTP before another rank finishes its atomic JSON save.
    # Do not arm the next request until every rank has published this request.
    paths = [Path(control["output_root"]) / f"logits.rank{rank}.json" for rank in range(tp_size)]
    deadline = time.monotonic() + wait_timeout
    while not all(path.is_file() for path in paths):
        if time.monotonic() >= deadline:
            missing = [path.name for path in paths if not path.is_file()]
            raise TimeoutError(f"missing logits rank artifacts after {wait_timeout}s: {missing}")
        time.sleep(0.01)
    refs = []
    for rank in range(tp_size):
        path = Path(control["output_root"]) / f"logits.rank{rank}.json"
        refs.append({"path": str(path), "sha256": file_digest(path)})
    enriched = {**response, "raw_logits_refs": refs}
    enriched["raw_logits"] = validate_logits(enriched, control, tp_size=tp_size)
    return enriched


def validate_logits(response: dict, control: dict, *, tp_size: int = 4, tolerance: float = 2e-5) -> dict:
    refs = response["raw_logits_refs"]
    if len(refs) != tp_size:
        raise ValueError("missing logits ranks")
    records = []
    native = requested_scores(first_prediction(response, control.get("max_new_tokens", 1)), control["scored_token_ids"])
    for rank, ref in enumerate(refs):
        path = Path(ref["path"])
        if file_digest(path) != ref["sha256"]:
            raise ValueError("logits artifact checksum mismatch")
        record = json.loads(path.read_text())
        if any(record[k] != control[k] for k in IDENTITY) or record["rank"] != rank:
            raise ValueError("stale/misaligned logits request or rank")
        if record["answer_position"] != control["num_tokens"] - 1:
            raise ValueError("wrong logits position")
        ids, values, z = record["scored_token_ids"], record["logits"], record["log_normalizer"]
        if (len(ids) != len(values) or len(ids) != len(set(ids))
                or any(t < 0 or t >= record["vocab_size"] for t in ids)
                or not all(math.isfinite(x) for x in [*values, z])):
            raise ValueError("invalid candidate logits")
        if any(abs(v - z - native[t]) > tolerance for t, v in zip(ids, values)):
            raise ValueError("raw logit/log-probability inconsistency")
        if [record["argmax"]] != response["output_ids"][:1]:
            raise ValueError("pre-sampler/native argmax mismatch")
        records.append(record)
    first = {k: v for k, v in records[0].items() if k != "rank"}
    if any({k: v for k, v in r.items() if k != "rank"} != first for r in records[1:]):
        raise ValueError("logits disagree across ranks")
    if "raw_logits" in response and response["raw_logits"] != first:
        raise ValueError("saved logits differ from rank artifacts")
    return first
