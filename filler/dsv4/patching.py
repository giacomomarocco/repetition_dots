"""Frozen one-fact patching design, exact candidate scoring and durable records."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

PILOT_PANEL = "discovery-filler-0743614aafcd1ceb0a07"
MODES = ("full_downstream", "answer_only")
LAYER_SETS = {"33-42": list(range(33, 43)), "0-42": list(range(43))}
LABELS = ("target_sum", "donor_sum", "mixed_sum")
TOLERANCE = 0.15
DESIGNS = ("original", "filler-coverage", "filler10-coverage", "filler10-confirmation", "filler-redundancy", "filler-repeat", "filler-random")


def design_interventions(design: str, *, repeat_sources=None) -> list[dict]:
    if repeat_sources is not None and design != "filler-repeat":
        raise ValueError("repeat_sources requires filler-repeat")
    if repeat_sources is not None:
        if (not repeat_sources or any(type(i) is not int or not 0 <= i < 19 for i in repeat_sources)
                or len(set(repeat_sources)) != len(repeat_sources)):
            raise ValueError("repeat_sources must be distinct integers from 0 through 18")
    if design == "original":
        pairs = [(f"filler_{i}", name, layers) for i in (5, 10)
                 for name, layers in LAYER_SETS.items()]
    elif design == "filler-coverage":
        pairs = [("all_fillers", "0-42", list(range(43))),
                 ("filler_5", "32-37", list(range(32, 38)))]
    elif design == "filler10-coverage":
        pairs = [("filler_10", "32-37", list(range(32, 38)))]
    elif design == "filler10-confirmation":
        pairs = [("filler_10", name, layers) for name, layers in
                 (("0-42", list(range(43))), ("32-37", list(range(32, 38))),
                  ("33-42", list(range(33, 43))))]
    elif design == "filler-random":
        return [{"site": f"random_after_{j}", "cutoff": j,
                 "filler_indices": list(range(j + 1, 20)), "layers": list(range(43)),
                 "layer_set": "0-42", "recomputation": "full_downstream"} for j in range(6)]
    elif design == "filler-repeat":
        return [{"site": f"repeat_filler_{i}", "source_filler_index": i,
                 "filler_indices": list(range(i + 1, 20)), "layers": list(range(43)),
                 "layer_set": "0-42", "recomputation": "full_downstream"} for i in ((5, 0) if repeat_sources is None else repeat_sources)]
    elif design == "filler-redundancy":
        return redundancy_interventions(PILOT_PANEL)
    else:
        raise ValueError(f"unknown design: {design}")
    return [{"site": site, "layer_set": name, "layers": layers, "recomputation": mode}
            for site, name, layers in pairs for mode in MODES]


def redundancy_interventions(panel_id: str, draws: int = 5, seed: int = 42) -> list[dict]:
    if not isinstance(draws, int) or isinstance(draws, bool) or draws < 1:
        raise ValueError("draw count must be a positive integer")
    rows = [{"site": "early_block", "family": "early_block", "subset_size": 5,
             "draw_id": 0, "filler_indices": list(range(5))}]
    for size in range(1, 6):
        for draw in range(draws):
            # Separate reproducible streams prevent nesting and dependence on loop order.
            rng = random.Random(int(digest([seed, panel_id, size, draw]), 16))
            rows.append({"site": f"later_{size}_draw_{draw}", "family": "later_subset",
                         "subset_size": size, "draw_id": draw,
                         "filler_indices": sorted(rng.sample(range(5, 20), size))})
    return [{**r, "layers": list(range(43)), "layer_set": "0-42",
             "recomputation": "full_downstream"} for r in rows]


def baseline_modes(manifest: dict) -> tuple[str, ...]:
    observed = {s["recomputation"] for s in manifest["trials"]}
    return tuple(m for m in MODES if m in observed)


def layer42_diagnostics(manifest: dict, panel_id: str) -> list[dict]:
    """One invariance check per target/donor/position set/mode, ignoring layer ranges.

    IDs also match the historical schema-2 campaign's diagnostic records.
    """
    unique = {}
    for spec in manifest["trials"]:
        if spec["panel_id"] != panel_id:
            continue
        key = (spec["target_id"], spec["donor_id"], tuple(spec["positions"]), spec["recomputation"])
        if key not in unique:
            unique[key] = {**spec, "layers": [42], "layer_set": "42",
                           "record_id": f"{spec['target_id']}|{spec['donor_role']}|{spec['site']}|L42|{spec['recomputation']}"}
    return list(unique.values())


def campaign_counts(manifest: dict, panel_ids: set[str] | None = None) -> dict:
    panels = [p for p in manifest["panels"] if panel_ids is None or p["panel_id"] in panel_ids]
    pids = {p["panel_id"] for p in panels}
    trials = [s for s in manifest["trials"] if s["panel_id"] in pids]
    return {"panels": len(panels), "targets": sum(len(p["cells"]) for p in panels),
            "trials": len(trials),
            "identity": sum(s["panel_id"] in pids for s in manifest["identity_controls"]),
            "baselines": len({(s["target_id"], s["recomputation"]) for s in trials})}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with temporary.open("xb") as sink:
        sink.write(data)
        sink.flush()
        os.fsync(sink.fileno())
    os.replace(temporary, path)
    sync_directory(path.parent)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())


class Journal:
    """Append+fsync results before atomically updating a disposable progress index.

    Recovery trusts checksummed results, never the index. Only an incomplete
    final line is quarantined; corrupt complete records fail closed. A complete
    last record missing its newline is retained and repaired.
    """
    def __init__(self, root: Path, config_hash: str):
        self.root, self.config_hash = root, config_hash
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / "results.jsonl"
        self.records: dict[str, dict] = {}
        if self.path.exists():
            data = self.path.read_bytes()
            offset = 0
            for line in data.splitlines(keepends=True):
                try:
                    envelope = json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    if offset + len(line) != len(data) or line.endswith(b"\n"):
                        raise ValueError("corrupt complete journal record")
                    atomic_bytes(root / f"interrupted-tail-{time.time_ns()}.bin", line)
                    atomic_bytes(self.path, data[:offset])
                    break
                record = envelope["record"]
                if envelope["sha256"] != digest(record) or record["config_hash"] != config_hash:
                    raise ValueError("journal checksum/config mismatch")
                rid = record["record_id"]
                if rid in self.records:
                    raise ValueError(f"duplicate durable result {rid}")
                self.records[rid] = record
                offset += len(line)
            else:
                if data and not data.endswith(b"\n"):
                    atomic_bytes(self.path, data + b"\n")
        self.progress()

    def progress(self, **status: Any) -> None:
        atomic_json(self.root / "progress.json", {
            "config_hash": self.config_hash, "completed_record_ids": sorted(self.records),
            "counts": {kind: sum(r["kind"] == kind for r in self.records.values())
                       for kind in sorted({r["kind"] for r in self.records.values()})},
            "updated_at": time.time(), **status,
        })

    def append(self, record: dict) -> dict:
        record = {**record, "config_hash": self.config_hash}
        rid = record["record_id"]
        if rid in self.records:
            raise ValueError(f"already completed {rid}")
        encoded = json.dumps({"record": record, "sha256": digest(record)},
                             sort_keys=True, allow_nan=False).encode() + b"\n"
        with self.path.open("ab") as sink:
            sink.write(encoded)
            sink.flush()
            os.fsync(sink.fileno())
        sync_directory(self.root)
        self.records[rid] = record
        self.progress()
        return record


def candidates(target: dict, donor: dict, tokenizer: Any) -> list[dict]:
    values = (target["left_value"] + target["right_value"],
              donor["left_value"] + donor["right_value"],
              donor["left_value"] + target["right_value"])
    result = []
    for label, value in zip(LABELS, values):
        tokens = list(tokenizer.encode(str(value), add_special_tokens=False))
        if len(tokens) != 1 or tokenizer.decode(tokens) != str(value):
            raise ValueError(f"candidate {label}={value} is not a canonical single token")
        result.append({"label": label, "value": int(value), "token_id": int(tokens[0])})
    return result


def requested_scores(response: dict, token_ids: list[int]) -> dict[int, float]:
    if len(response["output_ids"]) != 1:
        raise ValueError("expected exactly one generated token")
    rows = response["meta_info"]["output_token_ids_logprobs"]
    if len(rows) != 1:
        raise ValueError("expected one candidate scoring step")
    scores = {int(row[1]): float(row[0]) for row in rows[0]}
    if len(rows[0]) != len(scores) or set(scores) != set(token_ids):
        raise ValueError("missing, duplicate or unexpected explicitly requested candidate token")
    if any(not math.isfinite(v) or v > 1e-6 for v in scores.values()):
        raise ValueError("invalid natural log probability")
    return scores


def score_candidates(mapping: list[dict], clean: dict, patched: dict) -> list[dict]:
    # Baselines request the union of all candidates for their target.
    clean_ids = [int(row[1]) for row in clean["meta_info"]["output_token_ids_logprobs"][0]]
    patched_ids = [int(row[1]) for row in patched["meta_info"]["output_token_ids_logprobs"][0]]
    a, b = requested_scores(clean, clean_ids), requested_scores(patched, patched_ids)
    rows = [{**c, "clean_logprob": a[c["token_id"]], "patched_logprob": b[c["token_id"]],
             "delta_logprob": b[c["token_id"]] - a[c["token_id"]]} for c in mapping]
    if "raw_logits" in clean or "raw_logits" in patched:
        ca, cb = clean["raw_logits"], patched["raw_logits"]
        la = dict(zip(ca["scored_token_ids"], ca["logits"]))
        lb = dict(zip(cb["scored_token_ids"], cb["logits"]))
        for row in rows:
            token = row["token_id"]
            row.update(clean_logit=la[token], patched_logit=lb[token], delta_logit=lb[token] - la[token],
                       clean_log_normalizer=ca["log_normalizer"], patched_log_normalizer=cb["log_normalizer"])
    return rows


def invariant(clean: dict, changed: dict, token_ids: list[int], tolerance: float = TOLERANCE) -> dict:
    a, b = requested_scores(clean, token_ids), requested_scores(changed, token_ids)
    errors = {str(t): abs(b[t] - a[t]) for t in set(token_ids)}
    result = {"argmax_equal": clean["output_ids"] == changed["output_ids"],
              "candidate_errors": errors, "max_logprob_error": max(errors.values())}
    result["passed"] = result["argmax_equal"] and result["max_logprob_error"] <= tolerance
    if "raw_logits" in clean or "raw_logits" in changed:
        ca, cb = clean["raw_logits"], changed["raw_logits"]
        la = dict(zip(ca["scored_token_ids"], ca["logits"]))
        lb = dict(zip(cb["scored_token_ids"], cb["logits"]))
        result["max_logit_error"] = max(abs(lb[t] - la[t]) for t in token_ids)
        result["log_normalizer_error"] = abs(cb["log_normalizer"] - ca["log_normalizer"])
        result["passed"] &= max(result["max_logit_error"], result["log_normalizer_error"]) <= tolerance
    return result


def build_manifest(rendered: dict, tokenizer: Any, design: str = "original", *, draws: int | None = None, seed: int = 42, repeat_sources=None) -> dict:
    draws = (1 if design == "filler-random" else 5) if draws is None else draws
    if design == "filler-random" and (type(draws) is not int or draws != 1 or type(seed) is not int or seed != 42):
        raise ValueError("filler-random requires one realization, seed 42")
    interventions = design_interventions(design, repeat_sources=repeat_sources)
    panels = sorted(rendered["eligible_panels"], key=lambda p: p["panel_id"])
    splits = {c.get("split", rendered.get("split")) for p in panels for c in p["cells"]}
    split = "confirmation" if design == "filler10-confirmation" else "discovery"
    if rendered["filler_length"] != 20 or splits != {split}:
        raise ValueError(f"requires the existing k=20 {split} set")
    if split == "discovery":
        if len(panels) != 24 or panels[0]["panel_id"] != PILOT_PANEL:
            raise ValueError("discovery panel identity/count changed")
    elif (len(panels) != 60 or len({p["panel_id"] for p in panels}) != 60
          or any(not p["panel_id"].startswith("confirmation-filler-") for p in panels)
          or rendered.get("rejected_panel_count", 0) or rendered.get("rejected_panels")):
        raise ValueError("requires all 60 distinct confirmation panels without exclusions")
    pilot_panel = panels[0]["panel_id"]
    trials, controls, prepared = [], [], []
    for panel in panels:
        if design == "filler-redundancy":
            interventions = redundancy_interventions(panel["panel_id"], draws, seed)
        cells = sorted(panel["cells"], key=lambda c: c["cell_id"])
        lookup = {(c["row"], c["col"]): c for c in cells}
        if len(cells) != 4 or set(lookup) != {(0, 0), (0, 1), (1, 0), (1, 1)}:
            raise ValueError("incomplete factorial panel")
        lengths = {len(c["input_ids"]) for c in cells}
        if len(lengths) != 1 or len(panel["positions"]["fillers"]) != 20:
            raise ValueError("panel token alignment changed")
        n = lengths.pop()
        fillers = panel["positions"]["fillers"]
        sites = {f"filler_{i}": fillers[i] for i in (5, 10)}
        if len(set(fillers)) != 20 or any(p < 0 or p >= n - 1 for p in fillers):
            raise ValueError("filler position is not before the answer prediction")
        if design == "filler-coverage" and (fillers != list(range(fillers[0], fillers[0] + 20))
                                            or n - 1 - fillers[-1] != 3):
            raise ValueError("filler-coverage requires exactly two non-filler rows before the answer position")
        for cell in cells:
            if list(tokenizer.encode(cell["rendered_prompt"], add_special_tokens=False)) != cell["input_ids"]:
                raise ValueError("rendered prompt no longer matches saved tokenization")
            if cell["target"] != cell["left_value"] + cell["right_value"]:
                raise ValueError("incorrect target arithmetic")
        for target in cells:
            for role in (("identity", "random") if design == "filler-random" else ("identity", "repeat") if design == "filler-repeat" else ("identity", "left", "both")):
                donor = target if role in {"identity", "repeat", "random"} else lookup[(1 - target["row"],
                           target["col"] if role == "left" else 1 - target["col"])]
                mapping = candidates(target, donor, tokenizer)
                for intervention in interventions:
                    site, name, mode = (intervention[k] for k in ("site", "layer_set", "recomputation"))
                    positions = ([fillers[i] for i in intervention["filler_indices"]] if design in {"filler-redundancy", "filler-repeat", "filler-random"}
                                 else list(fillers) if site == "all_fillers" else [sites[site]])
                    row = {"record_id": f"{target['cell_id']}|{role}|{site}|L{name}|{mode}",
                           "kind": "identity" if role == "identity" else "trial",
                           "panel_id": panel["panel_id"], "target_id": target["cell_id"],
                           "donor_id": donor["cell_id"], "donor_role": role,
                           **intervention, "positions": positions, "candidates": mapping}
                    if design == "filler-repeat":
                        row["source_positions"] = positions.copy() if role == "identity" else [fillers[intervention["source_filler_index"]]] * len(positions)
                        row["candidates"] = [mapping[0]]
                    if design == "filler-random":
                        row.update(source_positions=positions.copy(), candidates=[mapping[0]],
                                   replacement="clean" if role == "identity" else "random")
                    (controls if role == "identity" else trials).append(row)
        prepared.append({**panel, "cells": cells, "answer_position": n - 1, "sites": sites})
    payload = {"schema_version": 2, "experiment": "one_fact_simultaneous_residual_patching",
               "panels": prepared, "pilot_panel": pilot_panel, "trials": trials,
               "identity_controls": controls, "filler_length": 20,
               "semantics": {"residual": "complete post-block mHC residual on every TP rank",
                   "scoring": "natural log, full vocabulary normalization, explicitly requested tokens",
                   "answer_only": "restore clean post-block rows at all layers except patched filler and final prompt row",
                   "clean_filter": f"none; all {split} cells retained"},
               "runtime_requirements": {"tp_size": 4, "disable_radix_cache": True,
                   "disable_cuda_graph": True, "chunked_prefill_size": -1,
                   "max_running_requests": 1, "enable_return_hidden_states": True,
                   "fused_mhc_post_pre": False}, "tolerance": TOLERANCE}
    payload["counts"] = campaign_counts(payload)
    payload["pilot_counts"] = {k: v for k, v in campaign_counts(payload, {pilot_panel}).items() if k != "panels"}
    if design != "original":
        payload.update(schema_version=3, design=design, interventions=interventions,
                       diagnostics_per_runtime=len(layer42_diagnostics(payload, pilot_panel)))
    if design == "filler-coverage":
        payload["semantics"].update(
            answer_only="restore clean post-block rows at every layer except all selected fillers and final prompt row; selected rows keep evolving after replacement ends",
            all_fillers_mode_contrast="tests propagation through the two intervening non-filler rows before the answer-prediction row; all 20 fillers are overwritten at every layer",
            comparisons="matched recomputation-mode and donor-type contrasts; position set and layer range change together between intervention families")
    if design == "filler10-coverage":
        payload["semantics"].update(
            answer_only="restore clean post-block rows at every layer except filler_10 and final prompt row; both continue evolving through layer 42 after replacement ends at 37",
            comparisons="reuse filler_10 at 0-42 and 33-42 and filler_5 at all three ranges with original runtime/mode baselines")
    if design == "filler-redundancy":
        payload.update(schema_version=4, raw_logits=True, draws=draws, seed=seed,
                       interventions_by_panel={p["panel_id"]: redundancy_interventions(p["panel_id"], draws, seed) for p in panels},
                       input_hashes={c["cell_id"]: {"input_ids": digest(c["input_ids"]),
                           "prompt": digest(c["rendered_prompt"])} for p in panels for c in p["cells"]})
        payload.pop("interventions")
        payload["semantics"].pop("answer_only")
        payload["semantics"].update(recomputation="fresh full prompt; no prefix cache or clean restoration",
            sampling="uniform without replacement within subsets; independent panel/size/draw streams; repeats allowed",
            scoring="pre-sampling raw candidate logits and full-vocabulary logsumexp; native log probabilities also retained")
    if design == "filler-repeat":
        payload.update(schema_version=5, raw_logits=True, bootstrap={"resamples": 2000, "seed": 42, "unit": "panel"},
                       input_hashes={c["cell_id"]: {"input_ids": digest(c["input_ids"]), "prompt": digest(c["rendered_prompt"])} for p in panels for c in p["cells"]})
        payload["semantics"].pop("answer_only")
        payload["semantics"].update(recomputation="fresh full prompt; downstream and answer-prefix rows evolve normally",
            source="same runtime clean prompt; paired source_positions and positions at each layer on all ranks",
            comparisons="source comparisons combine source-position and replacement-count effects")
        payload["repeat_sources"] = [s["source_filler_index"] for s in interventions]
    if split == "confirmation":
        payload["split"] = split
        payload["semantics"]["comparisons"] = "all three filler_10 ranges rerun on the full confirmation split; no discovery trials pooled"
    if design == "filler-random":
        from filler.dsv4.filler_random import manifest_metadata
        payload.update(manifest_metadata(prepared, tokenizer, seed))
        payload["semantics"].pop("answer_only")
        payload["semantics"].update(
            recomputation="fresh full prompt; answer-prefix and downstream rows evolve normally",
            sampling="independent Gaussian direction per prompt/layer/token, clean full-tensor L2 norm; shared across cutoffs and ranks",
            comparisons="conditional on one noise realization; cutoffs also change replacement count")
    payload["config_hash"] = digest(payload)
    return payload
