"""Pilot-to-full-split controller; all records carry their original runtime baseline."""
from __future__ import annotations

import json
import signal
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any

from filler.dsv4.patching import (
    Journal, baseline_modes, atomic_json, digest, file_digest, invariant,
    requested_scores, score_candidates, campaign_counts, layer42_diagnostics,
)


class WalltimeReached(Exception):
    pass


def native_equivalence(capture: dict, response: dict, checkpoint: Path, token_ids: list[int],
                       *, device: str = "cpu", projector=None) -> dict:
    """Native mHC collapse + RMSNorm + full-vocabulary head, at the final row."""
    import torch
    from filler.dsv4.lens import load_checkpoint_readout, project_logits

    states = []
    for rank in range(4):
        ref = capture["ranks"][str(rank)]
        if file_digest(Path(ref["path"])) != ref["sha256"]:
            raise ValueError("native validation capture checksum mismatch")
        saved = torch.load(ref["path"], map_location="cpu", weights_only=True)
        states.append(saved["states"][42][-1])
    rank_errors = [float((s.float() - states[0].float()).abs().max()) for s in states[1:]]
    returned = torch.tensor(response["meta_info"]["hidden_states"][0])
    if returned.ndim == 2:
        returned = returned[-1]
    hidden_error = float((returned.flatten().float() - states[0].flatten().float()).abs().max())
    weights = load_checkpoint_readout(checkpoint, device=device)
    projector = project_logits if projector is None else projector
    with torch.inference_mode():
        projected = projector(states[0].to(device), weights).float()
        lp = torch.log_softmax(projected, -1)
    native = requested_scores(response, token_ids)
    native.update({int(row[1]): float(row[0]) for row in response["meta_info"]["output_top_logprobs"][0]})
    errors = {str(token): abs(float(lp[token]) - value) for token, value in native.items()}
    result = {"argmax_equal": int(lp.argmax()) == response["output_ids"][0],
              "returned_hidden_max_abs_difference": hidden_error,
              "rank_max_abs_differences": rank_errors,
              "candidate_and_top_logprob_errors": errors, "max_logprob_error": max(errors.values())}
    result["passed"] = result["argmax_equal"] and hidden_error == 0 and not any(rank_errors) and max(errors.values()) <= 0.15
    if "raw_logits" in response:
        raw = response["raw_logits"]
        result["max_logit_error"] = max(abs(float(projected[t]) - v) for t, v in zip(raw["scored_token_ids"], raw["logits"]))
        result["log_normalizer_error"] = abs(float(torch.logsumexp(projected, -1)) - raw["log_normalizer"])
        result["passed"] &= max(result["max_logit_error"], result["log_normalizer_error"]) <= 0.15
    return result


class HTTPTransport:
    def __init__(self, base_url: str, control_root: Path):
        self.base_url, self.control_root = base_url, control_root

    def run(self, control: dict, input_ids: list[int], scored_ids: list[int]) -> tuple[dict, list[dict]]:
        from filler.dsv4.campaign_hook import validate_acknowledgements
        from scripts.dsv4.run_activation_patching import flush_cache, post_json

        # Flush before arming; a flush must never consume an intervention.
        flush_cache(self.base_url)
        atomic_json(self.control_root / "NEXT.json", control)
        response = post_json(self.base_url + "/generate", {
            "input_ids": input_ids, "sampling_params": {"temperature": 0, "max_new_tokens": control.get("max_new_tokens", 1)},
            "return_logprob": True, "top_logprobs_num": 20,
            "token_ids_logprob": list(dict.fromkeys(scored_ids)),
            "return_hidden_states": control["capture_all"],
        }, timeout=600)
        # Preserve even failed checks' raw response and exact control for diagnosis.
        atomic_json(Path(control["output_root"]) / "response.json", response)
        if response["meta_info"].get("cached_tokens") != 0:
            raise RuntimeError("prefix cache reuse detected")
        from filler.dsv4.patching_logits import first_prediction
        requested_scores(first_prediction(response, control.get("max_new_tokens", 1)), scored_ids)
        acks = validate_acknowledgements(self.control_root, control)
        if control.get("raw_logits"):
            from filler.dsv4.patching_logits import collect_logits
            response = collect_logits(control, response)
        return response, acks


def read_response(record: dict) -> dict:
    ref = record["response"]
    path = Path(ref["path"])
    if file_digest(path) != ref["sha256"]:
        raise ValueError("raw response checksum mismatch")
    return json.loads(path.read_text())


class Campaign:
    def __init__(self, manifest: dict, root: Path, runtime_id: str, transport: Any,
                 checkpoint: Path, *, deadline: float, validation=native_equivalence):
        self.manifest, self.root, self.runtime_id = manifest, root, runtime_id
        self.journal = Journal(root, manifest["config_hash"])
        self.transport, self.checkpoint, self.deadline = transport, checkpoint, deadline
        self.validation = validation
        self.runtime_root = root / "runtimes" / runtime_id
        self.durations: list[float] = []
        self.runtime_validated = False
        self.stop_requested = False
        self.current: dict | None = None
        self.replacement_banks: dict = {}
        unfinished_panels = {s["panel_id"] for s in self.remaining_specs()}
        rechecks = sum(s["panel_id"] in unfinished_panels and s["record_id"] in self.journal.records
                       for s in self.manifest["identity_controls"])
        first_active = next((p["panel_id"] for p in manifest["panels"] if p["panel_id"] in unfinished_panels), None)
        self.diagnostic_count = len(layer42_diagnostics(manifest, first_active)) if first_active else 0
        self.planned_passes = (len(self.remaining_specs()) + campaign_counts(manifest, unfinished_panels)["baselines"]
                              + rechecks + self.diagnostic_count)

    def request_stop(self, *_args):
        # Finish and fsync an in-flight result when Slurm gives sufficient notice.
        self.stop_requested = True

    def remaining_specs(self) -> list[dict]:
        return [s for key in ("identity_controls", "trials") for s in self.manifest[key]
                if s["record_id"] not in self.journal.records]

    def check_time(self):
        mean = sum(self.durations) / len(self.durations) if self.durations else 30.0
        remaining = self.deadline - time.time()
        estimate = mean * max(0, self.planned_passes - len(self.durations))
        self.journal.progress(runtime_id=self.runtime_id, seconds_remaining=remaining,
                              mean_pass_seconds=mean, projected_remaining_seconds=estimate,
                              projection_includes=f"unfinished trials/controls, fresh baselines, identity rechecks and {self.diagnostic_count} layer-42 diagnostics; excludes reporting",
                              phase="pilot" if any(s["panel_id"] == self.manifest["pilot_panel"] for s in self.remaining_specs()) else self.manifest.get("split", "discovery"))
        if self.stop_requested or remaining < max(120, 2 * mean):
            raise WalltimeReached("checkpoint retained before allocation deadline")

    def execute(self, *, cell: dict, token_ids: list[int], mode: str, layers=(), positions=(),
                clean=None, donor=None, capture_all=False, source_positions=None, replacement_bank=None) -> dict:
        self.check_time()
        request_id = uuid.uuid4().hex
        output_root = self.runtime_root / "passes" / request_id
        control = {"request_id": request_id, "runtime_id": self.runtime_id,
                   "config_hash": self.manifest["config_hash"], "cell_id": cell["cell_id"],
                   "input_ids_hash": digest(cell["input_ids"]), "num_tokens": len(cell["input_ids"]),
                   "layers": list(layers), "positions": list(positions), "recomputation": mode,
                   "capture_all": capture_all, "clean_capture": clean, "donor_capture": donor,
                   "output_root": str(output_root)}
        if source_positions is not None:
            control["source_positions"] = list(source_positions)
        if self.manifest.get("design") == "filler-random":
            control["capture_positions"] = [p["absolute_position"] for p in self.manifest["position_coverage"][cell["cell_id"]]]
        if replacement_bank is not None:
            control["replacement_bank"] = replacement_bank
        if self.manifest.get("raw_logits"):
            control.update(raw_logits=True, scored_token_ids=list(dict.fromkeys(token_ids)))
        if self.manifest.get("max_new_tokens"):
            control["max_new_tokens"] = self.manifest["max_new_tokens"]
        self.current = control
        atomic_json(output_root / "control.json", control)
        started = time.time()
        response, acks = self.transport.run(control, cell["input_ids"], list(dict.fromkeys(token_ids)))
        # Fake transports in CPU integration tests follow the same durable contract.
        atomic_json(output_root / "response.json", response)
        atomic_json(output_root / "acks.json", acks)
        from filler.dsv4.patching_logits import first_prediction
        requested_scores(first_prediction(response, control.get("max_new_tokens", 1)), token_ids)
        if control.get("raw_logits"):
            from filler.dsv4.patching_logits import validate_logits
            if response["raw_logits"] != validate_logits(response, control):
                raise ValueError("required raw logits missing or inconsistent")
        self.durations.append(time.time() - started)
        self.current = None
        return {"runtime_id": self.runtime_id, "request_id": request_id,
                "response": {"path": str(output_root / "response.json"),
                             "sha256": file_digest(output_root / "response.json")},
                "control_path": str(output_root / "control.json"),
                "control_sha256": file_digest(output_root / "control.json"),
                "ack_path": str(output_root / "acks.json"),
                "ack_sha256": file_digest(output_root / "acks.json"),
                "capture": {"cell_id": cell["cell_id"], "runtime_id": self.runtime_id,
                            "ranks": {str(a["rank"]): a["capture"] for a in acks}},
                "seconds": self.durations[-1]}

    def panel_baselines(self, panel: dict) -> dict:
        panel_specs = [s for s in self.manifest["trials"] if s["panel_id"] == panel["panel_id"]]
        baselines = {}
        for cell in panel["cells"]:
            ids = sorted({c["token_id"] for s in panel_specs if s["target_id"] == cell["cell_id"] for c in s["candidates"]})
            clean = None
            for mode in baseline_modes(self.manifest):
                result = self.execute(cell=cell, token_ids=ids, mode=mode, clean=clean,
                                      capture_all=mode == "full_downstream")
                if mode == "full_downstream":
                    clean = result["capture"]
                baseline = self.journal.append({**result, "record_id": f"baseline|{self.runtime_id}|{cell['cell_id']}|{mode}",
                    "kind": "baseline", "panel_id": panel["panel_id"], "target_id": cell["cell_id"],
                    "recomputation": mode, "scored_token_ids": ids, "clean_capture": clean})
                baselines[cell["cell_id"], mode] = baseline
                if self.manifest.get("design") == "filler-random":
                    from filler.dsv4.filler_random import create_bank
                    self.replacement_banks[baseline["record_id"]] = create_bank(
                        self.manifest, baseline, self.runtime_root / "replacement_banks" / f"{result['request_id']}.pt")
                if not self.runtime_validated:
                    report = self.validation(clean, read_response(baseline), self.checkpoint, ids)
                    self.journal.append({"record_id": f"native-validation|{self.runtime_id}",
                        "kind": "native_validation", "runtime_id": self.runtime_id,
                        "baseline_id": baseline["record_id"], "report": report})
                    if not report["passed"]:
                        raise RuntimeError("native final-layer equivalence failed")
                    self.runtime_validated = True
        return baselines

    def run_spec(self, spec: dict, cells: dict, baselines: dict, *, recheck=False, diagnostic=False):
        target = cells[spec["target_id"]]
        baseline = baselines[spec["target_id"], spec["recomputation"]]
        clean = baseline["clean_capture"]
        donor = baselines[spec["donor_id"], "full_downstream"]["clean_capture"]
        is_control = spec["kind"] == "identity" or diagnostic
        token_ids = baseline["scored_token_ids"] if is_control else list(dict.fromkeys(c["token_id"] for c in spec["candidates"]))
        bank = self.replacement_banks[baseline["record_id"]] if spec.get("replacement") == "random" else None
        result = self.execute(cell=target, token_ids=token_ids, mode=spec["recomputation"],
                              layers=spec["layers"], positions=spec["positions"], clean=clean, donor=donor,
                              source_positions=spec.get("source_positions"), replacement_bank=bank)
        raw_clean, raw_patched = read_response(baseline), read_response(result)
        scores = score_candidates(spec["candidates"], raw_clean, raw_patched)
        gate = invariant(raw_clean, raw_patched, token_ids) if is_control else None
        record = {**spec, **result, "baseline_id": baseline["record_id"],
                  "clean_capture": clean, "donor_capture": donor, "scores": scores, "gate": gate}
        if bank is not None:
            record["replacement_bank"] = bank
        if recheck or diagnostic:
            record.update(record_id=f"{self.runtime_id}|{spec['record_id']}",
                          planned_id=spec["record_id"], kind="layer42_validation" if diagnostic else "identity_recheck")
        if gate is not None and not gate["passed"]:
            # A failed gate is diagnostic evidence, never a completed planned control.
            record.update(record_id=f"failed|{self.runtime_id}|{spec['record_id']}", kind="failed_validation")
        self.journal.append(record)
        if gate is not None and not gate["passed"]:
            raise RuntimeError(f"invariance gate failed: {spec['record_id']}")
        print(f"completed {record['kind']} {spec['record_id']} ({result['seconds']:.1f}s)", flush=True)

    def run(self) -> dict:
        first_panel = True
        try:
            for panel in self.manifest["panels"]:
                pid = panel["panel_id"]
                if not any(s["panel_id"] == pid for s in self.remaining_specs()):
                    continue
                cells = {c["cell_id"]: c for c in panel["cells"]}
                baselines = self.panel_baselines(panel)
                # Recheck every identity against fresh baselines after any model restart.
                for spec in self.manifest["identity_controls"]:
                    if spec["panel_id"] == pid:
                        self.run_spec(spec, cells, baselines, recheck=spec["record_id"] in self.journal.records)
                if first_panel:
                    for diag in layer42_diagnostics(self.manifest, pid):
                        self.run_spec(diag, cells, baselines, diagnostic=True)
                    first_panel = False
                self.journal.append({"record_id": f"panel-validation|{self.runtime_id}|{pid}",
                    "kind": "panel_validation", "runtime_id": self.runtime_id, "panel_id": pid,
                    "passed": True, "baseline_ids": [r["record_id"] for r in baselines.values()]})
                for spec in self.manifest["trials"]:
                    if spec["panel_id"] == pid and spec["record_id"] not in self.journal.records:
                        self.run_spec(spec, cells, baselines)
                if pid == self.manifest["pilot_panel"]:
                    verify_records(self.manifest, self.journal, panels={pid})
                    self.journal.append({"record_id": f"pilot-passed|{self.runtime_id}", "kind": "pilot_passed",
                                         "runtime_id": self.runtime_id, "passed": True})
                    print(f"Pilot gates passed; continuing {self.manifest.get('split', 'discovery')} on the same server.", flush=True)
            integrity = verify_records(self.manifest, self.journal)
            self.journal.progress(status="inference_complete", integrity=integrity)
            return integrity
        except WalltimeReached:
            self.journal.progress(status="checkpointed", runtime_id=self.runtime_id)
            raise
        except BaseException as exc:
            atomic_json(self.runtime_root / "failure.json", {
                "error": f"{type(exc).__name__}: {exc}", "active_control": self.current,
                "time": time.time(), "runtime_id": self.runtime_id})
            self.journal.progress(status="validation_or_execution_failed", runtime_id=self.runtime_id)
            raise


def verify_records(manifest: dict, journal: Journal, *, panels: set[str] | None = None) -> dict:
    """Recompute all stored deltas from original raw responses and matched baselines."""
    planned = {s["record_id"]: s for key in ("identity_controls", "trials") for s in manifest[key]
               if panels is None or s["panel_id"] in panels}
    missing = set(planned) - journal.records.keys()
    if missing:
        raise ValueError(f"{len(missing)} planned trials/controls unfinished")
    extra = {rid for rid, r in journal.records.items() if r["kind"] in {"trial", "identity"}
             and (panels is None or r["panel_id"] in panels)} - planned.keys()
    if extra:
        raise ValueError("unexpected planned records")
    baselines = set()
    checked = set()
    diagnostic_runtimes = set()
    bank_cache = OrderedDict()
    verified_banks = {}
    def check_capture(capture: dict):
        if set(capture["ranks"]) != {"0", "1", "2", "3"}:
            raise ValueError("incomplete referenced capture ranks")
        for ref in capture["ranks"].values():
            key = (ref["path"], ref["sha256"])
            if key not in checked:
                if file_digest(Path(ref["path"])) != ref["sha256"]:
                    raise ValueError("saved tensor capture checksum mismatch")
                checked.add(key)
    def check_repeat_tensors(record, control, acks):
        import torch
        for ack in acks:
            saved = torch.load(ack["capture"]["path"], map_location="cpu", weights_only=True)
            meta = saved["metadata"]
            for key in ("request_id", "runtime_id", "config_hash", "cell_id", "num_tokens", "rank", "positions"):
                if meta[key] != ack[key]:
                    raise ValueError("tensor metadata differs from acknowledgement")
            if saved["audits"] != ack["audits"] or set(saved["states"]) != set(range(43)):
                raise ValueError("tensor audit or layer coverage mismatch")
            if "source_positions" in control and meta.get("source_positions") != control["source_positions"]:
                raise ValueError("tensor source mapping mismatch")
            positions = meta["positions"]
            if any(t.ndim != 3 or len(t) != len(positions) for t in saved["states"].values()):
                raise ValueError("tensor position coverage mismatch")
            if manifest.get("design") == "filler-random":
                if meta.get("replacement_bank") != control.get("replacement_bank"):
                    raise ValueError("tensor random bank metadata mismatch")
                if any(not torch.isfinite(t).all() for t in saved["states"].values()):
                    raise ValueError("nonfinite saved residual")
            if record["kind"] == "baseline":
                continue
            ref = record["donor_capture"]["ranks"][str(ack["rank"])]
            donor = torch.load(ref["path"], map_location="cpu", weights_only=True)
            if donor["metadata"]["positions"] != list(range(control["num_tokens"])):
                raise ValueError("repeat requires complete clean capture")
            destinations = [positions.index(p) for p in control["positions"]]
            for lid in control["layers"]:
                actual = saved["states"][lid][destinations]
                if control.get("replacement_bank"):
                    bank = bank_cache[control["replacement_bank"]["path"]]
                    indices = [bank["metadata"]["positions"].index(p) for p in control["positions"]]
                    expected = bank["states"][lid][indices].to(actual.dtype)
                else:
                    expected = donor["states"][lid][control["source_positions"]].to(actual.dtype)
                if not torch.equal(actual, expected):
                    raise ValueError("saved destination tensor differs from clean source")

    def check_artifacts(record: dict):
        from filler.dsv4.campaign_hook import validate_ack_records
        for stem in ("control", "ack"):
            if file_digest(Path(record[f"{stem}_path"])) != record[f"{stem}_sha256"]:
                raise ValueError(f"{stem} artifact checksum mismatch")
        control = json.loads(Path(record["control_path"]).read_text())
        acks = json.loads(Path(record["ack_path"]).read_text())
        validate_ack_records(acks, control)
        if manifest.get("raw_logits"):
            from filler.dsv4.patching_logits import validate_logits
            if not control.get("raw_logits"):
                raise ValueError("required raw logits capture missing")
            raw = read_response(record)
            if raw["raw_logits"] != validate_logits(raw, control):
                raise ValueError("required raw logits missing or inconsistent")
        if control["runtime_id"] != record["runtime_id"] or control["config_hash"] != manifest["config_hash"]:
            raise ValueError("control runtime mismatch")
        if control["cell_id"] != record["target_id"] or control["recomputation"] != record["recomputation"]:
            raise ValueError("control target/mode mismatch")
        if record["kind"] != "baseline":
            for key in ("layers", "positions", "clean_capture", "donor_capture"):
                if control[key] != record[key]:
                    raise ValueError(f"control {key} differs from recorded intervention")
        if "source_positions" in record and control.get("source_positions") != record["source_positions"]:
            raise ValueError("control source_positions differs from recorded intervention")
        if manifest.get("design") in {"filler-repeat", "filler-random"} and record["kind"] != "baseline" and record["donor_capture"] != record["clean_capture"]:
            raise ValueError("repeat source must be the same runtime clean capture")
        for ack in acks:
            ref = record["capture"]["ranks"][str(ack["rank"])]
            if ref != ack["capture"]:
                raise ValueError("result and acknowledged capture differ")
        check_capture(record["capture"])
        if manifest.get("design") == "filler-random":
            required = [p["absolute_position"] for p in manifest["position_coverage"][record["target_id"]]]
            if control.get("capture_positions") != required:
                raise ValueError("incomplete required prompt position coverage")
            if record.get("replacement") == "random":
                from filler.dsv4.filler_random import validate_bank
                ref = record["replacement_bank"]
                if control.get("replacement_bank") != ref:
                    raise ValueError("control random bank mismatch")
                baseline = journal.records[record["baseline_id"]]
                key = ref["path"]
                if key not in verified_banks:
                    bank_cache[key] = validate_bank(ref, manifest, baseline)
                    verified_banks[key] = ref.copy()
                else:
                    if ref != verified_banks[key] or file_digest(Path(key)) != ref["sha256"] or ref["baseline_id"] != baseline["record_id"]:
                        raise ValueError("random bank reference mismatch")
                    if key not in bank_cache:
                        from filler.dsv4.filler_random import load_capture
                        bank_cache[key] = load_capture(ref)
                bank_cache.move_to_end(key)
                while len(bank_cache) > 4:
                    bank_cache.popitem(last=False)
                # One runtime baseline must use the same bank across all cutoffs.
                for other in journal.records.values():
                    if other.get("baseline_id") == record["baseline_id"] and other.get("replacement") == "random" and other.get("replacement_bank") != ref:
                        raise ValueError("cutoffs must share the same random bank")
            elif control.get("replacement_bank") is not None:
                raise ValueError("identity/baseline cannot use random replacements")
        if manifest.get("design") in {"filler-repeat", "filler-random"}:
            expected_hash = manifest["input_hashes"][record["target_id"]]["input_ids"]
            if control["input_ids_hash"] != expected_hash:
                raise ValueError("control input differs from frozen prompt")
            check_repeat_tensors(record, control, acks)
    for rid, spec in planned.items():
        record = journal.records[rid]
        for key, value in spec.items():
            if record[key] != value:
                raise ValueError(f"result differs from manifest: {key}")
        baseline = journal.records[record["baseline_id"]]
        baselines.add((baseline["target_id"], baseline["recomputation"]))
        if (baseline["runtime_id"] != record["runtime_id"] or baseline["target_id"] != record["target_id"]
                or baseline["recomputation"] != record["recomputation"]
                or baseline["clean_capture"] != record["clean_capture"]
                or record["donor_capture"]["runtime_id"] != record["runtime_id"]
                or record["donor_capture"]["cell_id"] != spec["donor_id"]):
            raise ValueError("baseline/capture runtime matching failed")
        check_capture(record["clean_capture"])
        check_capture(record["donor_capture"])
        for artifact_record in (baseline, record):
            if artifact_record["record_id"] not in checked:
                check_artifacts(artifact_record)
                checked.add(artifact_record["record_id"])
        native = journal.records[f"native-validation|{record['runtime_id']}"]
        if not native["report"]["passed"]:
            raise ValueError("native runtime validation missing")
        if spec["kind"] == "trial":
            panel_gate = journal.records[f"panel-validation|{record['runtime_id']}|{record['panel_id']}"]
            if not panel_gate["passed"] or record["baseline_id"] not in panel_gate["baseline_ids"]:
                raise ValueError("panel validation missing")
            diagnostic_runtimes.add(record["runtime_id"])
        scores = score_candidates(spec["candidates"], read_response(baseline), read_response(record))
        if scores != record["scores"]:
            raise ValueError("stored candidate deltas disagree with raw responses")
        if spec["kind"] == "identity":
            gate = invariant(read_response(baseline), read_response(record), baseline["scored_token_ids"])
            if not gate["passed"] or gate != record["gate"]:
                raise ValueError("identity gate failed during integrity check")
    if manifest.get("design") in {"filler-repeat", "filler-random"}:
        for record in journal.records.values():
            if record["kind"] != "identity_recheck" or (panels is not None and record["panel_id"] not in panels):
                continue
            spec = next(s for s in manifest["identity_controls"] if s["record_id"] == record["planned_id"])
            for key, value in spec.items():
                if key not in {"kind", "record_id"} and record[key] != value:
                    raise ValueError("identity recheck differs from manifest")
            baseline = journal.records[record["baseline_id"]]
            if baseline["runtime_id"] != record["runtime_id"] or baseline["clean_capture"] != record["clean_capture"]:
                raise ValueError("identity recheck baseline mismatch")
            check_artifacts(record)
            gate = invariant(read_response(baseline), read_response(record), baseline["scored_token_ids"])
            if not gate["passed"] or gate != record["gate"]:
                raise ValueError("identity recheck gate failed")
    for runtime in diagnostic_runtimes:
        diagnostics = [r for r in journal.records.values()
                       if r["kind"] == "layer42_validation" and r["runtime_id"] == runtime]
        first_gate = next(r for r in journal.records.values()
                          if r["kind"] == "panel_validation" and r["runtime_id"] == runtime)
        expected = {s["record_id"]: s for s in layer42_diagnostics(manifest, first_gate["panel_id"])}
        if len(diagnostics) != len(expected) or {r["planned_id"] for r in diagnostics} != expected.keys():
            raise ValueError(f"all {len(expected)} layer-42 diagnostics required before substantive trials")
        for record in diagnostics:
            for key, value in expected[record["planned_id"]].items():
                if key not in {"record_id", "kind"} and record[key] != value:
                    raise ValueError(f"layer-42 diagnostic differs from manifest: {key}")
            baseline = journal.records[record["baseline_id"]]
            if (baseline["runtime_id"] != runtime or baseline["target_id"] != record["target_id"]
                    or baseline["recomputation"] != record["recomputation"]
                    or baseline["clean_capture"] != record["clean_capture"]
                    or record["donor_capture"]["runtime_id"] != runtime
                    or record["donor_capture"]["cell_id"] != record["donor_id"]):
                raise ValueError("diagnostic baseline/capture runtime matching failed")
            check_capture(record["clean_capture"])
            check_capture(record["donor_capture"])
            check_artifacts(baseline)
            check_artifacts(record)
            gate = invariant(read_response(baseline), read_response(record), baseline["scored_token_ids"])
            if record["layers"] != [42] or not gate["passed"] or gate != record["gate"]:
                raise ValueError("layer-42 invariance failed during integrity check")
    return {"trials": sum(s["kind"] == "trial" for s in planned.values()),
            "identity": sum(s["kind"] == "identity" for s in planned.values()),
            "matched_baseline_cells": len(baselines), "passed": True}
