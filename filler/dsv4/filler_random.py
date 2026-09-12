"""Frozen Gaussian directions with runtime-clean full mHC norms."""
from __future__ import annotations

from pathlib import Path

import torch

from filler.dsv4.patching import digest, file_digest

NORM_TOLERANCE = .005
ALGORITHM = "cpu-torch-float32-gaussian-full-mhc-l2-v1"


def stream_seed(seed, cell_id, layer, position):
    return int(digest([ALGORITHM, seed, cell_id, layer, position])[:16], 16) & ((1 << 63) - 1)


def manifest_metadata(panels, tokenizer, seed):
    from filler.dsv4.lens_positions import prompt_position_labels, validate_positions
    coverage, seeds = {}, {}
    for panel in panels:
        for cell in panel["cells"]:
            ids = cell["input_ids"]
            absolute = [panel["positions"]["last_question"], *panel["positions"]["fillers"],
                        *range(len(ids) - 3, len(ids))]
            positions = [{"label": label, "absolute_position": p, "token_id": ids[p],
                          "token": tokenizer.decode([ids[p]])}
                         for label, p in zip(prompt_position_labels(20), absolute)]
            validate_positions(dict(input_ids=ids, positions=positions, filler_length=20), tokenizer.decode)
            coverage[cell["cell_id"]] = positions
            seeds[cell["cell_id"]] = {str(l): {str(p): stream_seed(seed, cell["cell_id"], l, p)
                for p in panel["positions"]["fillers"][1:]} for l in range(43)}
    return dict(schema_version=6, raw_logits=True, draws=1, seed=seed,
                random_algorithm=ALGORITHM, norm_tolerance=NORM_TOLERANCE,
                bootstrap=dict(resamples=2000, seed=42, unit="panel", conditional_on_noise=True),
                position_coverage=coverage, random_seeds=seeds,
                input_hashes={c["cell_id"]: {"input_ids": digest(c["input_ids"]),
                    "prompt": digest(c["rendered_prompt"])} for p in panels for c in p["cells"]})


def norm_error(actual, clean):
    a, c = actual.float(), clean.float()
    if not torch.isfinite(a).all() or not torch.isfinite(c).all():
        raise ValueError("nonfinite residual")
    an, cn = torch.linalg.vector_norm(a), torch.linalg.vector_norm(c)
    if not torch.isfinite(an) or not torch.isfinite(cn):
        raise ValueError("nonfinite residual norm")
    error = float(abs(an - cn) / cn) if cn != 0 else (0. if an == 0 else float("inf"))
    if error > NORM_TOLERANCE:
        raise ValueError("random replacement norm error exceeds 0.5%")
    return error


def random_row(clean, seed):
    clean = clean.detach().cpu()
    norm_error(clean, clean)
    norm = torch.linalg.vector_norm(clean.float())
    if norm == 0:
        return torch.zeros_like(clean)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    z = torch.randn(clean.shape, generator=generator, dtype=torch.float32)
    zn = torch.linalg.vector_norm(z)
    if not torch.isfinite(zn) or zn == 0:
        raise ValueError("invalid Gaussian direction")
    result = (z / zn * norm).to(clean.dtype)
    norm_error(result, clean)
    return result


def load_capture(ref):
    if file_digest(Path(ref["path"])) != ref["sha256"]:
        raise ValueError("capture checksum mismatch")
    return torch.load(ref["path"], map_location="cpu", weights_only=True)


def create_bank(manifest, baseline, output):
    from filler.dsv4.campaign_hook import atomic_torch_save
    clean_ref = baseline["clean_capture"]
    cid = baseline["target_id"]
    seeds = manifest["random_seeds"][cid]
    positions = sorted(map(int, seeds["0"]))
    clean = load_capture(clean_ref["ranks"]["0"])
    if clean["metadata"]["positions"] != list(range(clean["metadata"]["num_tokens"])):
        raise ValueError("random bank requires full clean prompt")
    # Identical norms and directions must hold for every rank, not only rank 0.
    for rank in ("1", "2", "3"):
        other = load_capture(clean_ref["ranks"][rank])
        for lid in range(43):
            if not torch.equal(clean["states"][lid][positions], other["states"][lid][positions]):
                raise ValueError("clean filler residuals differ across GPU ranks")
    states = {l: torch.stack([random_row(clean["states"][l][p], seeds[str(l)][str(p)])
                              for p in positions]) for l in range(43)}
    metadata = dict(algorithm=ALGORITHM, seed=manifest["seed"], seeds=seeds,
        cell_id=cid, config_hash=manifest["config_hash"], runtime_id=baseline["runtime_id"],
        baseline_id=baseline["record_id"], clean_capture=clean_ref, positions=positions,
        input_hashes=manifest["input_hashes"][cid], torch_version=str(torch.__version__))
    checksum = atomic_torch_save(output, dict(metadata=metadata, states=states))
    return dict(path=str(output), sha256=checksum, baseline_id=baseline["record_id"])


def validate_bank(ref, manifest, baseline):
    bank = load_capture(ref)
    meta = bank["metadata"]
    cid = baseline["target_id"]
    expected = dict(algorithm=ALGORITHM, seed=manifest["seed"], seeds=manifest["random_seeds"][cid],
        cell_id=cid, config_hash=manifest["config_hash"], runtime_id=baseline["runtime_id"],
        baseline_id=baseline["record_id"], clean_capture=baseline["clean_capture"],
        input_hashes=manifest["input_hashes"][cid])
    if any(meta.get(k) != v for k, v in expected.items()) or ref["baseline_id"] != baseline["record_id"]:
        raise ValueError("random bank provenance mismatch")
    positions = sorted(map(int, expected["seeds"]["0"]))
    if meta["positions"] != positions or set(bank["states"]) != set(range(43)):
        raise ValueError("random bank position/layer coverage mismatch")
    for rank in range(4):
        clean = load_capture(baseline["clean_capture"]["ranks"][str(rank)])
        for lid in range(43):
            if bank["states"][lid].shape != clean["states"][lid][positions].shape:
                raise ValueError("random bank tensor shape mismatch")
            for i, p in enumerate(positions):
                actual = bank["states"][lid][i]
                norm_error(actual, clean["states"][lid][p])
                if rank == 0:
                    seed = stream_seed(manifest["seed"], cid, lid, p)
                    if meta["seeds"][str(lid)][str(p)] != seed:
                        raise ValueError("random stream seed mismatch")
                    expected_row = random_row(clean["states"][lid][p], seed)
                    if actual.dtype != expected_row.dtype or not torch.equal(actual, expected_row):
                        raise ValueError("random bank differs from seeded reconstruction")
    return bank
