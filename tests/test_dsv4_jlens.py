from pathlib import Path
import json
import math
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ports/jacobian-lens-open-frontier"))

from jlens.lens import JacobianLens
from filler.dsv4.jlens import (
    JLens, load_jlens, project_jlens_logits, published_readout_fixture, transport,
    unembed_transported,
)
from filler.dsv4.jlens_validation import compare_readouts, export_results, read_examples, token_scores, validate_paris
from filler.dsv4.lens import DeepseekV4LensWeights, project_logits


def fixture(dtype=torch.float32):
    generator = torch.Generator().manual_seed(17)
    def rand(*shape):
        return torch.randn(*shape, generator=generator).to(dtype)
    weights = DeepseekV4LensWeights(rand(4, 20), rand(4), rand(1), rand(5), rand(31, 5))
    lens = JLens({layer: rand(5, 20).half() for layer in (19, 30)}, 1000, 5, 20)
    return lens, weights, rand(4, 4, 5)


def save_lens(path, mutate=lambda value: None):
    lens, _, _ = fixture()
    saved = {"J": lens.jacobians, "n_prompts": 1000, "d_model": 5, "d_source": 20,
             "source_layers": [19, 30]}
    mutate(saved)
    torch.save(saved, path)


def load_tiny(path):
    return load_jlens(path, expected_shape=(5, 20), expected_layers=(19, 30))


def test_published_format_load_and_original_transport(tmp_path):
    path = tmp_path / "lens.pt"
    save_lens(path)
    lens = load_tiny(path)
    original = JacobianLens.load(str(path))
    _, _, states = fixture()
    assert lens.n_prompts == 1000 and lens.source_layers == (19, 30)
    assert lens.jacobians[19].dtype == torch.float16
    for inputs in (states, states.flatten(-2), states[0], states[0].flatten(), states.unsqueeze(0)):
        expected = original.transport(inputs.float(), 19)
        torch.testing.assert_close(transport(inputs, lens, 19), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mutate,match", [
    (lambda value: value.pop("d_source"), "rectangular"),
    (lambda value: value.update(d_source=5), "dimensions"),
    (lambda value: value.update(n_prompts=100), "prompt count"),
    (lambda value: value.update(source_layers=[20, 31]), "layer keys"),
    (lambda value: value["J"].update({19: torch.zeros(5, 20, dtype=torch.float16)}), "identically zero"),
    (lambda value: value["J"].update({19: torch.full((5, 20), float("nan"), dtype=torch.float16)}), "nonfinite"),
    (lambda value: value["J"].update({19: torch.ones(5, 20)}), "fp16"),
    (lambda value: value["J"].update({19: torch.ones(20, 5, dtype=torch.float16)}), "shape"),
])
def test_incompatible_checkpoint_rejected(tmp_path, mutate, match):
    path = tmp_path / "lens.pt"
    save_lens(path, mutate)
    with pytest.raises(ValueError, match=match):
        load_tiny(path)


@pytest.mark.parametrize("states", [torch.zeros(3, 5), torch.zeros(2, 10), torch.tensor(1.), torch.zeros(20, dtype=torch.int32)])
def test_invalid_residual_shape_or_dtype_rejected(states):
    lens, _, _ = fixture()
    with pytest.raises(ValueError):
        transport(states, lens, 19)


def test_missing_layer_and_nonfinite_input_fail():
    lens, _, states = fixture()
    with pytest.raises(ValueError, match="no fitted Jacobian"):
        transport(states, lens, 42)
    states[0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="nonfinite"):
        transport(states, lens, 19)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_complete_readout_matches_published_adapter_and_real_hf_norm(dtype):
    lens, weights, states = fixture(dtype)
    original = JacobianLens(lens.jacobians, n_prompts=1000, d_model=5, d_source=20)
    reference = published_readout_fixture(weights)
    transported = original.transport(states.float(), 19)
    # Batch size four must remain four transported vectors, without a second collapse.
    assert transported.shape == (4, 5)
    expected = reference.unembed(transported, collapse=False).float()
    actual = project_jlens_logits(states, lens, 19, weights)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    # This shape is ambiguous to a sniffing adapter; explicitly disable collapse.
    nested = transported.unsqueeze(0)
    torch.testing.assert_close(unembed_transported(nested, weights), reference.unembed(nested, collapse=False).float())
    if dtype == torch.bfloat16:
        value = transported.bfloat16().float()
        fused = (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + weights.norm_eps)
                 * weights.norm_weight.float()).bfloat16()
        fused_logits = torch.nn.functional.linear(fused, weights.lm_head_weight).float()
        assert not torch.equal(actual, fused_logits), "fixture must detect HF versus fused norm rounding"


def test_checkpoint_readout_width_mismatch():
    lens, weights, states = fixture()
    wrong = JLens(lens.jacobians, 1000, 5, 10)
    with pytest.raises(ValueError, match="dimensions disagree"):
        project_jlens_logits(states, wrong, 19, weights)


class Tokenizer:
    def decode(self, ids):
        return f"token{ids[0]}"


def test_target_scores_include_ties_and_exact_odds():
    logits = torch.tensor([0., 2., 2., -1.])
    scores = token_scores(logits, {"target": 2}, Tokenizer())
    target = scores["targets"]["target"]
    assert target["rank"] == 1
    assert target["logprob"] == pytest.approx(2 - math.log(1 + 2 * math.exp(2) + math.exp(-1)))
    assert target["log_odds_vs_rest"] == pytest.approx(2 - math.log(1 + math.exp(2) + math.exp(-1)))
    assert len(scores["top10"]) == 4
    with pytest.raises(ValueError, match="outside vocabulary"):
        token_scores(logits, {"target": -1}, Tokenizer())


def test_native_gate_rejects_hidden_or_logprob_mismatch():
    _, weights, states = fixture()
    state = states[0]
    lp = project_logits(state, weights).log_softmax(-1)
    top = lp.topk(10)
    native = {"output_ids": [int(lp.argmax())], "meta_info": {
        "hidden_states": [state.flatten().tolist()],
        "output_top_logprobs": [[[float(value), int(token)] for value, token in zip(top.values, top.indices)]]}}
    assert validate_paris(state, native, weights)["passed"]
    native["meta_info"]["hidden_states"][0][0] += 1
    assert not validate_paris(state, native, weights)["passed"]
    native["meta_info"]["hidden_states"][0] = state.flatten().tolist()
    native["meta_info"]["output_top_logprobs"][0][0][0] += 0.3
    assert not validate_paris(state, native, weights)["passed"]


def test_native_topk_boundary_allows_only_exact_ties(monkeypatch):
    import filler.dsv4.jlens_validation as validation
    _, weights, states = fixture()
    logits = torch.tensor([20., 19., 18., 17., 16., 15., 14., 13., 12., 11., 11., 10.])
    monkeypatch.setattr(validation, "project_logits", lambda *_args: logits)
    logprobs = logits.log_softmax(-1)
    selected = set(logprobs.topk(10).indices.tolist())
    chosen = next(iter(selected & {9, 10}))
    other = 19 - chosen
    native_ids = sorted((selected - {chosen}) | {other})
    native = {"output_ids": [0], "meta_info": {
        "hidden_states": [states[0].flatten().tolist()],
        "output_top_logprobs": [[[float(logprobs[token]), token] for token in native_ids]]}}
    result = validate_paris(states[0], native, weights)
    assert result["passed"] and result["top10_overlap"] == 9
    assert result["boundary_tie_count"] == 2
    assert result["unmatched_native"][0]["exactly_at_boundary"]
    # An unequal boundary replacement must fail even if its logprob is accurate.
    native["meta_info"]["output_top_logprobs"][0][-1] = [float(logprobs[11]), 11]
    assert not validate_paris(states[0], native, weights)["passed"]


def test_small_end_to_end_comparison_and_exports(tmp_path):
    lens, weights, states = fixture(torch.bfloat16)
    labels = ["last_prompt_token", "last_question", *[f"filler_{i}" for i in range(10)], "answer_prompt"]
    examples = [{"case": "paris" if i == 0 else "one_fact", "position_label": label,
                 "absolute_position": i, "pass_id": i, "native_token_id": 1,
                 "target_token_ids": {"Paris": 2} if i == 0 else {"A": 2, "X": 3, "A+X": 4}}
                for i, label in enumerate(labels)]
    captures = [{layer: states[i % 4] for layer in lens.source_layers} for i in range(len(examples))]
    counts = []
    rows, checks = compare_readouts(examples, captures, lens, weights, Tokenizer(),
                                   on_layer=lambda rows, checks: counts.append(len(rows)))
    assert counts == [13, 26]
    assert all(check["passed"] for check in checks)
    assert len(rows) == 26
    assert {row["position_label"] for row in rows} == set(labels)
    assert any(row["jlens"]["targets"] != row["ordinary"]["targets"] for row in rows)
    export_results(tmp_path, rows, {"vocab_size": 31, "started_at": "test"})
    assert (tmp_path / "target_ranks.png").stat().st_size > 1000
    assert (tmp_path / "target_ranks.pdf").stat().st_size > 1000
    assert "Paris" in (tmp_path / "REPORT.md").read_text()
    assert len((tmp_path / "target_scores.csv").read_text().splitlines()) == 149


def test_saved_manifest_labels_are_normalized_and_response_matching_is_enforced(tmp_path):
    capture_root = tmp_path / "captures"
    labels = ["last_question", *[f"filler_{i}" for i in range(10)], "answer_prompt"]
    entries = []
    shared = torch.zeros(4, 4096, dtype=torch.bfloat16)
    # Reuse storage: this tests the real on-disk schema without large fixtures.
    for pass_id in [0, *range(26, 50, 2)]:
        for rank in range(4):
            directory = capture_root / f"rank{rank}"
            directory.mkdir(exist_ok=True, parents=True)
            torch.save({"metadata": {"pass": pass_id, "rank": rank, "position": -1},
                        "states": {layer: shared for layer in range(43)}}, directory / f"pass{pass_id:05d}.pt")
        response = tmp_path / f"response{pass_id}.json"
        response.write_text(json.dumps({"output_ids": [1]}))
        if pass_id:
            entries.append({"label": labels[len(entries)], "pass_id": pass_id,
                            "absolute_position": pass_id, "generated_token_id": 1,
                            "response": str(response)})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"captures": entries, "target_values": {"A": 57, "X": 11, "A+X": 68},
                                    "target_token_ids": {"A": 3351, "X": 779, "A+X": 2973}}))
    examples, captures, native, hashes = read_examples(capture_root, manifest, tmp_path / "response0.json")
    assert [row["position_label"] for row in examples] == ["last_prompt_token", *labels]
    assert len(captures) == 13 and len(hashes) == 66
    assert all(row["native_token_id"] == 1 for row in examples)
    assert all(row["case"] == "one_fact" for row in examples[1:])
    (tmp_path / "response26.json").write_text(json.dumps({"output_ids": [2]}))
    with pytest.raises(ValueError, match="response does not match"):
        read_examples(capture_root, manifest, tmp_path / "response0.json")
