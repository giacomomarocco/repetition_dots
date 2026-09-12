"""Small CPU checks for the square release and four-stream readout contract."""
from dataclasses import replace
import io
import json
from pathlib import Path

import pytest
import torch

from filler.dsv4.jlens import (
    load_workspace_jlens, mean_mhc_streams, project_jlens_logits,
    published_readout_fixture, transport, unembed_transported,
)
from filler.dsv4 import workspace_jlens_artifact as artifact
from filler.dsv4.lens import DeepseekV4LensWeights


def save_square(path, mutate=lambda _: None):
    # Match the inspected public ZIP header: integer-keyed dict, no d_source.
    saved = {
        "J": {0: torch.tensor([[1., 2., 0.], [0., 1., 3.], [4., 0., 1.]]).half(),
              1: torch.eye(3, dtype=torch.float16)[[2, 0, 1]],
              2: torch.eye(3, dtype=torch.float16)},
        "n_prompts": 25, "source_layers": [0, 1, 2], "d_model": 3,
        "provenance": {"model_id": "deepseek-ai/DeepSeek-V4-Flash",
                       "target_layer": 2, "skip_first": 4, "n_prompts": 25,
                       "dataset_id": "NeelNanda/pile-10k", "t_max": 128,
                       "config_json": '{"estimator": "standard", "arm": "std"}'},
    }
    mutate(saved)
    torch.save(saved, path)


def load_tiny(path, **kwargs):
    return load_workspace_jlens(path, expected_d_model=3, expected_layers=(0, 1, 2), **kwargs)


@pytest.fixture
def square(tmp_path):
    path = tmp_path / "lens.pt"
    save_square(path)
    return load_tiny(path)


def readout(dtype=torch.float32):
    # Poison all mHC head parameters: no learned collapse may be used here.
    return DeepseekV4LensWeights(
        torch.full((4, 12), float("nan")), torch.full((4,), float("nan")),
        torch.full((1,), float("nan")), torch.tensor([1., 2., 3.]).to(dtype),
        torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [-1., 2., 0.]]).to(dtype),
    )


def test_mean_precedes_square_matrix_with_correct_orientation(square):
    states = torch.tensor([[0., 1., 2.], [2., 3., 4.], [4., 5., 6.], [6., 7., 8.]])
    # Mean [3,4,5], J rows [1,2,0], [0,1,3], [4,0,1].
    torch.testing.assert_close(transport(states, square, 0), torch.tensor([11., 19., 17.]))
    assert square.d_source == square.d_model == 3
    assert square.source_layers == (0, 1, 2) and square.target_layer == 2
    assert square.stream_reduction == "mean" and square.n_prompts == 25
    with pytest.raises(ValueError, match="no fitted Jacobian"):
        transport(states, square, 42)


@pytest.mark.parametrize("shape", [(4, 3), (7, 4, 3), (4, 4, 3), (2, 4, 4, 3)])
def test_stream_axis_batch_four_flattened_and_noncontiguous_inputs(square, shape):
    states = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape)
    expected = torch.stack([
        torch.tensor([sum(float(row[s, d]) for s in range(4)) / 4 for d in range(3)])
        for row in states.reshape(-1, 4, 3)
    ]).reshape(*shape[:-2], 3)
    for inputs in (states, states.flatten(-2), states.transpose(-1, -2).contiguous().transpose(-1, -2)):
        actual = transport(inputs, square, square.target_layer)
        assert actual.shape == expected.shape
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Stream permutation must not change the result.
    torch.testing.assert_close(transport(states[..., [3, 1, 0, 2], :], square, 0), transport(states, square, 0))


def test_mean_is_accumulated_in_fp32_before_bf16_rounding(square):
    states = torch.ones(4, 3, dtype=torch.bfloat16)
    states[1, 0] = 1.0078125
    result = mean_mhc_streams(states, hidden_size=3)
    assert result.dtype == torch.float32
    assert result[0].item() == 1.001953125
    assert result[0].item() != states.mean(-2)[0].item()
    torch.testing.assert_close(transport(states, square, 2), result, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_square_readout_matches_existing_hf_oracle_without_mhc(square, dtype, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "ports/jacobian-lens-open-frontier"))
    weights = readout(dtype)
    generator = torch.Generator().manual_seed(34)
    states = torch.randn(2, 4, 4, 3, generator=generator).to(dtype)
    mean = sum(states[..., stream, :].float() for stream in range(4)) / 4
    projected = torch.einsum("ij,...j->...i", square.jacobians[0].float(), mean)
    reference = published_readout_fixture(weights)
    expected = reference.unembed(projected, collapse=False).float()
    actual = project_jlens_logits(states, square, 0, weights)
    assert actual.shape == (2, 4, 4)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    # The identity anchor matches mean-only readout, not the native learned head.
    torch.testing.assert_close(project_jlens_logits(states, square, 2, weights),
                               unembed_transported(mean, weights), rtol=0, atol=0)


@pytest.mark.parametrize("states", [torch.zeros(3), torch.zeros(2, 3), torch.zeros(4, 4),
                                    torch.tensor(1.), torch.zeros(4, 3, dtype=torch.long),
                                    torch.full((4, 3), float("inf"))])
def test_invalid_raw_residuals_fail(square, states):
    with pytest.raises(ValueError):
        transport(states, square, 0)


def test_mismatched_readout_dimensions_and_reduction_fail(square):
    with pytest.raises(ValueError, match="dimensions disagree"):
        project_jlens_logits(torch.ones(4, 3), replace(square, hc_mult=2), 0, readout())
    with pytest.raises(ValueError, match="square Jacobian"):
        transport(torch.ones(4, 3), replace(square, d_source=12), 0)
    with pytest.raises(ValueError, match="unknown stream reduction"):
        transport(torch.ones(4, 3), replace(square, stream_reduction="sum"), 0)


@pytest.mark.parametrize("mutate,match", [
    (lambda v: v.pop("provenance"), "square J-Lens"),
    (lambda v: v.update(d_model=4), "dimensions"),
    (lambda v: v.update(d_source=12), "dimensions"),
    (lambda v: v.update(n_prompts=1000), "prompt count"),
    (lambda v: v.update(source_layers=[1, 2, 3]), "layer keys"),
    (lambda v: v["J"].pop(0), "layer keys"),
    (lambda v: v["J"].update({0: torch.eye(3)}), "fp16"),
    (lambda v: v["J"].update({0: torch.zeros(3, 12).half()}), "shape"),
    (lambda v: v["J"].update({0: torch.zeros(3, 3).half()}), "identically zero"),
    (lambda v: v["J"].update({0: torch.full((3, 3), float("nan")).half()}), "nonfinite"),
    (lambda v: v["J"][2].fill_(1), "exactly identity"),
    (lambda v: v["provenance"].update(target_layer=1), "provenance"),
    (lambda v: v["provenance"].update(model_id="another-model"), "provenance"),
    (lambda v: v["provenance"].update(skip_first=0), "provenance"),
    (lambda v: v["provenance"].update(config_json='{"estimator":"relp","arm":"all-c4"}'), "standard J-Lens"),
])
def test_incompatible_or_corrupt_release_is_rejected(tmp_path, mutate, match):
    path = tmp_path / "lens.pt"
    save_square(path, mutate)
    with pytest.raises(ValueError, match=match):
        load_tiny(path)


def test_metadata_preflight_defers_value_checks_only(tmp_path):
    path = tmp_path / "lens.pt"
    save_square(path, lambda v: v["J"][0].fill_(float("nan")))
    assert load_tiny(path, validate_values=False).source_layers == (0, 1, 2)
    with pytest.raises(ValueError, match="nonfinite"):
        load_tiny(path)
    save_square(path, lambda v: v.update(d_model=12))
    with pytest.raises(ValueError, match="dimensions"):
        load_tiny(path, validate_values=False)


def test_pinned_download_resume_checksum_and_existing_file_safety(tmp_path, monkeypatch):
    payload = b"synthetic tensor artifact"
    monkeypatch.setattr(artifact, "SIZE", len(payload))
    monkeypatch.setattr(artifact, "SHA256", artifact.hashlib.sha256(payload).hexdigest())
    path = tmp_path / "lens.pt"
    partial = tmp_path / "lens.pt.partial"
    partial.write_bytes(payload[:5])
    requests = []

    def open_url(request, timeout):
        requests.append(request)
        response = io.BytesIO(payload[5:])
        response.status = 206
        response.headers = {"Content-Range": f"bytes 5-{len(payload)-1}/{len(payload)}"}
        return response

    monkeypatch.setattr(artifact.urllib.request, "urlopen", open_url)
    assert artifact.prepare_artifact(path) == path
    assert path.read_bytes() == payload
    assert requests[0].get_header("Range") == "bytes=5-"
    assert artifact.REVISION in requests[0].full_url
    assert artifact.prepare_artifact(path) == path and len(requests) == 1
    assert json.loads((tmp_path / "SOURCE.json").read_text())["stream_reduction"] == "mean"
    path.write_bytes(b"x" * len(payload))
    with pytest.raises(ValueError, match="SHA-256"):
        artifact.prepare_artifact(path)
    assert path.read_bytes() == b"x" * len(payload)


def test_download_refuses_unhonored_resume_and_corrupt_partial(tmp_path, monkeypatch):
    payload = b"payload"
    monkeypatch.setattr(artifact, "SIZE", len(payload))
    monkeypatch.setattr(artifact, "SHA256", artifact.hashlib.sha256(payload).hexdigest())
    path = tmp_path / "lens.pt"
    partial = tmp_path / "lens.pt.partial"
    partial.write_bytes(b"pay")
    response = io.BytesIO(payload)
    response.status, response.headers = 200, {}
    monkeypatch.setattr(artifact.urllib.request, "urlopen", lambda *_a, **_k: response)
    with pytest.raises(ValueError, match="resume"):
        artifact.prepare_artifact(path)
    assert not path.exists() and partial.read_bytes() == b"pay"
    partial.write_bytes(b"xxxxxxx")
    with pytest.raises(ValueError, match="SHA-256"):
        artifact.prepare_artifact(path)
    assert not path.exists()
