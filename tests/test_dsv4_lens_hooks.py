from pathlib import Path

import torch
import torch.nn as nn

from dsv4_lens_hooks import hook_spec, make_layer_capture_hook
from sglang.srt.model_executor.hook_manager import register_forward_hooks


def _run_pass(hook, layers=3):
    for layer in range(layers):
        hidden = torch.full((2, 4, 5), float(layer))
        hook(None, (), (hidden, None, None, None))


def test_first_pass_only_then_trigger(tmp_path: Path):
    trigger = tmp_path / "capture-next"
    hook = make_layer_capture_hook(
        {
            "output_dir": str(tmp_path),
            "trigger_file": str(trigger),
            "num_layers": 3,
        }
    )
    _run_pass(hook)
    first = torch.load(tmp_path / "rank0" / "pass00000.pt", weights_only=True)
    assert first["metadata"]["position"] == -1
    assert set(first["states"]) == {0, 1, 2}
    assert first["states"][2].shape == (4, 5)

    _run_pass(hook)
    assert not (tmp_path / "rank0" / "pass00001.pt").exists()

    trigger.touch()
    _run_pass(hook)
    assert (tmp_path / "rank0" / "pass00002.pt").is_file()
    _run_pass(hook)
    assert not (tmp_path / "rank0" / "pass00003.pt").exists()


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Linear(2, 2)
        self.mlp = nn.Sequential(nn.Linear(2, 2), nn.ReLU())

    def forward(self, hidden):
        return hidden, None, None, None


class _Core(nn.Module):
    def __init__(self, layers: int):
        super().__init__()
        self.layers = nn.ModuleList([_Block() for _ in range(layers)])


class _CausalLM(nn.Module):
    def __init__(self, layers: int):
        super().__init__()
        self.model = _Core(layers)


def test_real_sglang_registration_matches_exact_blocks_only(tmp_path: Path):
    model = _CausalLM(43)
    register_forward_hooks(model, hook_spec(tmp_path))
    modules = dict(model.named_modules())
    matched = [name for name, module in modules.items() if module._forward_hooks]
    assert matched == [f"model.layers.{i}" for i in range(43)]
    assert len(matched) == 43
    hidden = torch.zeros(2, 4, 5)
    for layer in model.model.layers:
        layer(hidden)
    capture = torch.load(tmp_path / "rank0" / "pass00000.pt", weights_only=True)
    assert list(capture["states"]) == list(range(43))
    assert all(state.shape == (4, 5) for state in capture["states"].values())
