import json
import os

import torch

from filler.dsv4.hooks import make_layer_transplant_hook


class Layer:
    use_fused_mhc_post_pre = False


def test_triggered_transplant_changes_only_selected_layer_and_last_token(tmp_path):
    capture_root = tmp_path / "captures"
    rank_dir = capture_root / "rank0"
    rank_dir.mkdir(parents=True)
    torch.save({"states": {1: torch.full((2, 3), 7.0)}}, rank_dir / "pass00012.pt")
    control = tmp_path / "TRANSPLANT_NEXT.json"
    control.write_text(json.dumps({"request_id": "trial", "capture_root": str(capture_root),
                                   "pass_id": 12, "layer": 1}))
    os.utime(control, ns=(100, 100))
    hook = make_layer_transplant_hook({"control_file": str(control),
                                      "ack_dir": str(tmp_path / "acks"), "num_layers": 3})
    outputs = []
    for _ in range(3):
        outputs.append(hook(Layer(), (), (torch.zeros(4, 2, 3), "tail")))
    assert torch.equal(outputs[0][0], torch.zeros(4, 2, 3))
    assert torch.equal(outputs[1][0][:-1], torch.zeros(3, 2, 3))
    assert torch.equal(outputs[1][0][-1], torch.full((2, 3), 7.0))
    assert outputs[1][1] == "tail"
    assert torch.equal(outputs[2][0], torch.zeros(4, 2, 3))
    assert (tmp_path / "acks" / "trial.rank0.json").is_file()


def test_trigger_is_consumed_once(tmp_path):
    capture_root = tmp_path / "captures" / "rank0"
    capture_root.mkdir(parents=True)
    torch.save({"states": {0: torch.ones(2, 3)}}, capture_root / "pass00001.pt")
    control = tmp_path / "control.json"
    control.write_text(json.dumps({"request_id": "once", "capture_root": str(tmp_path / "captures"),
                                   "pass_id": 1, "layer": 0}))
    os.utime(control, ns=(100, 100))
    hook = make_layer_transplant_hook({"control_file": str(control), "num_layers": 2})
    first = [hook(Layer(), (), torch.zeros(1, 2, 3)) for _ in range(2)]
    second = [hook(Layer(), (), torch.zeros(1, 2, 3)) for _ in range(2)]
    assert first[0].sum() == 6
    assert second[0].sum() == 0
