import json
import os

from scripts.dsv4.run_activation_patching import (
    arm_transplant,
    requested_logprobs,
    wait_for_acks,
)


def test_arm_transplant_is_atomic_and_advances_mtime(tmp_path):
    path = tmp_path / "control.json"
    arm_transplant(path, {"request_id": "a"})
    first = path.stat().st_mtime_ns
    arm_transplant(path, {"request_id": "b"})
    assert path.stat().st_mtime_ns > first
    assert json.loads(path.read_text()) == {"request_id": "b"}
    assert not list(tmp_path.glob("*.tmp.*"))


def test_wait_for_all_rank_acknowledgements(tmp_path):
    for rank in range(4):
        (tmp_path / f"trial.rank{rank}.json").write_text("{}")
    wait_for_acks(tmp_path, "trial", 4, timeout=0.01)


def test_requested_logprobs_names_multiple_scored_tokens():
    response = {"meta_info": {"output_token_ids_logprobs": [[
        [-1.25, 101, None], [-3.5, 202, None],
    ]]}}
    assert requested_logprobs(response) == {101: -1.25, 202: -3.5}
