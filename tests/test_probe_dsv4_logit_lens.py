from pathlib import Path

import pytest

from probe_dsv4_logit_lens import arm_capture, capture_names, discover_capture


def _touch_capture(root: Path, pass_id: int, ranks=range(4)) -> None:
    for rank in ranks:
        path = root / f"rank{rank}" / f"pass{pass_id:05d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def test_validation_discovers_new_pass_after_startup_forwards(tmp_path: Path):
    _touch_capture(tmp_path, 0)
    _touch_capture(tmp_path, 3)
    before = capture_names(tmp_path)
    first_stamp = arm_capture(tmp_path / "CAPTURE_NEXT")
    _touch_capture(tmp_path, 17)

    assert discover_capture(tmp_path, before) == 17
    assert arm_capture(tmp_path / "CAPTURE_NEXT") > first_stamp


def test_validation_rejects_incomplete_tp_capture(tmp_path: Path):
    before = capture_names(tmp_path)
    _touch_capture(tmp_path, 8, ranks=range(3))
    with pytest.raises(RuntimeError, match="missing ranks=\\[3\\]"):
        discover_capture(tmp_path, before)
