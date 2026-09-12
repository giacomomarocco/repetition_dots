"""Frozen-helper bootstrap for scoring existing activations with both lenses."""
import importlib.util
from pathlib import Path
import sys

# An unrelated campaign edits patching.py frequently. Import the exact frozen
# version used by this scoring campaign before any numerical/runtime imports.
_frozen = Path(__file__).resolve().parents[2] / 'runs/deepseek-v4-flash/five-shot-jlens-heatmap/source/patching.py'
if not _frozen.is_file():
    raise RuntimeError('Missing frozen scoring helper; prepare its source snapshot first')
_spec = importlib.util.spec_from_file_location('filler.dsv4.patching', _frozen)
_module = importlib.util.module_from_spec(_spec)
sys.modules['filler.dsv4.patching'] = _module
_spec.loader.exec_module(_module)

from filler.dsv4.five_shot_jlens_heatmap import main

if __name__ == '__main__':
    main()
