"""Delegate to the pinned port, installing timing before its imports/load."""
import importlib.machinery
import os
from pathlib import Path
import runpy
import sys
import traceback

try:
    from filler.dsv4.startup_timing import install_jit, install_model
    install_jit()
    here = Path(__file__).resolve().parent
    search = [p for p in sys.path if Path(p or ".").resolve() != here]
    spec = importlib.machinery.PathFinder.find_spec("sitecustomize", search)
    if spec is None or not spec.origin:
        raise RuntimeError("A100 port sitecustomize is missing")
    runpy.run_path(spec.origin)
    install_model()
except BaseException:
    traceback.print_exc()
    # Python normally swallows sitecustomize errors; an uninstrumented expensive
    # load would invalidate the experiment, so fail before server import.
    os._exit(1)
