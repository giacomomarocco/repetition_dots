"""Opt-in, per-process startup spans. Never synchronizes CUDA when disabled."""
from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import json
import os
from pathlib import Path
import resource
import sys
import time

_sinks = {}


def enabled():
    return os.environ.get("DSV4_STARTUP_TIMING") == "1"


def counters():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    result = {key: getattr(usage, key) for key in (
        "ru_utime", "ru_stime", "ru_minflt", "ru_majflt", "ru_inblock", "ru_oublock")}
    result.update(child_utime=children.ru_utime, child_stime=children.ru_stime)
    try:
        for line in Path("/proc/self/io").read_text().splitlines():
            key, value = line.split(":")
            result[key] = int(value)
    except OSError:
        pass
    return result


def rank():
    torch = sys.modules.get("torch")
    if torch is not None and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return os.environ.get("RANK", "controller")


def emit(event, **fields):
    if not enabled():
        return
    row = {"event": event, "pid": os.getpid(), "rank": rank(), "time": time.time(),
           "since_launch_s": time.time() - float(os.environ.get("DSV4_STARTUP_EPOCH", time.time())),
           "counters": counters(), **fields}
    line = json.dumps(row, sort_keys=True) + "\n"
    root = os.environ.get("DSV4_STARTUP_TIMING_DIR")
    if root:
        key = (os.getpid(), root)
        if key not in _sinks:
            Path(root).mkdir(parents=True, exist_ok=True)
            _sinks[key] = (Path(root) / f"{os.getpid()}.jsonl").open("a", buffering=1)
        _sinks[key].write(line)
    else:
        sys.stderr.write("DSV4_STARTUP " + line)


def synchronize():
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        torch.cuda.synchronize()


@contextmanager
def span(name, *, gpu=False, **fields):
    if not enabled():
        yield
        return
    if gpu:
        synchronize()
    start, before = time.perf_counter(), counters()
    emit(name + ".begin", **fields)
    outcome = "ok"
    try:
        yield
    except BaseException:
        outcome = "error"
        raise
    finally:
        if gpu and outcome == "ok":
            synchronize()
        elapsed = time.perf_counter() - start
        delta = {k: v - before.get(k, v) for k, v in counters().items()}
        emit(name + ".end", elapsed_s=elapsed, delta=delta,
             cpu_percent=100 * (delta["ru_utime"] + delta["ru_stime"]) / max(elapsed, 1e-9),
             outcome=outcome, **fields)


def wrap(owner, attribute, event, *, gpu=False):
    original = getattr(owner, attribute)
    if getattr(original, "_dsv4_timed", False):
        return

    @wraps(original)
    def call(*args, **kwargs):
        label = str(args[0]) if args and isinstance(args[0], (str, Path)) else getattr(args[0], "prefix", "") if args else ""
        with span(event, gpu=gpu, label=label):
            return original(*args, **kwargs)
    call._dsv4_timed = True
    setattr(owner, attribute, call)


def install_jit():
    if not enabled():
        return
    import tvm_ffi.cpp as cpp
    from tvm_ffi.cpp import extension
    for attr in ("load_inline", "load", "build_inline", "build"):
        wrap(cpp, attr, "jit." + attr)
        wrap(extension, attr, "jit." + attr)
    wrap(extension, "load_module", "jit.module_load")
    # Ninja is invoked on cache hits too. Changed log entries count actual build
    # edges (compilation/link), not Python load calls or lock waiters.
    original = extension.build_ninja

    def build(build_dir):
        log = Path(build_dir) / ".ninja_log"
        before = log.read_bytes() if log.exists() else b""
        with span("jit.ninja", module=Path(build_dir).name):
            result = original(build_dir)
        after = log.read_bytes() if log.exists() else b""
        changed = after[len(before):] if after.startswith(before) else after
        edges = [line.split(b"\t")[3].decode() for line in changed.splitlines()
                 if not line.startswith(b"#") and len(line.split(b"\t")) >= 5]
        emit("jit.build_edges", module=Path(build_dir).name, outputs=edges,
             compile_count=sum(p.endswith(".o") for p in edges), build_count=len(edges))
        return result
    extension.build_ninja = build


def install_model():
    if not enabled():
        return
    from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
    from sglang.srt.layers.quantization.mxfp4_marlin_moe import Mxfp4MarlinMoEMethod
    original = DeepseekV4ForCausalLM.load_weights

    @wraps(original)
    def load(self, weights, *args, **kwargs):
        # The pinned loader already materializes this iterator. This explicit
        # boundary measures enumeration separately from deferred mmap reads.
        with span("weights.enumeration"):
            weights = list(weights)
        with span("weights.assignment_and_transfer", gpu=True, tensors=len(weights)):
            return original(self, weights, *args, **kwargs)
    DeepseekV4ForCausalLM.load_weights = load
    wrap(DeepseekV4ForCausalLM, "post_load_weights", "weights.post_load", gpu=True)
    wrap(Mxfp4MarlinMoEMethod, "process_weights_after_loading", "experts.prepare", gpu=True)
    from dsv4_a100_patch import triton_kernels
    wrap(triton_kernels, "prepare_mxfp4_int8_moe", "experts.remap_and_prewarm", gpu=True)
    from dsv4_a100_patch.triton_kernels import mxfp4_int8_moe
    wrap(mxfp4_int8_moe, "remap_mxfp4_weight_for_int8", "experts.remap", gpu=True)
    wrap(mxfp4_int8_moe, "_try_prewarm_mxfp4_int8_moe_jit", "experts.prewarm", gpu=True)
    emit("instrumentation.installed")
