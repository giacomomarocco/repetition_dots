"""Validated TVM-FFI bundles; active builds always stay on node-local /tmp.

No pickle, archive extraction, kernel substitution, or prepared model weights.
Bundles are private executable artifacts from this user's validated runs.
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import socket
import subprocess
import sys
import uuid

WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = WORKSPACE / "runs/deepseek-v4-flash/startup-cache"
SCHEMA = 1
SOURCE_SUFFIXES = {".py", ".h", ".hpp", ".cuh", ".cu", ".c", ".cc", ".cpp", ".inl"}


def kernel_environment():
    # Only variable names referenced by the bounded local kernel implementation
    # are relevant. Exclude credentials even if supplied under a familiar prefix.
    import re
    roots = [WORKSPACE / "ports/deepseek-v4-a100-sglang/dsv4_a100_patch",
             WORKSPACE / "ports/sglang/python/sglang/jit_kernel"]
    names = set()
    for root in roots:
        for path in root.rglob("*.py"):
            names.update(re.findall(r"\b(?:SGLANG|TVM_FFI)_[A-Z0-9_]+\b", path.read_text()))
    return {k: os.environ.get(k) for k in sorted(names)
            if not any(s in k for s in ("TOKEN", "SECRET", "PASSWORD", "API_KEY"))
            and k not in {"TVM_FFI_CACHE_DIR", "SGLANG_PORT"}}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temp.open("w") as out:
        json.dump(value, out, indent=2, sort_keys=True)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    os.replace(temp, path)


def local_base():
    return Path(f"/tmp/dsv4-kernels-{os.getuid()}")


def source_identity(roots, runtime):
    files = {}
    for root in sorted(set(Path(p).resolve() for p in roots)):
        if not root.exists():
            raise ValueError(f"missing fingerprint input: {root}")
        paths = [root] if root.is_file() else sorted(root.rglob("*"))
        for path in paths:
            if path.is_file() and (root.is_file() or path.suffix in SOURCE_SUFFIXES):
                files[str(path)] = file_hash(path)
    if not files:
        raise ValueError("empty fingerprint inputs")
    # Canonical JSON types make a live identity equal to its persisted form
    # (e.g. platform.libc_ver() returns a tuple, which JSON stores as a list).
    return json.loads(json.dumps({"schema": SCHEMA, "sources": files, "runtime": runtime}))


def runtime_identity():
    """Run on the approved compute node, before the model process exists."""
    import torch
    import tvm_ffi
    from tvm_ffi.cpp import extension as ext
    from sglang.jit_kernel import utils

    cuda_home = Path(ext._find_cuda_home())
    cxx = shlex.split(os.environ.get("CXX", "c++"))

    def version(command):
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=30).strip()

    packages = {}
    for package in ("torch", "apache-tvm-ffi", "flashinfer-python", "triton", "ninja"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    runtime = {
        "python": sys.version, "executable": sys.executable, "machine": platform.machine(),
        "libc": platform.libc_ver(), "packages": packages, "torch_cuda": torch.version.cuda,
        "gpu_arch": sorted(set(str(torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count()))),
        "cxx": {"command": cxx, "path": shutil.which(cxx[0]), "version": version([*cxx, "--version"])},
        "nvcc": {"path": str(cuda_home / "bin/nvcc"), "version": version([str(cuda_home / "bin/nvcc"), "--version"])},
        "ninja": version(["ninja", "--version"]),
        "driver": sorted(set(version(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]).splitlines())),
        "cflags": utils.DEFAULT_CFLAGS, "cuda_flags": utils._get_default_target_flags(),
        "tvm_cuda_target": ext._get_cuda_target(), "ldflags": utils.DEFAULT_LDFLAGS,
        # Whitelist compilation/behavior inputs; never serialize arbitrary env.
        "env": kernel_environment(),
        "compiler_env": {k: os.environ.get(k) for k in (
            "CC", "CXX", "CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS", "CUDA_HOME", "CUDA_PATH",
            "TORCH_CUDA_ARCH_LIST", "TVM_FFI_CUDA_ARCH_LIST", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS",
            "CPATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH", "LD_LIBRARY_PATH")},
    }
    if not runtime["gpu_arch"]:
        raise RuntimeError("kernel fingerprint requires a visible GPU on an approved compute node")
    roots = [WORKSPACE / "ports/deepseek-v4-a100-sglang/dsv4_a100_patch",
             utils.KERNEL_PATH, Path(tvm_ffi.__file__).parent,
             *utils.get_cutlass_include_paths(), ext.find_include_path(), ext.find_dlpack_include_path(),
             ext.find_libtvm_ffi(), cuda_home / "include", Path(__file__),
             WORKSPACE / "filler/dsv4/startup_timing.py", WORKSPACE / "run_deepseek_v4_startup.sh"]
    return source_identity(roots, runtime)


@contextmanager
def locked(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("a") as lock:
        # Fail rather than waiting behind another active model or publication.
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield lock


def inventory(root):
    result = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlink in executable cache: {path}")
        if path.is_file():
            if path.name == "lock" or path.suffix == ".lock":
                continue
            if path.stat().st_uid != os.getuid() or path.stat().st_mode & 0o022:
                raise ValueError(f"cache artifact must be owned by user and not writable by others: {path}")
            result[str(path.relative_to(root))] = {"sha256": file_hash(path), "size": path.stat().st_size}
    return result


def validate_bundle(bundle, identity, local):
    bundle = Path(bundle)
    if bundle.is_symlink() or bundle.stat().st_uid != os.getuid() or bundle.stat().st_mode & 0o022:
        raise ValueError("untrusted bundle directory")
    meta = json.loads((bundle / "COMPLETE.json").read_text())
    if (meta.get("schema") != SCHEMA or meta.get("fingerprint") != digest(identity)
            or meta.get("identity") != identity or meta.get("local_path") != str(local)):
        raise ValueError("incompatible kernel bundle")
    files = inventory(bundle / "tvm")
    if not files or meta.get("files") != files or not any(p.endswith(".so") for p in files):
        raise ValueError("incomplete or corrupt kernel bundle")
    return meta


def copy_cache(source, target):
    # copy2 preserves timestamps needed by Ninja along with .ninja_deps/.ninja_log.
    shutil.copytree(source, target, copy_function=shutil.copy2,
                    ignore=lambda _dir, names: [n for n in names if n == "lock" or n.endswith(".lock")])


def preserve(path):
    """Keep interrupted or invalid work for diagnosis; never delete user caches."""
    if path.exists():
        path.rename(path.with_name(path.name + ".retired-" + uuid.uuid4().hex))


def restore(identity, root, local, *, fresh=False, force_restore=False):
    """Caller must hold local lease. An absent/corrupt bundle is a local rebuild."""
    root, local = Path(root), Path(local)
    key = digest(identity)
    meta_path = local.parent / "LOCAL.json"
    if not fresh and not force_restore and local.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, ValueError):
            meta = {}
        if meta == {"fingerprint": key, "state": "validated"}:
            # The last successful publication recorded the exact local inventory.
            try:
                validated = json.loads((local.parent / "VALIDATED.json").read_text())
                if validated == inventory(local):
                    return "same_node"
            except (OSError, ValueError):
                pass
    preserve(local)
    status = "empty"
    bundle = root / key
    if not fresh and bundle.exists():
        try:
            validate_bundle(bundle, identity, local)
            temp = local.with_name(".restore-" + uuid.uuid4().hex)
            copy_cache(bundle / "tvm", temp)
            if inventory(temp) != inventory(bundle / "tvm"):
                raise ValueError("bundle changed during restore")
            temp.rename(local)
            status = "restored"
        except (OSError, ValueError, KeyError) as exc:
            print(f"Rejecting kernel bundle: {exc}; rebuilding locally", file=sys.stderr)
    local.mkdir(parents=True, exist_ok=True, mode=0o700)
    atomic_json(meta_path, {"fingerprint": key, "state": "building"})
    return status


def publish(identity, root, local, validation):
    """After the owned server exits and equivalence/prompts have passed."""
    root, local = Path(root), Path(local)
    if validation.get("passed") is not True:
        raise ValueError("successful validation required to publish")
    key = digest(identity)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with locked(local.parent / "lease.lock"), locked(root / f".{key}.lock"), ExitStack() as module_locks:
        # Also honor TVM's own flock files, including compiler children that may
        # briefly outlive a terminating server. Never snapshot an active build.
        for recipe in sorted(local.rglob("build.ninja")):
            module_locks.enter_context(locked(recipe.parent / "lock"))
        files = inventory(local)
        if not any(p.endswith(".so") for p in files):
            raise ValueError("no completed kernel modules")
        # Each module must contain its Ninja recipe and compiled output.
        for recipe in local.rglob("build.ninja"):
            if not list(recipe.parent.glob("*.so")):
                raise ValueError(f"unfinished module: {recipe.parent.name}")
        target = root / key
        if target.exists():
            try:
                validate_bundle(target, identity, local)
            except (OSError, ValueError, KeyError):
                preserve(target)
        if not target.exists():
            stage = root / f".{key}.{uuid.uuid4().hex}.staging"
            stage.mkdir(mode=0o700)
            copy_cache(local, stage / "tvm")
            if files != inventory(stage / "tvm") or files != inventory(local):
                raise ValueError("cache changed during publication")
            atomic_json(stage / "COMPLETE.json", {
                "schema": SCHEMA, "fingerprint": key, "identity": identity,
                "local_path": str(local), "files": files, "validation": validation})
            # Directory rename is the sole visibility point for complete bundles.
            stage.rename(target)
        atomic_json(local.parent / "VALIDATED.json", files)
        atomic_json(local.parent / "LOCAL.json", {"fingerprint": key, "state": "validated"})
    return target


def launch(command):
    from filler.dsv4.startup_timing import span
    mode = os.environ.get("DSV4_KERNEL_CACHE_MODE", "off")
    if mode not in {"off", "reuse"}:
        raise ValueError("DSV4_KERNEL_CACHE_MODE must be off or reuse")
    env = dict(os.environ)
    # Honor the last --port argument, matching argparse's handling of overrides.
    port = None
    for i, arg in enumerate(command):
        if arg == "--port":
            port = int(command[i + 1])
        elif arg.startswith("--port="):
            port = int(arg.split("=", 1)[1])
    if port is not None:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))
    lock = None
    if mode == "reuse":
        os.umask(0o077)
        with span("cache.fingerprint"):
            identity = runtime_identity()
        key = digest(identity)
        # TMPDIR often includes the job ID: it cannot preserve absolute Ninja paths.
        base = local_base()
        base.mkdir(mode=0o700, exist_ok=True)
        if base.is_symlink() or base.stat().st_uid != os.getuid() or base.stat().st_mode & 0o077:
            raise ValueError("node-local cache must be a private user-owned directory")
        local = base / key / "tvm"
        local.parent.mkdir(exist_ok=True, mode=0o700)
        lock = (local.parent / "lease.lock").open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root = Path(env.get("DSV4_KERNEL_CACHE_ROOT", DEFAULT_ROOT)).expanduser().resolve()
        with span("cache.restore"):
            status = restore(identity, root, local, fresh=env.get("DSV4_KERNEL_CACHE_FRESH") == "1",
                             force_restore=env.get("DSV4_KERNEL_CACHE_RESTORE") == "1")
        # A crash after restoration must not qualify as a same-node validated hit.
        atomic_json(local.parent / "LOCAL.json", {"fingerprint": key, "state": "building"})
        env["TVM_FFI_CACHE_DIR"] = str(local)
        if env.get("DSV4_STARTUP_RUN_DIR"):
            atomic_json(Path(env["DSV4_STARTUP_RUN_DIR"]) / "cache.json", {
                "identity": identity, "fingerprint": key, "local": str(local),
                "root": str(root), "restore_status": status})
        os.set_inheritable(lock.fileno(), True)
        print(f"TVM-FFI cache: {status}, {key}, {local}", file=sys.stderr, flush=True)
    if env.get("DSV4_STARTUP_TIMING") == "1":
        env["PYTHONPATH"] = str(WORKSPACE / "filler/dsv4/startup_bootstrap") + os.pathsep + env.get("PYTHONPATH", "")
    os.execvpe(command[0], command, env)
