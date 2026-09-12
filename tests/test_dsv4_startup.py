import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from filler.dsv4 import startup_cache as cache
from filler.dsv4 import startup_timing as timing
from scripts.dsv4.benchmark_startup import compare, phase_sequence, source_changes, timing_summary


@pytest.fixture
def built(tmp_path):
    source = tmp_path / "kernel.cuh"
    source.write_text("original header")
    identity = cache.source_identity([source], {"arch": "sm80", "flags": ["-O3"], "compiler": "v1"})
    local = tmp_path / "node" / "tvm"
    module = local / "moe"
    module.mkdir(parents=True)
    for name, value in {"build.ninja": "recipe", "kernel.so": "binary", "kernel.o": "object",
                        ".ninja_deps": "dependencies", ".ninja_log": "log", "lock": ""}.items():
        path = module / name
        path.write_text(value)
        path.chmod(0o600)
    return identity, tmp_path / "bundles", local


def test_fingerprint_covers_contents_flags_arch_versions_and_paths(tmp_path):
    source = tmp_path / "kernel.cuh"
    source.write_text("v1")
    first = cache.source_identity([source], {"flags": ["-O3"], "arch": "sm80", "compiler": "v1"})
    for key, value in (("flags", ["-O2"]), ("arch", "sm90"), ("compiler", "v2")):
        assert cache.digest(cache.source_identity([source], {**first["runtime"], key: value})) != cache.digest(first)
    source.write_text("v2")
    assert cache.digest(cache.source_identity([source], first["runtime"])) != cache.digest(first)
    moved = tmp_path / "moved.cuh"
    moved.write_text("v1")
    assert cache.digest(cache.source_identity([moved], first["runtime"])) != cache.digest(first)


def test_runtime_identity_equals_persisted_json_types(tmp_path):
    source = tmp_path / "kernel.cuh"
    source.write_text("source")
    identity = cache.source_identity([source], {"libc": ("glibc", "2.31")})
    assert identity == json.loads(json.dumps(identity))


def test_publish_restore_preserves_ninja_dependencies_and_mtimes(built):
    identity, root, local = built
    before = (local / "moe/kernel.o").stat().st_mtime_ns
    bundle = cache.publish(identity, root, local, {"passed": True})
    assert cache.validate_bundle(bundle, identity, local)
    assert not (bundle / "tvm/moe/lock").exists()
    assert cache.restore(identity, root, local) == "same_node"
    cache.preserve(local)
    assert cache.restore(identity, root, local) == "restored"
    assert (local / "moe/kernel.o").stat().st_mtime_ns == before
    assert (local / "moe/.ninja_deps").read_text() == "dependencies"


def test_force_restore_exercises_bundle_on_same_node(built):
    identity, root, local = built
    cache.publish(identity, root, local, {"passed": True})
    assert cache.restore(identity, root, local, force_restore=True) == "restored"
    assert list(local.parent.glob("tvm.retired-*"))
    assert phase_sequence("sequential") == ("cold", "same", "restored")


def test_driver_updates_cannot_hide_inference_or_cache_source_changes():
    baseline = {"scripts/dsv4/benchmark_startup.py": "old", "filler/dsv4/startup_cache.py": "same"}
    current = {**baseline, "scripts/dsv4/benchmark_startup.py": "new"}
    with pytest.raises(ValueError):
        source_changes(baseline, current)
    assert source_changes(baseline, current, True) == ["scripts/dsv4/benchmark_startup.py"]
    with pytest.raises(ValueError):
        source_changes(baseline, {**current, "filler/dsv4/startup_cache.py": "new"}, True)


@pytest.mark.parametrize("damage", ["missing_marker", "corruption", "wrong_path", "wrong_identity", "symlink"])
def test_reject_invalid_bundle_and_rebuild_locally(built, damage):
    identity, root, local = built
    bundle = cache.publish(identity, root, local, {"passed": True})
    if damage == "missing_marker":
        (bundle / "COMPLETE.json").rename(bundle / "INCOMPLETE.json")
    elif damage == "corruption":
        (bundle / "tvm/moe/kernel.so").write_text("corrupt")
    elif damage in {"wrong_path", "wrong_identity"}:
        meta = json.loads((bundle / "COMPLETE.json").read_text())
        meta["local_path" if damage == "wrong_path" else "identity"] = "different"
        cache.atomic_json(bundle / "COMPLETE.json", meta)
    else:
        (bundle / "tvm/moe/symlink").symlink_to(bundle / "tvm/moe/kernel.so")
    cache.preserve(local)
    assert cache.restore(identity, root, local) == "empty"
    assert not list(local.iterdir())


def test_incomplete_local_cache_and_failed_validation_cannot_publish(built):
    identity, root, local = built
    with pytest.raises(ValueError, match="validation"):
        cache.publish(identity, root, local, {"passed": False})
    unfinished = local / "unfinished"
    unfinished.mkdir()
    (unfinished / "build.ninja").write_text("unfinished")
    with pytest.raises(ValueError, match="unfinished"):
        cache.publish(identity, root, local, {"passed": True})
    assert not (root / cache.digest(identity)).exists()


def test_active_local_lease_prevents_publication(built):
    identity, root, local = built
    with cache.locked(local.parent / "lease.lock"):
        with pytest.raises(BlockingIOError):
            cache.publish(identity, root, local, {"passed": True})


def test_active_tvm_build_prevents_publication(built):
    identity, root, local = built
    with cache.locked(local / "moe/lock"):
        with pytest.raises(BlockingIOError):
            cache.publish(identity, root, local, {"passed": True})


def test_concurrent_publications_expose_only_complete_bundle(built):
    identity, root, local = built

    def publish():
        try:
            return cache.publish(identity, root, local, {"passed": True})
        except BlockingIOError:
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: publish(), range(2)))
    assert any(results)
    assert cache.validate_bundle(root / cache.digest(identity), identity, local)
    assert cache.publish(identity, root, local, {"passed": True}) == root / cache.digest(identity)


def test_fresh_preserves_previous_cache_but_ignores_bundle(built):
    identity, root, local = built
    cache.publish(identity, root, local, {"passed": True})
    assert cache.restore(identity, root, local, fresh=True) == "empty"
    assert not list(local.iterdir())
    assert len(list(local.parent.glob("tvm.retired-*"))) == 1


def test_interrupted_local_build_cannot_be_a_same_node_hit(built):
    identity, root, local = built
    cache.publish(identity, root, local, {"passed": True})
    cache.atomic_json(local.parent / "LOCAL.json", {"fingerprint": cache.digest(identity), "state": "building"})
    assert cache.restore(identity, root, local) == "restored"


def test_cache_off_does_not_probe_gpu_or_change_arguments(monkeypatch):
    monkeypatch.setenv("DSV4_KERNEL_CACHE_MODE", "off")
    monkeypatch.delenv("DSV4_STARTUP_TIMING", raising=False)
    monkeypatch.setenv("TVM_FFI_CACHE_DIR", "/tmp/original")
    monkeypatch.setattr(cache, "runtime_identity", lambda: pytest.fail("off must not fingerprint"))
    seen = []
    monkeypatch.setattr(os, "execvpe", lambda *args: seen.append(args))
    command = ["python", "-m", "sglang.launch_server", "--forward-hooks", '{"literal": "$HOME"}']
    cache.launch(command)
    assert seen[0][1] == command
    assert seen[0][2]["TVM_FFI_CACHE_DIR"] == "/tmp/original"
    monkeypatch.setenv("DSV4_KERNEL_CACHE_MODE", "invalid")
    with pytest.raises(ValueError):
        cache.launch(command)


def test_reuse_environment_root_override_and_timing_overlay(tmp_path, monkeypatch):
    identity = {"schema": 1, "test": "identity"}
    monkeypatch.setattr(cache, "runtime_identity", lambda: identity)
    monkeypatch.setattr(cache, "local_base", lambda: tmp_path / "private")
    monkeypatch.setenv("DSV4_KERNEL_CACHE_MODE", "reuse")
    monkeypatch.setenv("DSV4_KERNEL_CACHE_ROOT", str(tmp_path / "override"))
    monkeypatch.setenv("DSV4_STARTUP_RUN_DIR", str(tmp_path / "run"))
    monkeypatch.setenv("DSV4_STARTUP_TIMING_DIR", str(tmp_path / "timing"))
    monkeypatch.setenv("DSV4_STARTUP_TIMING", "1")
    monkeypatch.setenv("PYTHONPATH", "original-path")
    seen = []
    monkeypatch.setattr(os, "execvpe", lambda *args: seen.append(args))
    mask = os.umask(0o077)
    try:
        cache.launch(["python", "-m", "sglang.launch_server"])
    finally:
        os.umask(mask)
    env = seen[0][2]
    assert env["TVM_FFI_CACHE_DIR"] == str(tmp_path / "private" / cache.digest(identity) / "tvm")
    assert env["PYTHONPATH"].endswith(os.pathsep + "original-path")
    assert "startup_bootstrap" in env["PYTHONPATH"]
    report = json.loads((tmp_path / "run/cache.json").read_text())
    assert report["root"] == str(tmp_path / "override")
    assert report["restore_status"] == "empty"


def test_bootstrap_delegates_port_between_jit_and_model_hooks(tmp_path):
    import textwrap
    workspace = Path(__file__).resolve().parents[1]
    fake_port = tmp_path / "port"
    fake_port.mkdir()
    (fake_port / "sitecustomize.py").write_text('import builtins\nbuiltins.events.append("port")\n')
    code = textwrap.dedent(f"""
        import builtins, runpy, sys, types
        builtins.events = []
        fake = types.ModuleType('filler.dsv4.startup_timing')
        fake.install_jit = lambda: builtins.events.append('jit')
        fake.install_model = lambda: builtins.events.append('model')
        sys.modules['filler.dsv4.startup_timing'] = fake
        sys.path.insert(0, {str(fake_port)!r})
        runpy.run_path({str(workspace / 'filler/dsv4/startup_bootstrap/sitecustomize.py')!r})
        assert builtins.events == ['jit', 'port', 'model'], builtins.events
    """)
    subprocess.run([sys.executable, "-S", "-c", code], check=True, timeout=15)


@pytest.mark.parametrize("profile", ["fast", "balanced", "throughput"])
def test_launcher_preserves_serving_arguments_hooks_and_internal_port(tmp_path, profile):
    workspace = Path(__file__).resolve().parents[1]
    activation = tmp_path / ".venv-sglang/bin"
    activation.mkdir(parents=True)
    (activation / "activate").write_text(f'export PATH="{activation}:$PATH"\n')
    fake = activation / "python"
    fake.write_text(f'#!{sys.executable} -S\nimport json, os, sys\nprint(json.dumps({{"args": sys.argv[1:], "port_env": os.environ.get("SGLANG_PORT"), "hooks_env": os.environ.get("SGLANG_OPT_FUSE_MHC_POST_PRE")}}))\n')
    fake.chmod(0o700)
    results = []
    hooks = '[{"hook_factory":"filler.dsv4.campaign_hook:make_campaign_hook"}]'
    for name in ("run_deepseek_v4_a100.sh", "run_deepseek_v4_startup.sh"):
        shutil.copy2(workspace / name, tmp_path / name)
        out = subprocess.check_output(["bash", str(tmp_path / name), "--enable-return-hidden-states", "--forward-hooks", hooks],
            env={**os.environ, "SGLANG_PORT": "30123", "DEEPSEEK_STARTUP_PROFILE": profile,
                 "TMPDIR": str(tmp_path / "tmp"), "DSV4_KERNEL_CACHE_MODE": "off"}, text=True)
        results.append(json.loads(out))
    old, new = results
    # The new launcher adds exactly the cache-manager entry point before exec.
    assert new["args"][:4] == ["-m", "scripts.dsv4.startup_cache", "--", "python"]
    assert new["args"][4:] == old["args"]
    assert new["port_env"] is None
    assert new["hooks_env"] == old["hooks_env"] == "0"
    # The pinned allocator must see an unset convenience port too.
    import ast
    network = ast.parse((workspace / "ports/sglang/python/sglang/srt/utils/network.py").read_text())
    function = next(n for n in network.body if isinstance(n, ast.FunctionDef) and n.name == "get_open_port")
    class Socket:
        def getsockname(self):
            return ("127.0.0.1", 49234)
        def close(self):
            pass
    from types import SimpleNamespace
    namespace = {"os": SimpleNamespace(getenv=lambda key: new["port_env"]),
                 "try_bind_socket": Socket,
                 "is_port_available": lambda port: pytest.fail("HTTP port leaked to allocator")}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "network.py", "exec"), namespace)
    assert namespace["get_open_port"]() == 49234


def test_disabled_timing_never_synchronizes_or_reads_counters(monkeypatch):
    monkeypatch.delenv("DSV4_STARTUP_TIMING", raising=False)
    monkeypatch.setattr(timing, "synchronize", lambda: pytest.fail("disabled synchronization"))
    monkeypatch.setattr(timing, "counters", lambda: pytest.fail("disabled counters"))
    with timing.span("disabled", gpu=True):
        pass


def test_timing_records_failures_rank_cpu_faults_and_io(tmp_path, monkeypatch):
    monkeypatch.setenv("DSV4_STARTUP_TIMING", "1")
    monkeypatch.setenv("DSV4_STARTUP_TIMING_DIR", str(tmp_path))
    with pytest.raises(RuntimeError):
        with timing.span("test"):
            raise RuntimeError("original failure")
    rows = [json.loads(line) for line in next(tmp_path.glob("*.jsonl")).read_text().splitlines()]
    end = rows[-1]
    assert end["outcome"] == "error"
    assert end["elapsed_s"] >= 0
    assert "rank" in end and "cpu_percent" in end
    assert {"ru_minflt", "ru_majflt", "ru_inblock", "ru_oublock", "ru_utime"} <= end["delta"].keys()


def test_comparison_requires_tokens_numerics_and_throughput():
    base = {"tokens": [[1, 2]], "warm_tokens_per_second": 100, "native": {"passed": True}}
    assert compare(base, {**base, "warm_tokens_per_second": 95})["passed"]
    for changes in ({"tokens": [[1, 3]]}, {"warm_tokens_per_second": 94.9}, {"native": {"passed": False}}):
        assert not compare(base, {**base, **changes})["passed"]


def test_summary_counts_compile_edges_not_load_calls(tmp_path):
    rows = [{"event": "jit.load_inline.end", "rank": 0},
            {"event": "jit.build_edges", "compile_count": 2, "build_count": 3},
            {"event": "jit.build_edges", "compile_count": 0, "build_count": 0}]
    (tmp_path / "1.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    assert timing_summary(tmp_path)["compile_count"] == 2


def test_restored_real_ninja_build_needs_no_recompilation(tmp_path):
    """Small CPU recipe tests the absolute-path + mtime dependency contract."""
    ninja = shutil.which("ninja")
    if not ninja:
        pytest.skip("ninja unavailable")
    source = tmp_path / "input.cuh"
    source.write_text("input")
    identity = cache.source_identity([source], {"test": "ninja"})
    local = tmp_path / "node/tvm"
    module = local / "module"
    module.mkdir(parents=True)
    (module / "build.ninja").write_text(
        f"rule build\n  command = cp $in $out\nbuild kernel.so: build {source}\n")
    subprocess.run([ninja, "-C", str(module)], check=True, capture_output=True, umask=0o077)
    (module / "build.ninja").chmod(0o600)
    root = tmp_path / "bundles"
    cache.publish(identity, root, local, {"passed": True})
    cache.preserve(local)
    assert cache.restore(identity, root, local) == "restored"
    out = subprocess.check_output([ninja, "-C", str(module)], text=True)
    assert "no work to do" in out
