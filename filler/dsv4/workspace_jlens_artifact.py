"""Pinned public workspace J-Lens artifact; download without importing Torch."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = "camilablank/workspace-lenses"
REVISION = "d740106d1e0f95456dc8718fba2895e9c8ffd6ef"
FILENAME = "deepseek-v4-flash/j-lens/lens.pt"
SHA256 = "8b010eef8b2b08efb1b07601e5203ff5d215b1fcae63704847fdf001e61e0efc"
SIZE = 1409295893
DEFAULT_PATH = ROOT / "model/workspace-lenses" / FILENAME


def verify_artifact(path: Path) -> str:
    if path.stat().st_size != SIZE:
        raise ValueError("workspace lens size differs from the pinned artifact")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != SHA256:
        raise ValueError("workspace lens SHA-256 differs from the pinned artifact")
    return digest.hexdigest()


def prepare_artifact(path: Path = DEFAULT_PATH) -> Path:
    """Resume a download and publish it only after size/checksum verification."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        partial = path.with_name(path.name + ".partial")
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > SIZE:
            raise ValueError("partial download exceeds the pinned file size")
        if offset < SIZE:
            url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{FILENAME}"
            request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
            with urllib.request.urlopen(request, timeout=60) as response:
                if offset and (response.status != 206 or not response.headers.get(
                    "Content-Range", "").startswith(f"bytes {offset}-")):
                    raise ValueError("server did not honor download resume")
                with partial.open("ab" if offset else "xb") as sink:
                    report_at = time.monotonic()
                    while block := response.read(8 * 1024 * 1024):
                        offset += len(block)
                        if offset > SIZE:
                            raise ValueError("download exceeds the pinned file size")
                        sink.write(block)
                        if time.monotonic() - report_at >= 15:
                            print(f"Downloaded {offset}/{SIZE} bytes", flush=True)
                            report_at = time.monotonic()
        verify_artifact(partial)
        # Atomic publication that fails if another process already published it.
        path.hardlink_to(partial)
        partial.unlink()
    else:
        verify_artifact(path)
    provenance = {"repository": REPOSITORY, "revision": REVISION, "filename": FILENAME,
                  "size": SIZE, "sha256": SHA256, "stream_reduction": "mean",
                  "source_layers": list(range(42)), "target_layer": 41}
    source = path.with_name("SOURCE.json")
    content = json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    if source.exists():
        if source.read_text() != content:
            raise ValueError("existing SOURCE.json differs from the pinned artifact")
    else:
        with source.open("x") as sink:
            sink.write(content)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()
    print(f"Verified workspace J-Lens: {prepare_artifact(args.output)}", flush=True)
