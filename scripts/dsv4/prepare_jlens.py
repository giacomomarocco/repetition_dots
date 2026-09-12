"""Fetch the pinned public J-Lens source and its default 0731 matrices."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
CODE_REPO = "xiangchensong/jacobian-lens-open-frontier"
CODE_REVISION = "b8b840caaa246ad04d354b4650848f30519184b6"
LENS_REPO = "xiangchensong/jacobian-lens-deepseek-v4-flash-0731"
LENS_REVISION = "a5841da3565b88a79665a418e79d1641d346f667"
LENS_SHA256 = "429d6f3810392cf6af7df5f81058eba3619bba34901108e4cb257b76d2efdfff"
LENS_SIZE = 2818579540


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_once(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"existing file differs from pinned source: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)


def main() -> None:
    source = ROOT / "ports/jacobian-lens-open-frontier"
    url = f"https://codeload.github.com/{CODE_REPO}/tar.gz/{CODE_REVISION}"
    with urllib.request.urlopen(url, timeout=120) as response:
        archive = response.read()
    hashes = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        for member in tar.getmembers():
            parts = PurePosixPath(member.name).parts[1:]
            if not parts or member.isdir():
                continue
            if ".." in parts or member.name.startswith("/"):
                raise ValueError("unsafe archive path")
            relative = PurePosixPath(*parts)
            if parts[0] != "jlens" and str(relative) not in {"LICENSE", "README.md", "pyproject.toml"}:
                continue
            if not member.isfile():
                raise ValueError(f"unexpected source archive entry: {relative}")
            content = tar.extractfile(member).read()
            write_once(source / str(relative), content)
            hashes[str(relative)] = hashlib.sha256(content).hexdigest()
    metadata = {"repository": CODE_REPO, "revision": CODE_REVISION,
                "archive_sha256": hashlib.sha256(archive).hexdigest(), "files": hashes}
    write_once(source / "SOURCE.json", (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode())
    print(f"Pinned source: {CODE_REVISION}, {len(hashes)} files", flush=True)

    directory = ROOT / "model/jacobian-lens-deepseek-v4-flash-0731"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "lens.pt"
    if not target.exists():
        partial = directory / "lens.pt.partial"
        offset = partial.stat().st_size if partial.exists() else 0
        url = f"https://huggingface.co/{LENS_REPO}/resolve/{LENS_REVISION}/lens.pt?download=true&ts={int(time.time())}"
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
        with urllib.request.urlopen(request, timeout=120) as response:
            if offset and (response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                raise ValueError("server did not honor download resume")
            with partial.open("ab" if offset else "xb") as handle:
                last_report = time.monotonic()
                while block := response.read(8 * 1024 * 1024):
                    handle.write(block)
                    offset += len(block)
                    if offset > LENS_SIZE:
                        raise ValueError("download exceeds pinned file size")
                    if time.monotonic() - last_report >= 15:
                        print(f"Downloaded {offset}/{LENS_SIZE} bytes", flush=True)
                        last_report = time.monotonic()
        if partial.stat().st_size != LENS_SIZE or file_digest(partial) != LENS_SHA256:
            raise ValueError("lens download size/checksum mismatch")
        partial.rename(target)
    if target.stat().st_size != LENS_SIZE or file_digest(target) != LENS_SHA256:
        raise ValueError("existing lens file differs from pinned checkpoint")
    metadata = {"repository": LENS_REPO, "revision": LENS_REVISION,
                "filename": "lens.pt", "sha256": LENS_SHA256, "size": LENS_SIZE}
    write_once(directory / "SOURCE.json", (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode())
    print(f"Verified lens: {target}", flush=True)


if __name__ == "__main__":
    main()
