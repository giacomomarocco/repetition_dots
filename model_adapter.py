"""Architecture-neutral Hugging Face loading and prompt rendering helpers."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


@dataclass(frozen=True)
class LoadedModel:
    model: Any
    tokenizer: Any
    input_device: torch.device
    metadata: dict[str, Any]


def resolve_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps requested, but MPS is unavailable")
    return requested


def resolve_dtype(requested: str, device: str) -> torch.dtype:
    if requested != "auto":
        dtype = DTYPES[requested]
        if device == "cpu" and dtype == torch.float16:
            raise ValueError("float16 inference on CPU is unsupported")
        return dtype
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        return torch.bfloat16
    return torch.float32


def local_revision(path_or_id: str) -> str | None:
    path = Path(path_or_id).expanduser()
    if not path.is_dir():
        return None
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def configured_revision(path_or_id: str) -> str | None:
    path = Path(path_or_id).expanduser() / "config.json"
    if not path.is_file():
        return None
    config = json.loads(path.read_text(encoding="utf-8"))
    return config.get("_commit_hash")


def render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> tuple[str, Any]:
    if getattr(tokenizer, "chat_template", None):
        kwargs = {
            "add_generation_prompt": True,
            "tokenize": False,
        }
        try:
            text = tokenizer.apply_chat_template(
                messages, enable_thinking=False, **kwargs
            )
        except TypeError:
            text = tokenizer.apply_chat_template(messages, **kwargs)
        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        return text, encoded

    system = messages[0]["content"]
    question = messages[1]["content"]
    text = f"{system}\n\nQuestion: {question}\nAnswer:"
    encoded = tokenizer(text, return_tensors="pt")
    return text, encoded


def load_tokenizer(
    model_id: str,
    tokenizer_id: str | None,
    cache_dir: Path | None,
    local_files_only: bool,
) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        tokenizer_id or model_id,
        cache_dir=str(cache_dir) if cache_dir else None,
        local_files_only=local_files_only,
    )


def load_model(
    model_id: str,
    tokenizer_id: str | None,
    device_request: str,
    dtype_request: str,
    cache_dir: Path | None,
    local_files_only: bool,
) -> LoadedModel:
    from transformers import AutoModelForCausalLM

    device = resolve_device(device_request)
    dtype = resolve_dtype(dtype_request, device)
    tokenizer = load_tokenizer(model_id, tokenizer_id, cache_dir, local_files_only)
    load_kwargs = {
        "cache_dir": str(cache_dir) if cache_dir else None,
        "local_files_only": local_files_only,
        "dtype": dtype,
        "device_map": "auto" if device == "cuda" else device,
    }
    loader_name = "AutoModelForCausalLM"
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    except ValueError as error:
        if "Unrecognized configuration class" not in str(error):
            raise
        from transformers import AutoModelForImageTextToText

        loader_name = "AutoModelForImageTextToText"
        model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs)
    model.eval()
    input_device = next(model.parameters()).device
    gpu_names = []
    if device == "cuda":
        gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    revision = local_revision(model_id) or configured_revision(model_id)
    tokenizer_source = tokenizer_id or model_id
    tokenizer_revision = local_revision(tokenizer_source) or configured_revision(
        tokenizer_source
    )
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        input_device=input_device,
        metadata={
            "device": device,
            "input_device": str(input_device),
            "dtype": str(dtype).removeprefix("torch."),
            "model_loader": loader_name,
            "model_revision": revision,
            "tokenizer_revision": tokenizer_revision,
            "gpu_count": torch.cuda.device_count() if device == "cuda" else 0,
            "gpu_names": gpu_names,
        },
    )
