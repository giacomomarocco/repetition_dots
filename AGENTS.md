# Project guide

This repository is a minimal local runner for the text-only portion of
`Qwen/Qwen3.5-4B` on an Apple-silicon Mac with 16 GB unified memory.

## Layout

- `run_qwen.py`: Loads `Qwen3_5ForCausalLM`, selects MPS when available, and
  runs one short deterministic prompt.
- `pyproject.toml`: Declares the three direct runtime dependencies: PyTorch,
  Transformers, and Accelerate.
- `uv.lock`: Reproducible dependency lockfile.
- `.venv/`: Local `uv` environment. It is generated and ignored by Git.
- `.hf-cache/`: Project-local Hugging Face model cache. It contains the
  completed Qwen checkpoint, is generated, and is ignored by Git.
- `.uv-cache/`: Disposable project-local `uv` download cache, ignored by Git.

## Run

The environment and model are already installed locally. From the repository
root, run:

```bash
.venv/bin/python run_qwen.py
```

To recreate the environment from the lockfile:

```bash
uv sync --frozen
```

## Apple-silicon notes

- The runner uses MPS with BF16 and falls back to CPU only when MPS is not
  available.
- Keep `HF_DEACTIVATE_ASYNC_LOAD=1`: asynchronous dtype conversion caused the
  model loader to abort on MPS.
- Keep `PYTORCH_ENABLE_MPS_FALLBACK=1`: Qwen3.5 may require CPU fallback for
  operations that Metal does not implement.
- The script disables Qwen's thinking preamble and caps generation at 12
  tokens. The portable Gated DeltaNet path is unusually slow on MPS for longer
  generations because CUDA-only optimized kernels are unavailable.
- `HF_HOME` defaults to `.hf-cache` inside the repository to avoid downloading
  a second copy into the user's global Hugging Face cache.
- `HF_HUB_DISABLE_XET=1` keeps downloads on the standard HTTPS path, which was
  reliable on this machine.

## Verified result

The runner successfully loaded all 426 model weights on MPS using BF16 and
generated:

```text
Sunlight scatters off air molecules.
```

Do not delete `.hf-cache` unless the user explicitly wants to reclaim the
roughly 8.7 GB occupied by the model. Abandoned `.incomplete` model fragments
and the disposable `uv` cache were already cleaned after setup.
