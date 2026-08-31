#!/usr/bin/env bash
set -euo pipefail

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
model_dir="${workspace_dir}/model/DeepSeek-V4-Flash-0731-MoE-MXFP4-BF16"

source "${workspace_dir}/.venv-sglang/bin/activate"

export ENABLE_SGLANG_DSV4_A100_PATCH=1
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
export SGLANG_OPT_FUSE_WQA_WKV=0
export SGLANG_OPT_USE_TOPK_V2=0
export SGLANG_TOPK_TRANSFORM_512_TORCH=0
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1
export SGLANG_DSV4_A100_INT8_INDEXER=1
export SGLANG_DSV4_INDEXER_QUERY_CP_PREFILL=1
export SGLANG_DSV4_FP4_EXPERTS=1
export SGLANG_OPT_FP8_WO_A_GEMM=0
export SGLANG_DSV4_MXFP4_MOE_BACKEND=mxfp4_int8
export PYTHONPATH="${workspace_dir}/ports/deepseek-v4-a100-sglang:${workspace_dir}/ports/sglang/python"

# Perlmutter exports HOST as a login-node hostname. Rendezvous must stay local.
export HOST=127.0.0.1

# The external NCCL net plugin was linked against CUDA 12, while this environment
# uses CUDA 13. The socket transport over Slingshot was verified to work.
export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=hsn

# Avoid shared-home cache locks and metadata latency.
export TVM_FFI_CACHE_DIR="${TMPDIR:-/tmp}/deepseek-v4-tvm-${USER}"
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/deepseek-v4-triton-${USER}"
export FLASHINFER_WORKSPACE_BASE="${TMPDIR:-/tmp}/deepseek-v4-flashinfer-${USER}"
mkdir -p "${TVM_FFI_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${FLASHINFER_WORKSPACE_BASE}"

exec python -m sglang.launch_server \
  --model-path "${model_dir}" \
  --tp-size 4 \
  --host 127.0.0.1 \
  --port "${SGLANG_PORT:-30002}" \
  --trust-remote-code \
  --disable-custom-all-reduce \
  --skip-server-warmup
