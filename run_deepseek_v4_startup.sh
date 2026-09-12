#!/usr/bin/env bash
# Experimental launcher; the original is frozen by an active campaign manifest.
# Keep serving settings identical; tests compare the effective launch commands.
set -euo pipefail
export DSV4_STARTUP_EPOCH="${DSV4_STARTUP_EPOCH:-$(date +%s.%N)}"
case "${DSV4_KERNEL_CACHE_MODE:-off}" in off|reuse) ;; *) echo "Invalid DSV4_KERNEL_CACHE_MODE" >&2; exit 2 ;; esac

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
model_dir="${workspace_dir}/model/DeepSeek-V4-Flash-0731-MoE-MXFP4-BF16"
startup_profile="${DEEPSEEK_STARTUP_PROFILE:-throughput}"

case "${startup_profile}" in
  fast)
    # Interactive default: avoid the ~268 s CUDA-graph capture measured in the
    # initial audit. Eager execution can have lower steady-state throughput.
    startup_args=(--disable-cuda-graph)
    ;;
  balanced)
    # Capture only the small batch sizes most useful for interactive requests.
    startup_args=(--cuda-graph-max-bs 8)
    ;;
  throughput)
    # Preserve SGLang's auto-sized graph set for maximum serving throughput.
    startup_args=()
    ;;
  *)
    echo "Unknown DEEPSEEK_STARTUP_PROFILE=${startup_profile@Q}; expected fast, balanced, or throughput" >&2
    exit 2
    ;;
esac

source "${workspace_dir}/.venv-sglang/bin/activate"

export ENABLE_SGLANG_DSV4_A100_PATCH=1
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
# Layer-output hooks require complete post-block mHC residuals.  Keep the
# cross-layer deferred hc_post/hc_pre optimization off for Logit Lens work.
export SGLANG_OPT_FUSE_MHC_POST_PRE=0
export SGLANG_OPT_FUSE_WQA_WKV=0
export SGLANG_OPT_USE_TOPK_V2=0
export SGLANG_TOPK_TRANSFORM_512_TORCH=0
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1
export SGLANG_DSV4_A100_INT8_INDEXER=1
export SGLANG_DSV4_INDEXER_QUERY_CP_PREFILL=1
export SGLANG_DSV4_FP4_EXPERTS=1
export SGLANG_OPT_FP8_WO_A_GEMM=0
export SGLANG_DSV4_MXFP4_MOE_BACKEND=mxfp4_int8
export PYTHONPATH="${workspace_dir}:${workspace_dir}/ports/deepseek-v4-a100-sglang:${workspace_dir}/ports/sglang/python"

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

# SGLANG_PORT is a convenience input only; the internal broadcaster also reads it.
http_port="${SGLANG_PORT:-30002}"
unset SGLANG_PORT

echo "Launching DeepSeek with startup profile '${startup_profile}'" >&2

exec python -m scripts.dsv4.startup_cache -- python -m sglang.launch_server \
  --model-path "${model_dir}" \
  --tp-size 4 \
  --host 127.0.0.1 \
  --port "${http_port}" \
  --trust-remote-code \
  --disable-custom-all-reduce \
  --skip-server-warmup \
  "${startup_args[@]}" \
  "$@"
