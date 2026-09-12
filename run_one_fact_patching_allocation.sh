#!/usr/bin/env bash
# Inside a separately approved allocation:
# salloc --account=m5258_g --qos=interactive --nodes=1 --gpus=4 --constraint='gpu&hbm80g' --time=04:00:00
# srun --ntasks=1 --cpus-per-task=128 --cpu-bind=none --gpus=4 bash run_one_fact_patching_allocation.sh
set -euo pipefail
workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${SLURM_JOB_ID:?Use an explicitly approved four-hour interactive GPU allocation}"
campaign_root="${1:-${workspace_dir}/runs/deepseek-v4-flash/one-fact-patching-discovery}"
cd "${workspace_dir}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
exec "${workspace_dir}/.venv-sglang/bin/python" -m scripts.dsv4.one_fact_patching run --root "${campaign_root}"
