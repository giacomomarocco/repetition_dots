#!/usr/bin/env bash
set -euo pipefail

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${workspace_dir}"
: "${SLURM_JOB_ID:?Run through srun inside an approved CPU allocation}"
output_dir="${1:-${workspace_dir}/runs/deepseek-v4-flash/jlens-smoke}"
mkdir -p "${output_dir}"

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MPLCONFIGDIR="${TMPDIR:-/tmp}/jlens-matplotlib-${USER}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

TZ=UTC scontrol show job -o "${SLURM_JOB_ID}" > "${output_dir}/allocation.txt"
date -u '+%Y-%m-%dT%H:%M:%SZ' > "${output_dir}/allocation-start.txt"
"${workspace_dir}/.venv-sglang/bin/python" -m scripts.dsv4.validate_jlens \
  --device cpu --threads 16 --output-dir "${output_dir}" \
  > "${output_dir}/execution.log" 2>&1
