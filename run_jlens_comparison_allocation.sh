#!/usr/bin/env bash
set -euo pipefail
workspace_dir="/pscratch/sd/m/marocco/sandbox/mech_int"
cd "${workspace_dir}"
: "${SLURM_JOB_ID:?Run through srun inside an approved CPU allocation}"
output_dir="${1:-${workspace_dir}/runs/deepseek-v4-flash/jlens-comparison/${SLURM_JOB_ID}}"
log_dir="${workspace_dir}/runs/deepseek-v4-flash/jlens-comparison-execution/${SLURM_JOB_ID}"
mkdir -p "${log_dir}"
exec > >(tee -a "${log_dir}/execution.log") 2>&1
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/jlens-comparison-mpl-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/jlens-comparison-cache-${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PATH="${workspace_dir}/.venv-sglang/bin:${PATH}"
TZ=UTC scontrol show job -o "${SLURM_JOB_ID}" > "${log_dir}/allocation.txt"
date -u '+%Y-%m-%dT%H:%M:%SZ' > "${log_dir}/start.txt"
trap 'status=$?; date -u +%Y-%m-%dT%H:%M:%SZ > "${log_dir}/end.txt"; echo "Exit status: ${status}"' EXIT
"${workspace_dir}/.venv-sglang/bin/python" -m scripts.dsv4.compare_jlenses \
  --threads 16 --batch-size 64 --output-dir "${output_dir}"
