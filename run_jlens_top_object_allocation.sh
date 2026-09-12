#!/usr/bin/env bash
set -euo pipefail

workspace_dir="/pscratch/sd/m/marocco/sandbox/mech_int"
cd "${workspace_dir}"
: "${SLURM_JOB_ID:?Run through srun inside the approved CPU allocation}"
output_dir="${workspace_dir}/runs/deepseek-v4-flash/jlens-top-object-heatmap"
log_dir="${workspace_dir}/runs/deepseek-v4-flash/jlens-top-object-execution/${SLURM_JOB_ID}"
mkdir -p "${log_dir}"
exec > >(tee -a "${log_dir}/execution.log") 2>&1
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/jlens-top-object-mpl-${USER}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/jlens-top-object-cache-${USER}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
TZ=UTC scontrol show job -o "${SLURM_JOB_ID}" > "${log_dir}/allocation.txt"
date -u '+%Y-%m-%dT%H:%M:%SZ' > "${log_dir}/start.txt"
trap 'status=$?; date -u +%Y-%m-%dT%H:%M:%SZ > "${log_dir}/end.txt"; echo "Exit status: ${status}"' EXIT
"${workspace_dir}/.venv-sglang/bin/python" -m scripts.dsv4.prepare_jlens_top_object \
  --device cpu --threads 16 --batch-size 64 --output-dir "${output_dir}"
"${workspace_dir}/.venv-sglang/bin/python" -m scripts.dsv4.export_jlens_top_object \
  --run-dir "${output_dir}"
