#!/usr/bin/env bash
# salloc owns this srun command and releases the allocation when it exits.
set -euo pipefail
workspace_dir="/pscratch/sd/m/marocco/sandbox/mech_int"
: "${SLURM_JOB_ID:?Use an explicitly approved four-A100-80GB interactive allocation}"
cd "${workspace_dir}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/five-shot-recurrence-mpl-${SLURM_JOB_ID}"
exec "${workspace_dir}/.venv-sglang/bin/python" -m scripts.dsv4.five_shot_recurrence run "$@"
