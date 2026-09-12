#!/usr/bin/env bash
set -euo pipefail
workspace_dir="/pscratch/sd/m/marocco/sandbox/mech_int"
: "${SLURM_JOB_ID:?Use an explicitly approved four-A100-80GB allocation}"
capture_run="${1:?Pass the complete capture runtime directory}"
snapshot_dir="${capture_run}/recovery-source"
cd "${snapshot_dir}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/five-shot-recovery-mpl-${SLURM_JOB_ID}"
exec "${workspace_dir}/.venv-sglang/bin/python" -m scripts.dsv4.recover_five_shot_recurrence --output "${capture_run}"
