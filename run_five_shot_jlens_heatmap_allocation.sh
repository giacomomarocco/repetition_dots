#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Use an explicitly approved four-A100-80GB allocation}"
workspace_dir="/pscratch/sd/m/marocco/sandbox/mech_int"
cd "${workspace_dir}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset MPLBACKEND
exec "${workspace_dir}/.venv-sglang/bin/python" -m scripts.dsv4.five_shot_jlens_heatmap run
