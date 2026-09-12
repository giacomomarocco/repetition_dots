#!/usr/bin/env bash
# Run via salloc ... srun ... bash this_file after explicit allocation approval.
set -euo pipefail
workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${SLURM_JOB_ID:?Use an explicitly approved four-GPU interactive allocation}"
cd "${workspace_dir}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
exec "${workspace_dir}/.venv-sglang/bin/python" -m scripts.dsv4.answer_token_heatmap run "$@"
