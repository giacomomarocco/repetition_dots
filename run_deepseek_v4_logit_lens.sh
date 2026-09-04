#!/usr/bin/env bash
set -euo pipefail

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base_launcher="${DSV4_BASE_LAUNCHER:-${workspace_dir}/run_deepseek_v4_a100.sh}"
run_id="${DSV4_LENS_RUN_ID:-${SLURM_JOB_ID:-manual}}"
capture_root="${DSV4_LENS_CAPTURE_ROOT:-${workspace_dir}/runs/deepseek-v4-flash/logit-lens/${run_id}/captures}"
mkdir -p "${capture_root}"

hook_json="$("${workspace_dir}/.venv-sglang/bin/python" -c \
  'import sys; from filler.dsv4.hooks import print_hook_spec; print_hook_spec(sys.argv[1])' \
  "${capture_root}")"

echo "Logit Lens captures: ${capture_root}" >&2
echo "Arm one later capture by updating: ${capture_root}/CAPTURE_NEXT" >&2

exec "${base_launcher}" \
  --enable-return-hidden-states \
  --forward-hooks "${hook_json}" \
  "$@"
