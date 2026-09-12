#!/usr/bin/env bash
set -euo pipefail

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base_launcher="${DSV4_BASE_LAUNCHER:-${workspace_dir}/run_deepseek_v4_a100.sh}"
run_id="${DSV4_TRANSPLANT_RUN_ID:-${SLURM_JOB_ID:-manual}}"
run_root="${DSV4_TRANSPLANT_ROOT:-${workspace_dir}/runs/deepseek-v4-flash/transplants/${run_id}}"
capture_root="${run_root}/validation-captures"
control_root="${run_root}/control"
mkdir -p "${capture_root}" "${control_root}"

hook_json="$("${workspace_dir}/.venv-sglang/bin/python" -c '
import json, sys
from filler.dsv4.hooks import hook_spec, transplant_hook_spec
print(json.dumps(hook_spec(sys.argv[1]) + transplant_hook_spec(sys.argv[2]), separators=(",", ":")))
' "${capture_root}" "${control_root}")"

echo "Transplant controls: ${control_root}" >&2
echo "Validation captures: ${capture_root}" >&2
exec "${base_launcher}" --enable-return-hidden-states --forward-hooks "${hook_json}" "$@"
