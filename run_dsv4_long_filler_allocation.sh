#!/usr/bin/env bash
set -euo pipefail

workspace_dir="/pscratch/sd/m/marocco/sandbox/mech_int"
run_job_id="${SLURM_JOB_ID:?run inside the target allocation}"
filler_length="${1:?usage: $0 50|100}"
case "${filler_length}" in
  50|100) ;;
  *) echo "filler length must be 50 or 100" >&2; exit 2 ;;
esac
run_root="${workspace_dir}/runs/deepseek-v4-flash/logit-lens/${run_job_id}"
mkdir -p "${run_root}"

server_pid=""
stop_server() {
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
}
trap stop_server EXIT INT TERM

cd "${workspace_dir}"
DEEPSEEK_STARTUP_PROFILE=fast \
  ./run_deepseek_v4_logit_lens.sh \
  >"${run_root}/server.log" 2>&1 &
server_pid=$!

DSV4_FILLER_LENGTH="${filler_length}" ./run_dsv4_long_filler_unattended.sh
