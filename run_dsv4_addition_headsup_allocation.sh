#!/usr/bin/env bash
set -euo pipefail

workspace_dir="/pscratch/sd/m/marocco/sandbox/mech_int"
run_job_id="${SLURM_JOB_ID:?run inside an approved four-GPU allocation}"
run_root="${workspace_dir}/runs/deepseek-v4-flash/logit-lens/${run_job_id}/addition-headsup"
capture_root="${workspace_dir}/runs/deepseek-v4-flash/logit-lens/${run_job_id}/captures"
checkpoint="${workspace_dir}/model/DeepSeek-V4-Flash-0731-MoE-MXFP4-BF16"
one_output="${workspace_dir}/runs/deepseek-v4-flash/one-fact-addition-full-headsup"
two_output="${workspace_dir}/runs/deepseek-v4-flash/two-fact-addition-5shot-atomic-200-headsup"
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
DSV4_LENS_RUN_ID="${run_job_id}" DEEPSEEK_STARTUP_PROFILE=throughput \
  ./run_deepseek_v4_logit_lens.sh >"${run_root}/server.log" 2>&1 &
server_pid=$!

for attempt in $(seq 1 120); do
  if curl -fsS http://127.0.0.1:30002/health >/dev/null; then
    break
  fi
  if [[ ${attempt} -eq 120 ]]; then
    echo "Server did not become healthy within 30 minutes" >&2
    exit 1
  fi
  sleep 15
done

echo "Validating final-layer equivalence before experimental inference"
.venv-sglang/bin/python -m scripts.dsv4.probe_logit_lens \
  --output "${run_root}/native_response.json" --capture-root "${capture_root}" \
  --pass-id-output "${run_root}/validation_pass.json"
validation_pass_id="$(.venv-sglang/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["pass_id"])' \
  "${run_root}/validation_pass.json")"
.venv-sglang/bin/python -m scripts.dsv4.validate_logit_lens \
  --capture-root "${capture_root}" --native-response "${run_root}/native_response.json" \
  --checkpoint "${checkpoint}" --output "${run_root}/validation.json" \
  --pass-id "${validation_pass_id}" --device cuda:0

echo "Running the complete paired one-fact heads-up evaluation"
.venv-sglang/bin/python -m scripts.addition.one_fact \
  --prompt-variant local-headsup --max-facts 262 \
  --filler-lengths 0 10 20 50 100 --output-dir "${one_output}"

echo "Running the 200-pair atomic two-fact heads-up evaluation"
.venv-sglang/bin/python -m scripts.addition.two_fact \
  --facts runs/deepseek-v4-flash/two-fact-addition-full-under-999/eligible_facts.json \
  --fact-kind atomic --max-pairs 200 --prompt-variant local-headsup \
  --filler-lengths 0 10 20 50 100 --output-dir "${two_output}"

echo "Capturing stratified residual trajectories for both experiments"
.venv-sglang/bin/python -m scripts.dsv4.capture_addition_headsup \
  --prompts "${one_output}/prompts.json" "${two_output}/prompts.json" \
  --model "${checkpoint}" --capture-root "${capture_root}" \
  --output-root "${run_root}/trajectories" \
  --pairs-per-experiment 16 --trajectory-stride 5

echo "Projecting captures through the native unembedding"
.venv-sglang/bin/python -m scripts.dsv4.analyze_addition_headsup \
  --grid-root "${run_root}/trajectories" --capture-root "${capture_root}" \
  --checkpoint "${checkpoint}" --output "${run_root}/lens_rows.jsonl" --device cuda:0

.venv-sglang/bin/python -c \
  'import json,sys; from datetime import datetime,timezone; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({"status":"complete","job_id":sys.argv[2],"finished_at":datetime.now(timezone.utc).isoformat()},indent=2)+"\n")' \
  "${run_root}/COMPLETE.json" "${run_job_id}"
echo "Heads-up evaluations and residual analysis completed"
