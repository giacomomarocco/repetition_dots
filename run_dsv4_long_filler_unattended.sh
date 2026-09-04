#!/usr/bin/env bash
set -euo pipefail

workspace_dir="/pscratch/sd/m/marocco/sandbox/mech_int"
run_job_id="${SLURM_JOB_ID:?run inside the target allocation}"
filler_length="${DSV4_FILLER_LENGTH:?set DSV4_FILLER_LENGTH to 50 or 100}"
case "${filler_length}" in
  50|100) ;;
  *) echo "DSV4_FILLER_LENGTH must be 50 or 100" >&2; exit 2 ;;
esac
run_root="${workspace_dir}/runs/deepseek-v4-flash/logit-lens/${run_job_id}"
capture_root="${run_root}/captures"
pipeline_root="${run_root}/long-filler-pipeline"
mkdir -p "${pipeline_root}"
exec > >(tee -a "${pipeline_root}/pipeline.log") 2>&1

release_allocation() {
  code=$?
  trap - EXIT
  if [[ ${code} -eq 0 ]]; then
    status="complete"
    marker="${pipeline_root}/COMPLETE.json"
  else
    status="failed"
    marker="${pipeline_root}/FAILED.json"
  fi
  "${workspace_dir}/.venv-sglang/bin/python" -c \
    'import json,sys; from datetime import datetime,timezone; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({"status":sys.argv[2],"job_id":sys.argv[3],"finished_at":datetime.now(timezone.utc).isoformat()},indent=2)+"\n")' \
    "${marker}" "${status}" "${run_job_id}"
  sync
  echo "Pipeline ${status}; releasing allocation ${run_job_id}"
  scancel "${run_job_id}" || true
  exit "${code}"
}
trap release_allocation EXIT

cd "${workspace_dir}"
echo "Waiting for instrumented server in allocation ${run_job_id}"
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

echo "Validating native final-layer equivalence"
.venv-sglang/bin/python -m scripts.dsv4.probe_logit_lens \
  --output "${pipeline_root}/native_response.json" \
  --capture-root "${capture_root}" \
  --pass-id-output "${pipeline_root}/validation_pass.json"
validation_pass_id="$(.venv-sglang/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["pass_id"])' \
  "${pipeline_root}/validation_pass.json")"
.venv-sglang/bin/python -m scripts.dsv4.validate_logit_lens \
  --capture-root "${capture_root}" \
  --native-response "${pipeline_root}/native_response.json" \
  --checkpoint model/DeepSeek-V4-Flash-0731-MoE-MXFP4-BF16 \
  --output "${pipeline_root}/validation.json" \
  --pass-id "${validation_pass_id}" \
  --device cuda:0

echo "Capturing every discovery filler token at length ${filler_length}"
.venv-sglang/bin/python -m scripts.dsv4.capture_factorial_grid \
  --rendered \
    "runs/deepseek-v4-flash/factorial/filler-discovery-rendered-k${filler_length}.json" \
  --model model/DeepSeek-V4-Flash-0731-MoE-MXFP4-BF16 \
  --capture-root "${capture_root}" \
  --output-root "${run_root}/filler-discovery-long"

echo "Projecting long-filler discovery captures"
.venv-sglang/bin/python -m scripts.dsv4.analyze_factorial_grid \
  --grid-root "${run_root}/filler-discovery-long" \
  --capture-root "${capture_root}" \
  --checkpoint model/DeepSeek-V4-Flash-0731-MoE-MXFP4-BF16 \
  --output "${run_root}/filler-discovery-long/lens_rows.jsonl" \
  --device cuda:0
.venv-sglang/bin/python -m scripts.dsv4.summarize_frozen_filler \
  --rows "${run_root}/filler-discovery-long/lens_rows.jsonl" \
  --output "${run_root}/filler-discovery-long/summary.json"

echo "Capturing preregistered confirmation sites at length ${filler_length}"
.venv-sglang/bin/python -m scripts.dsv4.capture_factorial_grid \
  --rendered \
    "runs/deepseek-v4-flash/factorial/filler-confirmation-rendered-k${filler_length}.json" \
  --model model/DeepSeek-V4-Flash-0731-MoE-MXFP4-BF16 \
  --capture-root "${capture_root}" \
  --output-root "${run_root}/filler-confirmation-long" \
  --frozen-final-sites

echo "Projecting and summarizing long-filler confirmation"
.venv-sglang/bin/python -m scripts.dsv4.analyze_factorial_grid \
  --grid-root "${run_root}/filler-confirmation-long" \
  --capture-root "${capture_root}" \
  --checkpoint model/DeepSeek-V4-Flash-0731-MoE-MXFP4-BF16 \
  --output "${run_root}/filler-confirmation-long/lens_rows.jsonl" \
  --device cuda:0
.venv-sglang/bin/python -m scripts.dsv4.summarize_frozen_filler \
  --rows "${run_root}/filler-confirmation-long/lens_rows.jsonl" \
  --output "${run_root}/filler-confirmation-long/summary.json"

echo "All length-${filler_length} inference and analysis completed"
