#!/usr/bin/env bash
#SBATCH --account=m5258_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --job-name=top-object-lens
#SBATCH --output=runs/deepseek-v4-flash/logit-lens/top-object-%j.log

set -euo pipefail

workspace_dir="/pscratch/sd/m/marocco/sandbox/mech_int"
source_rows="${workspace_dir}/runs/deepseek-v4-flash/logit-lens/57804608/filler-grid-expanded/lens_rows.jsonl"
grid_root="${workspace_dir}/runs/deepseek-v4-flash/logit-lens/57804608/filler-grid-expanded"
capture_root="${workspace_dir}/runs/deepseek-v4-flash/logit-lens/57804608/captures"
checkpoint="${workspace_dir}/model/DeepSeek-V4-Flash-0731-MoE-MXFP4-BF16"
output_dir="${workspace_dir}/runs/deepseek-v4-flash/logit-lens/57804608/top-object-heatmap"

cd "${workspace_dir}"
mkdir -p "${output_dir}"

srun .venv-sglang/bin/python -m scripts.dsv4.add_numeric_argmax \
  --rows "${source_rows}" --grid-root "${grid_root}" \
  --capture-root "${capture_root}" --checkpoint "${checkpoint}" \
  --output "${output_dir}/lens_rows_with_numeric_argmax.jsonl" --device cuda:0

srun .venv-sglang/bin/python -m scripts.dsv4.plot_top_object_heatmap \
  --rows "${output_dir}/lens_rows_with_numeric_argmax.jsonl" --filler-length 20 \
  --output "${output_dir}/top_object_heatmap_k20.png" \
  --stats "${output_dir}/top_object_heatmap_k20.json"
