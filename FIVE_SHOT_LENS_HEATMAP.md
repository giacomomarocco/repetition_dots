# Five-shot Logit Lens and square J-Lens heatmaps

## 2026-09-12T01:48:20.053602+00:00 — scoring-only preparation

Objective: extend both top-object heatmap notebooks from the 96-prompt factorial discovery set to the existing 262-fact five-shot ensemble. User requested both square J-Lens and ordinary Logit Lens; no new prompt execution is needed.

Input: completed `runs/deepseek-v4-flash/five-shot-square-recurrence/runtimes/58202160-5cc37d34`, with 524 paired prompts and 7,860 selected positions. Preserve strict native labels: k=0 has 137 correct/125 wrong; k=20 has 179 correct/83 wrong. Each condition has 262 facts, one addend per fact. Five-shot prompts differ from the older factorial format: last question token, all fillers, Answer, colon, assistant marker and end-think marker are captured; there is no trailing space. Filler length changes in the demonstrations too.

Prepared `python -m scripts.dsv4.five_shot_jlens_heatmap run`, allocation wrapper `run_five_shot_jlens_heatmap_allocation.sh`. Four independent GPU workers apply both readouts per saved prompt: 330,120 square-lens rows (layers 0–41), 337,980 ordinary-lens rows (layers 0–42), 668,100 total. Square readout uses published FP32 mean/transport and BF16 norm/head; ordinary readout uses native SGLang fused mHC/RMSNorm and TP=4 vocabulary GEMMs. Numerical token IDs use the existing canonical-nonnegative-integer definition; exact ties choose the lowest vocabulary ID.

Validation: metadata-only preflight checks original completion hashes, every capture/control/response link, actual token positions, cohorts, source pins, checkpoint/capture file metadata. Current numerical source hashes match the original validated run. Compute will checksum and compare all four capture ranks, rerun native final-layer equivalence for every prompt, validate square reference transport/readout at every layer for both conditions on every GPU, and require exact agreement with saved square full-vocabulary argmax and target logits at every position/layer. Both notebooks execute with the selected NERSC Python kernel, actual inline PNG output and original Matplotlib rc settings before publishing COMPLETE. Partial outputs are never selected automatically.

CPU validation: 49 related tests passed (one initial existing cmr10.tfm failure passed after setting the notebook TeX Live PATH); final focused tests and shell syntax also passed. See [preflight checks](runs/deepseek-v4-flash/five-shot-jlens-heatmap/preflight-checks.txt). No GPU scoring has run yet. Final manifest hash: `39b2a0daac0bfc4847799ee04e012f1ac3233a9f825ce5210a1b866f4cd35e04`.

New output root: `runs/deepseek-v4-flash/five-shot-jlens-heatmap/`. Per-prompt NPZ winners, separate square/logit aggregate summaries, source snapshot, GPU/runtime logs, validation, rendered PNGs and executed notebooks will be saved in its allocation runtime. Both existing notebooks gain a dataset selector; the completed five-shot run is the default when available, with explicit factorial selection retained. Original capture/scoring files are read-only.

Prepared allocation: one interactive node, account m5258_g, four A100 80GB GPUs, quoted gpu&hbm80g constraint, 90-minute maximum. It runs the prepared scoring/export automatically and releases the allocation on success or failure. Submission awaits explicit Slurm approval as required by AGENTS.md. No commit has been made.

## 2026-09-12 — first scoring allocation stopped on unrelated source drift

Approved job 58218257 used nid008285, m5258_g/gpu_interactive, four A100 80GB GPUs. It saved 523 prompt scores before failing its final source verification because another task edited `filler/dsv4/patching.py`; the allocation released automatically. No completion pointer was published. Existing partial scores are retained in that runtime.

Audit: the six imported storage/native-score helpers (digest, file_digest, sync_directory, atomic_bytes, atomic_json, requested_scores) are AST-identical to the pinned versions. The exact diff is `runs/deepseek-v4-flash/five-shot-jlens-heatmap/source-drift-audit.diff`. The scoring entry point now imports the exact original patching module from a frozen local copy before all other imports; that copy is pinned in the new plan. The mutable campaign module is no longer an execution dependency. Numerical modules and source activations remain unchanged. Per-prompt validation reports are now persisted immediately, so a controller failure cannot discard validation evidence. A fresh scoring runtime will rerun both readouts and all gates.

Retry CPU preflight: 5 focused tests passed; frozen-module import paths and source verification passed. New configuration: `c3766011f8dd6e682ab946a981950021f1553f063205e96e573635a84e1dab63`. The observed scoring speed supports a 30-minute retry allocation, including notebook export.

## 2026-09-12T02:05:46.286330+00:00 — completed and verified

Job **58218428** completed both scoring paths and both notebooks, then released its allocation. All **524 prompts / 262 paired facts**, **330,120 square J-Lens readouts**, and **337,980 ordinary Logit Lens readouts** are complete. All 524 native checks passed with maximum log-probability error **0.0**; all 336 square reference checks passed, and every saved square winner and target logit matched exactly. Six inline figures per notebook were rendered in the selected NERSC Python environment with the original rc/TeX settings; both k=20 all-example layouts were visually checked. The final audit rehashed all 524 scored NPZs and both summaries.

The working notebooks now contain the twelve executed figures and default to the completed five-shot dataset. Explicit factorial selection remains available. Numerical output provenance is retained in the pre-execution source snapshots; merging executed outputs into working notebooks only changed notebook execution metadata/output data.

[Run report](runs/deepseek-v4-flash/five-shot-jlens-heatmap/runtimes/58218428/REPORT.md), [completion pointer](runs/deepseek-v4-flash/five-shot-jlens-heatmap/COMPLETE.json), and [final audit](runs/deepseek-v4-flash/five-shot-jlens-heatmap/runtimes/58218428/FINAL_AUDIT.json). No model execution or commit was performed.
