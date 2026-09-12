# One-fact top-object Logit Lens heatmap

## 2026-09-03: analysis and plotting path

Objective: render a 2×3 position-by-layer heatmap for one-fact addition. The
columns are the fact (`A`), addend (`X`), and sum (`A+X`). The first row reports
the fraction of examples for which that object is the vocabulary-wide top
token; the second reports the fraction for which it is the top token after
restricting the vocabulary to tokens that decode as canonical nonnegative
integers. Leading token whitespace is ignored, while signs, decimals,
punctuation, separators, and leading-zero forms are excluded.

The existing discovery data contain 96 manifests, 4,128 captures, and 177,504
scored rows at
`runs/deepseek-v4-flash/logit-lens/57804608/filler-grid-expanded/`. They already
record the unrestricted argmax, but not the numeric-only argmax. Raw 43-layer
captures and the native mHC readout weights remain available, so no model
server or new forward passes are needed.

Implementation:

- `scripts/dsv4/add_numeric_argmax.py` replays the saved residuals through the
  native mHC collapse, final RMSNorm, and only the integer-token rows of the LM
  head. It checks every output row against the manifest cell, position, and
  layer before writing an augmented JSONL file.
- `scripts/dsv4/plot_top_object_heatmap.py` aggregates a selected filler length
  (default 20) over examples and renders all six panels on one shared [0, 1]
  color scale. It also saves the exact plotted rates and denominators as JSON.
- `filler/dsv4/lens.py::project_selected_logits` was checked against column
  selection from the full native projection on synthetic tensors.

Validation completed on the login node: imports compiled under the project
Python 3.12 environment and `tests/test_top_object_heatmap.py` passed (3 tests).
The login node's system Python is 3.6; project commands must use
`.venv-sglang/bin/python`.

### Execution

With user approval, pending regular-QOS batch job 57912026 was cancelled before
it started and replaced with interactive-QOS GPU allocation 57912051 on
`nid001065`. The numeric projection completed all 177,504 rows and found 1,000
canonical integer tokens. Outputs are under
`runs/deepseek-v4-flash/logit-lens/57804608/top-object-heatmap/`; the initial
figure and exact aggregate JSON use filler length 20. The reproducible,
interactive view is `notebooks/one_fact_top_object_heatmap.ipynb`.

### Correct/wrong split

The same filler-length-20 panels were subsequently stratified by the recorded
`clean_correct` outcome, without rerunning the projection. Of the 96 examples,
83 are correct and 13 are wrong; these denominators are constant at every
position/layer cell within their respective figures. Both stratified plots use
the original shared [0, 1] scale. The notebook now displays all, correct, and
wrong cohorts in sequence and accepts any of those values through its `cohort`
argument. Separate PNG and aggregate-JSON artifacts have `_correct` and
`_wrong` suffixes in the output directory. The small wrong cohort should be
interpreted cautiously.

## 2026-09-10: J-Lens notebook and saved-grid scoring command

Objective: provide a J-Lens version of the code in
`notebooks/one_fact_top_object_heatmap.ipynb`. Added
[`notebooks/one_fact_top_object_jlens_heatmap.ipynb`](notebooks/one_fact_top_object_jlens_heatmap.ipynb),
with the same 2×3 panels, shared [0, 1] color scale, dotted-grid toggle, selectable
filler length, and separate all/correct/wrong figures. Figure titles include
cohort denominators. The original notebook was left unchanged.

The new notebook uses only zero-based layers **19–39**, the fitted band in the
published J-Lens. It reuses the validated `filler/dsv4/jlens.py` transport and
HF readout described in [the J-Lens compatibility log](DEEPSEEK_V4_JLENS.md),
including full-stream flattening, FP32 transport, BF16 HF RMSNorm rounding, and
no second mHC collapse. Correctness cohorts retain the saved model
`clean_correct` labels. Both full-vocabulary and canonical-integer argmaxes are
computed from the same complete J-Lens logits; exact ties choose the smallest
token ID. Target matching remains by exact single token ID.

Preparation is implemented in `filler/dsv4/jlens_top_object.py`, with the thin
entry point `python -m scripts.dsv4.prepare_jlens_top_object`. It reads the
existing grid manifests directly, preserving their target IDs and example
metadata. It checks position coverage, duplicate example/pass IDs, capture
metadata, fitted-layer shapes/dtypes, finite residuals/readouts, and target
tokenization. The production command verifies the pinned lens SHA-256.
Jacobian matrices are converted to FP32 once (about 5.6 GB), while raw captures
and complete vocabulary readouts are processed in batches (default 64).
Only the readout tensors and saved Jacobians are loaded, with no model server,
new forward passes, or fitting.

Default output is
`runs/deepseek-v4-flash/jlens-top-object-heatmap/jlens_rows_with_numeric_argmax.jsonl`.
Rows contain J-Lens argmax IDs and target IDs, without carrying over native-lens
logits or ranks. Each run uses a fresh output directory, records its command,
node/job ID, source/manifest hashes and pinned fit, and retains failed runs.
The notebook requires `COMPLETE.json`, the expected row count, explicit
`lens="jlens"` labels, and exactly the fitted layer band per captured position.
Partial scoring files cannot be mistaken for completed inputs.

### Validation and execution status

Lightweight manifest/file preflight on `login22` found all **4,128 captures**,
covering **96 examples at each of filler lengths 0, 5, 10, and 20**, for
**86,688** planned J-Lens rows. Selecting only length 20 gives 44,352 rows.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-sglang/bin/python -m scripts.dsv4.prepare_jlens_top_object --preflight

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg \
  .venv-sglang/bin/python -m pytest -q \
  tests/test_jlens_top_object.py tests/test_top_object_heatmap.py \
  tests/test_dsv4_jlens.py
```

Result: **39 passed in 34.43 seconds**. New tests exercise manifest-to-capture
scoring and both argmaxes, batch sizes 1/3/4, known cohort rates, exact ties,
output preservation, missing/mismatched/nonfinite captures, malformed
manifests, missing requested filler lengths, and invalid numeric targets.
All notebook code cells compiled and executed against 924 synthetic rows
(two examples × 22 positions × 21 layers), including all three cohorts and
the grid toggle. The synthetic PNG was visually inspected. Notebook JSON
structure was checked directly; `nbformat` is not installed in this environment.
Synthetic inputs/figures remained under `/tmp`, and the delivered notebook
contains no fabricated result outputs. `git diff --check` passed.

The environment emitted the existing NVML/MUNGE diagnostics during Torch
imports/tests; both tests and preflight exited zero. No scheduler actions or
real matrix scoring were performed. Real-data plots remain pending the
prepared command on an explicitly approved compute allocation:

```bash
srun --ntasks=1 --cpus-per-task=16 --cpu-bind=cores \
  env OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 \
  .venv-sglang/bin/python -m scripts.dsv4.prepare_jlens_top_object \
  --device cpu --threads 16 --batch-size 64
```

Run from the repository root inside an approved allocation. Add
`--filler-lengths 20` for that subset, or select an allocated GPU with
`--device cuda:0`. The scoring command rejects execution on a login node.
After successful preparation, run the notebook using its default `RUN` path;
if `--output-dir` was supplied, update `RUN` accordingly. Changes remain
uncommitted for user inspection.

## 2026-09-10 02:21 UTC: approved full-grid J-Lens computation

The user approved proceeding with the computation. Prepared
`run_jlens_top_object_allocation.sh` to perform scoring, validation and export
as one automatic compute-node sequence, releasing the allocation when the
command exits. Added `scripts/dsv4/export_jlens_top_object.py` to check every
row against its source manifest, execute the notebook's seven code cells,
and export PNG/PDF/JSON for each filler length and correctness cohort.
Its preflight passed on 924 synthetic rows, including three embedded notebook
plots and three PNG/PDF pairs; those synthetic artifacts remain under `/tmp`.
Shell syntax and source compilation passed before submission.

Submitted the approved CPU run using account `m5258`, public QOS `interactive`,
one node, a four-hour limit, and a 16-thread bound step:

```bash
salloc --account=m5258 --qos=interactive --nodes=1 --constraint=cpu \
  --time=04:00:00 --job-name=jlens-top-object \
  srun --ntasks=1 --cpus-per-task=16 --cpu-bind=cores \
  bash /pscratch/sd/m/marocco/sandbox/mech_int/run_jlens_top_object_allocation.sh
```

Job **58137198** started **2026-09-10 02:21:54 UTC** on **nid004149**, with
deadline **06:21:54 UTC**, actual account `m5258`, QOS `interactive`, and
partition `urgent_milan_ss11`. There are no allocated GPUs. Scheduler metadata,
timestamps, and streamed workload output are retained in
[`runs/deepseek-v4-flash/jlens-top-object-execution/58137198/`](runs/deepseek-v4-flash/jlens-top-object-execution/58137198/).
The full 86,688-row scoring and export are running; completion and scientific
results remain pending validation.

## 2026-09-10 02:43 UTC: completed full-grid computation

Job **58137198** completed successfully and released its allocation at
**02:43:47 UTC**. Scoring took **1,267.55 seconds** (21.13 minutes); the full
compute-node sequence, including export, took 21 minutes 50 seconds.

Validated **86,688 unique rows**, covering all **4,128 captures** and all
21 fitted layers. Every output row agrees with its source manifest's example,
position, pass, targets and correctness label. Source/manifest integrity
checks passed, and the restricted vocabulary contains **1,000** canonical
integer tokens. All 12 aggregate arrays have the expected shapes, rates in
[0, 1], and constant cohort denominators:

| Filler length | All | Correct | Wrong |
|---|---:|---:|---:|
| 0 | 96 | 80 | 16 |
| 5 | 96 | 72 | 24 |
| 10 | 96 | 72 | 24 |
| 20 | 96 | 83 | 13 |

Exported **12 PNG/PDF/JSON sets** and an executed notebook copy. All seven
notebook code cells ran successfully; the three default length-20 plots were
also saved into the main notebook. The all/wrong PNGs were visually inspected.
At the answer prefix, layer 39, the sum is the J-Lens argmax for **72/96**
examples: **69/83** in the correct cohort and **3/13** in the wrong cohort.
These counts are the same for the full and numeric-restricted vocabularies
at that particular position/layer; they describe this saved discovery grid.

Artifacts:

- [Notebook with computed plots](notebooks/one_fact_top_object_jlens_heatmap.ipynb)
- [All rows](runs/deepseek-v4-flash/jlens-top-object-heatmap/jlens_rows_with_numeric_argmax.jsonl)
- [Validation and row checksum](runs/deepseek-v4-flash/jlens-top-object-heatmap/EXPORT_VALIDATION.json)
- [Length-20 all-example heatmap](runs/deepseek-v4-flash/jlens-top-object-heatmap/top_object_jlens_k20_all.png)
- [Complete artifact directory](runs/deepseek-v4-flash/jlens-top-object-heatmap/)

The projection sources were unchanged throughout the run. `git diff --check`
passed; changes remain uncommitted.

## 2026-09-10: explicit Answer-token positions and k = 0 comparison

Request: expose the `Answer`, `:`, and trailing-space positions separately in
`notebooks/one_fact_top_object_heatmap.ipynb`, alongside k = 20 and no-filler
k = 0 figures. The original discovery grid saved only the final space under
`answer_prompt`; it did not capture the `Answer` and `:` residuals. For example,
one k = 20 manifest places `filler_19` at 81 and `answer_prompt` at 84. The
historical `last_question` label refers to `</think>` immediately before the
filler/answer suffix in these rendered prompts.

Completed without model inference:

- Updated the notebook with separate all/correct/wrong figures at k = 20 and
  k = 0, explicit trailing-space labeling, cohort counts, and an explicit
  missing-Answer/colon message until the expanded capture is available.
  Fixed the existing concatenated `stats_wrong = ...stats_correct = ...` line.
  Executed all 10 code cells using the project Python and Agg backend; six
  figures are embedded in the notebook. No nbclient package was installed;
  code cells were evaluated locally with stdout and figure outputs recorded.
- Original k = 0 cohorts: **96 total, 80 correct, 16 wrong**; k = 20 remains
  **96 total, 83 correct, 13 wrong**. Saved k = 0 PNG and exact aggregate JSON
  files as `top_object_heatmap_k0_{all,correct,wrong}.{png,json}` in
  `runs/deepseek-v4-flash/logit-lens/57804608/top-object-heatmap/`.
- `filler/dsv4/top_object_heatmap.py` now orders and labels `answer_word`,
  `answer_colon`, and `answer_prompt` after the fillers, including k = 0.
  Existing J-Lens plots continue to accept their original position labels.

Prepared new capture (not submitted):

`filler/dsv4/answer_token_heatmap.py` reuses the full-prompt campaign hook and
its serial, uncached launcher. The exact original rendered k = 0 and k = 20
prompts/token IDs were verified with the local tokenizer: 96 examples each,
**192 full-prompt forwards**, **2,688 selected positions**, and **115,584
43-layer rows**. This recaptures all displayed positions in the same runtime,
so every plot uses its own complete-prompt greedy-answer correctness cohort;
no old and new activations or correctness labels are silently mixed.

The first example at each length must pass native final-layer equivalence
(all four TP ranks, returned hidden state, argmax, and candidate/top-token
logprob error <= 0.15). Every projected final answer position must match that
example's native argmax. Captures and acknowledgements use the existing
campaign integrity checks. Projection uses the native mHC collapse, RMSNorm,
and complete vocabulary head; numerical argmax selects canonical integers
from those same logits. The server is stopped after capture, followed
immediately by GPU projection and six PNG/JSON exports. The allocation exits
when the automated workload ends. Only a complete, validated run publishes
`runs/deepseek-v4-flash/answer-token-heatmap/COMPLETE.json`; the notebook then
selects its rows automatically when rerun. Partial output cannot be selected.

CPU validation: **63 tests passed in 33.37 s**, covering new token selection,
zero-filler ordering, position-dependent scoring, partial/nonfinite captures,
native hooks/readout, and the existing launch/port allocator regressions.
Both shell launchers passed `bash -n`. Import shutdown printed sandbox MUNGE
socket diagnostics; preparation and tests exited successfully. No Slurm
submission or model load was performed.

Prepared inputs and verification:
[`preflight.json`](runs/deepseek-v4-flash/answer-token-heatmap/preflight.json),
[`verification.json`](runs/deepseek-v4-flash/answer-token-heatmap/verification.json).
Source/checkpoint fingerprints are checked again before model loading.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  TOKENIZERS_PARALLELISM=false MPLBACKEND=Agg \
  .venv-sglang/bin/python -m scripts.dsv4.answer_token_heatmap prepare

# Requires explicit user approval under AGENTS.md:
salloc --account=m5258_g --qos=interactive --nodes=1 --gpus=4 \
  --constraint='gpu&hbm80g' --time=02:00:00 \
  srun --ntasks=1 --cpus-per-task=128 --cpu-bind=none --gpus=4 \
  bash /pscratch/sd/m/marocco/sandbox/mech_int/run_answer_token_heatmap_allocation.sh
```

Outstanding: obtain allocation approval, execute the prepared workload, verify
the completed exports, and refresh notebook outputs to include the two newly
captured positions. Changes remain uncommitted for user inspection.

### Follow-up correction: k = 20 already exists in full-prompt patching captures

The statement above that the Answer/colon residuals require new captures was
too broad: it applies to the original heatmap capture grid. The completed
`runs/deepseek-v4-flash/one-fact-patching-discovery/` campaign saved **all prompt
positions at all 43 layers** for all 96 unpatched k = 20 baselines. Every
baseline's input IDs exactly match the original rendered heatmap prompts.
All 384 rank files exist, and acknowledgements cover Answer/colon positions
82/83 (with the final space at 84). For example, the rank-0 capture of
`discovery-filler-0743614aafcd1ceb0a07:00` has 119,781,666 bytes and positions
0–84. These tensors were overlooked when the first rerun was scoped.

Revised `answer_token_heatmap.prepare` and `run` to reuse these complete k = 20
baselines. CPU preflight checks completed-campaign integrity, journal digests,
exact prompt IDs, checkpoint/tokenizer identity, unpatched complete-prompt
controls, acknowledgement/response checksums, and rank file presence/stat
fingerprints. Capture tensor checksums and native readout are checked on the
compute node. k = 20 uses the saved baseline outputs for its cohorts:
**83 correct and 13 wrong**. No historical files are modified.

Only **96 new k = 0 forwards** remain necessary. The final projection still
produces 115,584 rows covering 24 positions for k = 20 and four for k = 0.
All plotted positions within a length use the same captures/cohort; k = 20
comes from the historical patching runtime and k = 0 from the new runtime.
Prepared manifest hash: `9f45b07029099ff45436b185a05887eafbe686cd61b6697d0440f8f2c2f3967f`. Targeted tests after this
revision: **10 passed in 9.70 s**, including equality of scoring selected-row
captures and full-prompt captures. Preflight completed successfully. The
allocation command remains prepared but unsubmitted, pending user approval.

### Approved execution: job 58168705

The user explicitly approved execution of the revised workload. Submitted the
prepared two-hour interactive allocation with account `m5258_g`, four GPUs,
constraint `gpu&hbm80g`, and an immediate one-task `srun` step using 128 CPUs
and `--cpu-bind=none`. Slurm granted job **58168705**. The controller runs
without waiting for additional agent actions. Its persistent output is
`runs/deepseek-v4-flash/answer-token-heatmap/allocation-58168705.log`.
Native validation, k = 0 capture, historical k = 20 readout, and exports run
automatically; the allocation exits at completion or failure.

Allocation validation passed on **nid008344**: account `m5258_g`, internal
QOS `gpu_interactive`, constraint `gpu&a100&hbm80g`, four NVIDIA
A100-SXM4-80GB GPUs with 85,093,777,408 bytes each. Slurm start/end times
(queried with TZ=UTC) are **2026-09-10 18:38:10–20:38:10 UTC**. Runtime:
`runs/deepseek-v4-flash/answer-token-heatmap/runtimes/58168705-18233e35/`.
Weight loading began at 18:40:11 UTC; all 48 shards were read by 18:40:21 UTC.
The frozen preflight hash was revalidated before submission and on the node.

### First execution stopped at readout validation; retry prepared

Job 58168705 loaded the model successfully (server ready at **18:53:00 UTC**)
and saved the first k = 0 capture by **18:55:58 UTC**. The four TP residuals
and the server-returned hidden state match exactly, and the argmax agrees.
The CPU reference readout nevertheless differs from one native top-token
log-probability (token ID 26) by **0.5002041**; the other tested values differ
by about 0.0002041. This exceeds the unchanged 0.15 validation threshold, so
the controller stopped, preserved the capture/response/report, and Slurm
released the allocation at about **18:56 UTC**. See the failed runtime's
`validation_k0.json`, `FAILED.json`, and `passes/k0-000/`.

Prepared `project_sglang_logits`: use the pinned serving implementation's
fused mHC kernel and RMSNorm, with TP=4 vocabulary partitions and M=1 head
GEMMs matching single-token generation. This avoids relying on a CPU
reconstruction for native equivalence. The existing Torch reference and
other experiments' default validation behavior remain unchanged; this
workflow explicitly selects the CUDA projector. The error threshold has
**not** been loosened. The same projector is used for every expanded heatmap
position and layer, so validation and plotting use the same arithmetic.

The retry includes the preserved first capture as a mandatory GPU readout
probe **before another full model load**. A failed probe exits immediately.
If it passes, the automated model load, fresh k = 0 captures, saved k = 20
projection and exports continue. The first allocation consumed approximately
18 minutes; the planned retry is capped at 100 minutes, within the original
two-hour total allocation budget. CPU tests for the optional projector and
existing native validation: 56 passed in 32.12 s before the final M=1 head
adjustment; the directly affected tests are rerun for that adjustment.

### Retry job 58169491: saved-capture readout discrepancy resolved

Submitted the corrected workload with a **100-minute** cap; job 58169491 was
granted on **nid008252**. Account `m5258_g`, QOS
`gpu_interactive`, four A100-SXM4-80GB GPUs. Slurm start/end
(TZ=UTC): **2026-09-10T19:07:02–2026-09-10T20:47:02**.
Runtime: `runs/deepseek-v4-flash/answer-token-heatmap/runtimes/58169491-f5aaee6f/`.

Before model loading, the actual CUDA serving readout on the preserved first
k = 0 residual matched **all 20 recorded candidate/top logprobs exactly**
(maximum error **0.0**), with exact TP residual/returned-state agreement and
the same argmax. See `preload_readout_validation.json`. This resolves the
CPU-reference rounding discrepancy without changing the 0.15 gate. The
automated controller then began model loading. Directly affected tests after
the M=1 head adjustment: **14 passed in 15.33 s**. Frozen configuration:
`cb45280108a018a38836699756aad6b316734d7f60b504f35b4448fb9410af93`.

### Completed: expanded k = 0 and k = 20 heatmaps

Job **58169491** completed at **2026-09-10 19:32:11 UTC**, and `salloc`
returned exit code 0 and relinquished the allocation. Both allocations are
released. All **96 fresh k = 0** examples were captured; all **96 historical
k = 20** baselines were reused. The expanded data contain **115,584 rows**:
43 layers × 96 examples × (4 k = 0 positions + 24 k = 20 positions).
Every figure includes separate `answer_word` (`Answer`), `answer_colon` (`:`),
and `answer_prompt` (space) positions.

Native equivalence for the fresh k = 0 and historical k = 20 validation
examples passes with **maximum candidate/top logprob error 0.0**, exact
returned-state agreement, and exact agreement across all four TP ranks.
Every example's final-layer argmax was checked against its own native
generated answer; all **192 checks passed**. Cohort coverage is constant
at every position/layer: k = 0 has **80 correct / 16 wrong**; k = 20 has
**83 correct / 13 wrong** (96 total each). These counts match the original
cohort counts. The completed JSONL SHA-256 was independently verified after
export: `7b0e79ea1ea3a19570198189c3a2282f07a21c3631206503a211dbc14bc6a0cf`.

Final output directory:
`runs/deepseek-v4-flash/answer-token-heatmap/runtimes/58169491-f5aaee6f/`.
It contains six `top_object_heatmap_k{0,20}_{all,correct,wrong}.{png,json}`
exports, the expanded JSONL, captures/references, validation reports, runtime
metadata and `COMPLETE.json`. The root completion pointer selects this
validated output for the notebook.

Updated and executed `notebooks/one_fact_top_object_heatmap.ipynb`: all ten
code cells ran, six figures are embedded, no error outputs remain, and both
k = 0 and k = 20 figures were visually checked. The notebook retains the
dotted-grid toggle and explicit provenance for the later k = 20 captures
and native CUDA readout. The local sandbox execution/image service became
unavailable during the run; final reads, notebook rendering and inspection
used the working approved host path. No system configuration was changed.

Comparison limitation: this refresh changes both the k = 20 capture source
(original truncated-prompt runs → later complete-prompt baselines) and the
readout arithmetic (Torch reference → exact serving CUDA kernels). Some
shared cells therefore differ from the original plots; the largest
all-example fraction difference is **0.1770833** in the numeric-X panel.
The two causes were not separately isolated, so the differences should not
be attributed to either alone. Per-panel comparison counts and maxima are
in `original_k20_comparison.json`. Original figures/data were preserved.

User monitoring preference: avoid frequent worker/GPU checks during model
loading; report meaningful milestones. After that correction, monitoring
used occasional controller-log reads without further worker/GPU polling.
Requested notebook work is complete. Changes remain uncommitted.


## 2026-09-11: repair top-object J-Lens notebook and use all square layers

User requested fixing `notebooks/one_fact_top_object_jlens_heatmap.ipynb` and
switching its rectangular lens to the square lens across the full layer range.
The saved traceback was Matplotlib's `text.usetex=True` trying to invoke a missing
`latex` executable. Plot creation/rendering now runs inside a local
`plt.rc_context({'text.usetex': False})`, preserving the caller's global setting.

The notebook now reads the small completed comparison summaries from
`runs/deepseek-v4-flash/jlens-comparison/58185137/`, through the existing
`load_completed` guard. It uses `square_top1` and `square_numeric_top1` at every
zero-based layer **0–41**. The six-panel A/X/A+X layout, shared 0–1 color scale,
cohort selectors and dotted-grid toggle remain available. Explicit final-layer
ticks include 41. Last-question, every filler token, Answer, colon and space
positions are checked and displayed separately. All target/position/layer grids,
rates and saved cohort denominators are validated before plotting; missing or
duplicate summary cells are rejected. Available complete-prefix lengths are
**0 and 20**; historical k=5/k=10 coverage is documented separately.

Removed empty cells and the duplicate hard-coded correct-cohort plot, replaced
stale rectangular-only preparation instructions, and refreshed notebook outputs.
Optional aggregate export targets `notebooks/plots/`. The existing accepted
scoring artifacts were read only; no model work or Slurm actions were needed.

Validation used `.venv-sglang/bin/python`, `MPLBACKEND=Agg`, a temporary
`MPLCONFIGDIR`, and OMP/MKL/OpenBLAS thread counts of one. Executed all **7 code
cells**, rendered all **6** combinations of k=0/k=20 and all/correct/wrong,
checked the six panels per figure, all 42 columns, shared color limits, final
tick, prefix positions, and denominators (k=20 correct/wrong 83/13). Started
with global `text.usetex=True`: every figure rendered without LaTeX and the global
setting remained true afterward. Missing-layer and duplicate-summary negative
checks passed. The notebook embeds the three default k=20 figures and contains
no error outputs. `git diff --check` passed. Changes remain uncommitted.


### 2026-09-11: restore requested LaTeX rendering

User clarified that LaTeX is desired and works in `addition_accuracy.ipynb`.
The earlier local `text.usetex=False` workaround is superseded. Root cause:
the J-Lens notebook selected **NERSC Python**, whose executable PATH lacked
TeX Live; `addition_accuracy.ipynb` selects **mech-int (sglang)**. That existing
kernel's configuration prepends
`/global/common/software/nersc9/texlive/2024/bin/x86_64-linux` to PATH. The user's
Matplotlib configuration requests `text.usetex=True` and Computer Modern.
`module show texlive/2024` confirmed the same binary directory; the installed
LaTeX and dvipng executables are available there. No LaTeX installation was missing.

Matched the notebook kernelspec to addition_accuracy and added conditional
setup of that existing TeX Live directory in the current kernel's PATH, so an
already-open NERSC Python session also works when its setup cell is rerun.
Rendering explicitly uses `text.usetex=True`, retains configured fonts and
escapes underscores in prompt-position labels. No shared installation, global
Matplotlib configuration or kernel configuration was changed.

Validation started without latex on PATH, loaded the user's Matplotlib rc file,
and executed all seven code cells under project Python 3.12.11. All six cohort
figures (k=0/k=20, all/correct/wrong) rendered with real LaTeX; all Matplotlib
Text objects had usetex enabled. Checked six panels and all 42 layers per figure.
Used the Agg backend, a temporary TeX/Matplotlib cache and one math thread.
Three refreshed LaTeX-rendered k=20 figures are embedded. No error outputs remain.


### 2026-09-11: sparse filler labels and TeX font lookup recovery

`notebooks/one_fact_top_object_jlens_heatmap.ipynb` labels filler positions
0, 3, 7, 11, 15, and 19, while retaining every data row and all non-filler labels.
Axis titles use 16 pt and ticks use 14 pt. Figure height scales with position
count to prevent adjacent answer-prefix labels from overlapping.

The user reported `cmr10.tfm` missing when plotting after the label edit.
The initial validation used an isolated Matplotlib configuration without loading
the user's rc file, so it did not validate the configured Computer Modern font.
The font exists in the installed NERSC TeX Live 2024 tree. Setup now prepends
that complete installation even when latex/dvipng are already discoverable,
checks kpsewhich too, and probes Matplotlib's actual cmr10.tfm lookup. On lookup
failure, it resets the cached LuaTeX helper, if present, and retries. This handles
an existing kernel whose lookup helper inherited an earlier PATH. The exact
state of the user's failing kernel was not available for inspection.

Validation used project Python, Agg, one math thread, a temporary cache, and
explicitly loaded the user's existing Matplotlib rc file. A controlled stale
LuaTeX helper reproduced missing cmr10.tfm; executing setup recovered lookup in
the same process, and repeating setup succeeded. Rendered and embedded all three
k=20 cohort figures with real LaTeX and configured serif fonts; checked all ten
visible position labels and verified their rendered bounding boxes do not
overlap. No model execution or Slurm changes. Rerun from the notebook setup cell
to apply the repair to an already-open kernel.


### 2026-09-11: verify the reported NERSC Python kernel and inline output

The user reported the font error again and supplied a full traceback identifying
NERSC Python 3.13.15 / Matplotlib 3.11.1 under
`/global/common/software/nersc/pe/conda-envs/26.8.0/python-3.13/nersc-python`.
Earlier Python 3.12 / Matplotlib 3.10 validation did not establish compatibility
with that kernel. Inspected the exact installed font-lookup code; the repaired
notebook rendered successfully with that interpreter and the user's rc file.
Then executed all notebook cells using nbclient with a KernelSpec explicitly
launching that interpreter through ipykernel, using the real inline backend.
All three plot cells emitted PNG image data (base64 lengths 482752, 482832,
347856); no error outputs occurred. Saved these actual executed notebook outputs.

The user's `<Figure size 1920x960 with 7 Axes>` matches the earlier 16x8 figure
at 120 dpi; the saved k=20 function uses 16x12. This suggests stale cell source or
function definitions in the open tab, but that live kernel is not accessible and
the cause there remains unconfirmed. Instructed the user to reload the notebook
from disk before restarting the kernel and running all cells; simply rerunning
an already-open tab need not pick up external file changes. Project guidance now
requires the selected notebook environment and real inline image output checks.


### 2026-09-11: bypass the failing font lookup during rendering

User confirmed reload/restart still fails, so the stale-tab explanation was
insufficient. Replaced LuaTeX-helper restart logic in the notebook with a
cached resolver that runs the installed TeX Live kpsewhich by absolute path,
without inherited TeX font-search overrides. The resolver is patched into
Matplotlib dviread only while rendering each PNG. The notebook displays those
PNG bytes explicitly and closes the figure in finally, preventing deferred
Jupyter rendering from reverting to the failing default lookup. LaTeX,
configured fonts, larger labels, sparse filler tick labels, and all data rows
are preserved. No global Matplotlib or TeX installation changes were made.

Regression used the exact NERSC Python 3.13.15 / Matplotlib 3.11.1 Jupyter kernel
and the user's rc file. Before plotting, forced the default LuaTeX lookup to
return no font and verified a real savefig raised the same cmr10.tfm error.
Kept that broken default lookup active while executing all three heatmaps:
each produced an inline PNG. Confirmed afterward that the default lookup still
failed, proving successful rendering used the scoped resolver. Saved the actual
three image outputs (base64 lengths 482752, 482832, 347856). Test injection was
not saved into notebook source. An initial IPC kernel startup timed out; the
working local TCP kernel connection completed the regression. The user's live
kernel remains inaccessible; provided a cell that executes saved notebook code
directly to apply the repair without depending on tab reload state.


### 2026-09-11: numerical-only default and optional full-vocabulary row

Updated both `notebooks/one_fact_top_object_jlens_heatmap.ipynb` and
`notebooks/one_fact_top_object_heatmap.ipynb` at the user's request. Each setup
cell now defines `SHOW_TOP_TOKEN = False`; set it to True and rerun the setup
and plotting cells to restore the full-vocabulary row above the numerical row.
Direct calls also accept `show_top_token=True`, with False as the default.
All cohort plot calls pass the toggle. Figure height follows the number of
rows and prompt positions; decoder labels stay on the bottom visible row.
The returned aggregates retain both scopes and all captured positions/layers.

The ordinary Logit Lens notebook also uses the J-Lens notebook's existing
scoped direct-kpsewhich font resolver and explicit PNG display, with escaped
underscores under LaTeX. It retains the user's Matplotlib rc fonts and TeX
setting. No model execution, Slurm action, or result-data changes were needed.

Validation: executed both notebooks with nbclient in their selected NERSC
Python 3.13.15 / Matplotlib 3.11.1 kernel using the real inline backend, the
original Matplotlib rc file, and one OMP/MKL/OpenBLAS thread. Confirmed real
cmr10.tfm lookup, LaTeX, and configured Computer Modern. Saved three default
J-Lens PNGs and six default ordinary Logit Lens PNGs, with no error outputs.
Additional unsaved checks rendered both row modes at k=0/k=20, checked three
versus six panels, numerical/full-vocabulary matrix equality, 0–1 color limits,
complete answer-prefix coverage, 42 J-Lens / 43 Logit Lens layers, and bottom
axis labels. Visually inspected the k=20 default layouts and restored J-Lens
two-row layout; labels are readable and unclipped. The user's existing live
kernel was not accessed. Reload saved notebooks before rerunning an open tab.
Changes remain uncommitted.
