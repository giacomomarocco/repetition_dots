# Five-shot square J-Lens word recurrence

## 2026-09-11 — implementation and CPU preflight

Objective: characterize recurring square J-Lens words/wordpieces across all 262
known facts at k=0 and k=20, preserving the exact historical five-shot addition
prompts. Select a typical successful trajectory objectively from the new native
responses. No model execution or Slurm submission has been authorized yet.

Inputs are `runs/deepseek-v4-flash/one-fact-addition-5shot-full/{prompts,results,run_config}.json`.
The source has 1,310 prompts; exactly 524 paired prompts are selected. Each pair
keeps its fact, addend, identifiers and five demonstrations. Greedy decoding uses
an eight-token limit, and correctness is the original strict integer parse of
the entire response. Historical correctness is retained only for comparison.

Actual tokenizer checks on every selected prompt confirm five positions at k=0
and 25 at k=20: final question token; every target filler token; `Answer`; `:`;
`<｜Assistant｜>`; `</think>`. There is no trailing space. Newline-containing
question/filler token pieces are preserved. This is 7,860 positions and 330,120
square-lens readouts at layers 0–41. The three categories are `age_facts`,
`atomic_facts` and `static_facts`. Existing capture runs inspected in this
workspace do not provide this complete five-shot ensemble; the applicable
square/full-prompt captures use the other prompt protocol.

Implementation:

- `filler/dsv4/five_shot_recurrence.py`: exact source reuse, token coverage,
  pairing, metadata preflight and source/model identity checks.
- `filler/dsv4/recurrence_runtime.py`: instrumented full-prefill capture, all 43
  selected-position residual layers on every TP rank, strict native outcomes,
  four-GPU scoring, and automatic process cleanup. The model server stops before
  scoring begins. Each GPU receives both conditions for its assigned facts.
- `filler/dsv4/recurrence_analysis.py`: exact token/group recurrence, paired
  fact bootstrap and overlap-based representative selection.
- `filler/dsv4/recurrence_export.py`: compact per-cell counts, CSV tables,
  PNG/PDF figures, integrity gates, executed notebook and completion marker.
- `notebooks/five_shot_square_recurrence.ipynb`: reads completed artifacts;
  cohort/category/token filters and exact top-ten inspection are rerunnable.
- `filler/dsv4/lens_positions.py`: explicit five-shot suffix mode; the existing
  `Answer`, colon and trailing-space requirement remains the default.

The square artifact is the pinned `camilablank/workspace-lenses` release at
`d740106d1e0f95456dc8718fba2895e9c8ffd6ef`, SHA-256
`8b010eef8b2b08efb1b07601e5203ff5d215b1fcae63704847fdf001e61e0efc`.
Preparation only inspects mmap metadata. Full artifact hashing, finite-matrix
checks and the layer-41 identity gate run on the allocated compute node before
model loading. Scoring uses FP32 stream means and transport, then the established
BF16 HF norm/head with FP32 log probabilities. Reference transport and the pinned
HF adapter are checked across all 42 layers. Native SGLang fused mHC/norm and
TP-vocabulary readout must agree with each prompt's first generated token and
candidate/top log probabilities (maximum error 0.15). All selected residuals
must agree across all four TP ranks. Returned hidden-state equivalence is also
checked on one complete prefill per condition; subsequent prompts avoid the
large redundant full-prompt hidden-state JSON response.

Ties use descending logit then ascending token ID, including ordinal target
ranks. All top-ten entries retain full-vocabulary log probabilities. Groups
strip surrounding whitespace and case-fold, deduplicate within each top-ten
list, and retain a complete vocabulary-to-group mapping. The top 20 textual
groups are selected by k20 filler persistence, then breadth and spelling, and
rescored over **all vocabulary variants** at every position. Missing top-ten
entries are never assigned zero probability.

Recurrence denominators are calculated within each prompt first. Question,
filler, answer-prefix and assistant-transition regions stay separate. The
2,000 bootstrap resamples (seed 42) share paired fact indices throughout.
Condition-specific summaries use that condition's native correctness; paired
differences and paired plots hold k20 cohort membership fixed for both k values.
Empty cohorts are recorded explicitly; bootstrap replicates without subgroup
members are omitted and the number retained is reported. Intervals are
descriptive and do not adjust for multiple comparisons or top-word selection.
The condition contrast changes filler in all demonstrations and the target.

Reproduction:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv-sglang/bin/python -m scripts.dsv4.five_shot_recurrence prepare
```

Prepared manifest:
`runs/deepseek-v4-flash/five-shot-square-recurrence/preflight.json`.
It records exact prompts/positions, source hashes, model shard sizes/mtimes,
software source pins and the launch/scoring contract. Any implementation/input
change requires rerunning prepare. Four-rank selected residual payloads total
approximately 44.3 GB before serialization overhead; scores and group masses
are compact compressed NumPy artifacts.

The concrete allocation command, **pending explicit user approval**, is:

```bash
salloc --account=m5258_g --qos=interactive --nodes=1 --gpus=4 \
  --constraint='gpu&hbm80g' --time=04:00:00 \
  srun --ntasks=1 --cpus-per-task=128 --cpu-bind=none --gpus=4 \
  bash /pscratch/sd/m/marocco/sandbox/mech_int/run_five_shot_recurrence_allocation.sh
```

The allocation-owned sequence captures all 524 prompts, stops the server,
scores on four GPUs, selects/rescores recurring groups, exports all tables and
figures, executes the notebook, logs results, then exits to release the
allocation. Failure stops owned child processes and leaves diagnostics and
partial artifacts. No further administrative approval should be needed inside
the allocation. Actual account, QOS, GPUs, node, job ID, UTC start/end/deadline
and launch command are recorded under the runtime directory.

For re-exporting already scored artifacts on an approved compute node:

```bash
.venv-sglang/bin/python -m scripts.dsv4.five_shot_recurrence export \
  --root runs/deepseek-v4-flash/five-shot-square-recurrence/runtimes/JOB-RUNTIME
```

Only after all capture, native-readout, square/reference, coverage and notebook
checks pass does a runtime publish `COMPLETE.json`. A complete run is immutable;
repeating export verifies it. The top-level pointer selects the completed run.
The current worktree contained substantial unrelated changes on arrival; those
were preserved, and no commit has been made.

Validation: the broader focused CPU suite passed 166 tests, including existing
native hooks, square-lens/reference fixtures, full-prompt coverage and port
allocator regressions. A small synthetic integration executes the actual
notebook cells and tests export checksums/publication; it does not substitute
for the pending real GPU gates. Shell syntax and Python compilation passed.
The local Python environment prints MUNGE socket messages at shutdown; the
checks exited successfully and no Slurm jobs were submitted. Final targeted
checks after the last implementation edits are recorded below.

Scientific results and the real executed notebook remain pending the approved
GPU run. Do not interpret synthetic test exports as experiment findings.

Final targeted verification: **53 tests passed in 26.91 seconds**, including the new exact group-mass test and both successful and failed capture lifecycle cleanup. The prior broader focused suite passed 166 tests. Notebook cell IDs are present; notebook execution was verified by the synthetic integration. Shell syntax, compilation and working-tree whitespace checks passed.

Final prepared configuration: `b0d4296587f3e4e7df89ec7b5f2d0dfaec752e4651fc7d3882ac0243b6ea0e1f`. Source and checkpoint metadata re-verification passed after preparation. No GPU allocation has been requested or submitted; real model/native/reference validation, scientific results, and the production executed notebook remain pending explicit Slurm approval.


## 2026-09-11T17:12:17.682348+00:00 — approved launch

User approved the prepared four-hour, four-A100-80GB workflow ("Go."). Rechecked all prepared source/checkpoint identities without changes. Submitted job **58202160** with `m5258_g`, public QOS `interactive`, constraint `gpu&hbm80g`, four GPUs and one 128-logical-CPU task. The allocation session runs the complete prepared sequence and releases its allocation on exit. Initial scheduler state: queued.

Allocation granted: node `nid008437`, account `m5258_g`, internal QOS `gpu_interactive`, four NVIDIA A100-SXM4-80GB GPUs (85,093,777,408 bytes each). UTC start 2026-09-11T17:16:46, end 2026-09-11T21:16:46. Runtime: `runs/deepseek-v4-flash/five-shot-square-recurrence/runtimes/58202160-5cc37d34`. Square artifact hashing/identity checks passed before model startup.


## 2026-09-11T17:50:23.204438+00:00 — capture complete; scoring recovery prepared

Job 58202160 captured all 524 prompts, then stopped the model server and automatically released its allocation when all four scoring workers rejected source drift. Concurrent confirmation-design edits changed `filler/dsv4/patching.py` and `scripts/dsv4/one_fact_patching.py` after launch. The capture hook, numerical lens code, tokenizer/model files and all other pinned inputs remained unchanged. No square-lens scores or scientific completion marker were published.

Native strict outcomes from this capture: **137/262 correct at k=0**, **179/262 at k=20**; **25 prompt-level correctness disagreements** with historical outcomes. These labels do not yet imply that native readout or J-Lens validation has passed.

Recovery uses an isolated source copy at `runs/deepseek-v4-flash/five-shot-square-recurrence/runtimes/58202160-5cc37d34/recovery-source`. The original `patching.py` was reconstructed and its complete SHA-256 matches the initial preflight exactly. AST comparison proves that digest, file-digest, atomic-write/sync and requested-score helpers are unchanged by the concurrent confirmation-design edits; the exact diff is saved in `recovery-audit/patching.diff`. The original launcher file was not fully recovered. Its current code reproduces the exact recorded capture launch command and keeps SGLANG_PORT absent; its changed whole-file hash is recorded separately as an audited controller-only change. No launch/model-execution function is invoked during recovery. All capture and numerical source hashes retain their original pins. The original capture configuration and all captures remain untouched.

Recovery manifest `132d70b7526a550ba56b128438a8f858f652b03f23d4ea3b37cae49b1ae6b47c` hashes the frozen code and separately records the one controller-only source adjustment. Recovery verification checks the original capture-plan hash before checking an adapted software-only copy; the adapted configuration is never used to relabel captures or scores. The complete 524-capture metadata preflight passed from this isolated copy. Five additional tests verify this separation, source-drift rejection and worker cleanup. The recovery will rerun all native/reference gates, score on four GPUs, rescore group masses, export, execute the notebook and release the allocation. No full model load is needed.

A **new 90-minute four-A100-80GB allocation requires explicit approval**, because the first allocation was released. Prepared command:

```bash
salloc --account=m5258_g --qos=interactive --nodes=1 --gpus=4 \
  --constraint='gpu&hbm80g' --time=01:30:00 \
  srun --ntasks=1 --cpus-per-task=128 --cpu-bind=none --gpus=4 \
  bash /pscratch/sd/m/marocco/sandbox/mech_int/runs/deepseek-v4-flash/five-shot-square-recurrence/runtimes/58202160-5cc37d34/recovery-source/run_five_shot_recurrence_recovery_allocation.sh \
  /pscratch/sd/m/marocco/sandbox/mech_int/runs/deepseek-v4-flash/five-shot-square-recurrence/runtimes/58202160-5cc37d34
```


## 2026-09-11T21:32:08.550654+00:00 — recovery deadline fix

Recovery allocation **58211848** started on nid008260 around 21:30 UTC after approval was received, but exited before scoring because the recovery harness incorrectly capped execution at the expired original-capture deadline (21:13:46 UTC). It released its allocation immediately. This was a recovery-controller bug; no model or scoring work ran. Fixed the harness to use the new approved allocation deadline for worker walltime checks while leaving the original capture runtime metadata unchanged on disk. Tests cover an expired capture deadline, an active recovery deadline, and rejection of a deadline from another job. All five recovery tests pass. Frozen harness manifest updated to `b2f69ce2a0782083447aabd5dc54fd9d05775c3921fdda93dc86bbd328e47c43`.


## 2026-09-11T21:42:26.272143+00:00 — completed capture and analysis

Run: [58202160-5cc37d34](/pscratch/sd/m/marocco/sandbox/mech_int/runs/deepseek-v4-flash/five-shot-square-recurrence/runtimes/58202160-5cc37d34/REPORT.md). Config `b0d4296587f3e4e7df89ec7b5f2d0dfaec752e4651fc7d3882ac0243b6ea0e1f`; 524 prompts, 330,120 readouts; native readout passed for all prompts, maximum log-probability error 0. Historical label disagreements: 25. Representative `pair-c2ac349e2130bbdad5bd` (overlap 0.477407).

Top five filler groups by persistence: '锁定' (0.3006; breadth 1.0000), 'ait' (0.2786; breadth 1.0000), 'walker' (0.2730; breadth 1.0000), '并能' (0.2677; breadth 1.0000), 'cheer' (0.2489; breadth 1.0000).

Allocation, GPU capacities, timestamps, exact launch, source hashes, native/reference checks, paired bootstrap definitions, complete prompts and output links are in the run provenance and report. These are recurring lens readouts, not generated sentences or evidence of causal computation. The k20-minus-k0 contrast changes filler length in all five demonstrations and the target, so it describes the complete prompt condition. Intervals are descriptive paired-fact bootstrap intervals (2,000 resamples, seed 42), not simultaneous or post-selection inference. Words/wordpieces strip surrounding whitespace and case-fold; they are not reconstructed words.


## 2026-09-11T21:47:42.330798+00:00 — final artifacts verified

Recovery job **58211966** ran on nid008236, m5258_g / gpu_interactive, four A100 80GB GPUs. UTC allocation start 21:34:29; scoring/export finished 21:42:26 and salloc released the allocation. All **524 native final-readout checks passed with maximum log-probability error 0.0**; all **42 square-layer reference gates** passed; **330,120 readouts** and exact masses for 20 textual groups are complete. Export produced **85 PNG/PDF pairs**, all cohort/category tables, paired intervals and the executed notebook. `RECOVERED.json` resolves the retained original `FAILED.json` chronologically.

Leading filler wordpieces: 锁定 30.06%, ait 27.86%, walker 27.30%, 并能 26.77%, cheer 24.89% persistence; all five enter the top ten somewhere in every k20 prompt. These are lens readouts, not generated text. Selected representative `pair-c2ac349e2130bbdad5bd` has mean top-ten overlap 0.4774074882. It asks the atomic number of gadolinium: A=64, X=62, expected 126; k0 response `64` (wrong), k20 response `126` (correct).

Static plots have missing CJK glyphs because the node lacks a matching scalable font. Added [a self-contained Unicode browser viewer](/pscratch/sd/m/marocco/sandbox/mech_int/runs/deepseek-v4-flash/five-shot-square-recurrence/runtimes/58202160-5cc37d34/recurrence_viewer.html) using the completed representative’s exact scores, with vertically aligned shared positions, separate filler columns, full prompts, exact group/numeric probabilities, and click/keyboard top-ten inspection. Its JavaScript syntax and embedded data/grid dimensions were checked; it requires no model computation or network. This rendering limitation is documented in the report.

Final report: [58202160-5cc37d34](/pscratch/sd/m/marocco/sandbox/mech_int/runs/deepseek-v4-flash/five-shot-square-recurrence/runtimes/58202160-5cc37d34/REPORT.md). Full checksums and numerical provenance remain in the completed runtime. No commits were made; unrelated worktree edits remain intact.
