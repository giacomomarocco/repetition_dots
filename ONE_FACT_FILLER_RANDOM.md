# Independently randomized filler activations

**Completed:** all 1,272 forwards and scientific integrity checks passed; allocation released. Reporting recovered from frozen sources after concurrent live-source edits.

[Random results](runs/deepseek-v4-flash/one-fact-patching-filler-random/analysis/REPORT.md) · [Repeat comparison](runs/deepseek-v4-flash/one-fact-patching-filler-random/comparison/REPORT.md) · [Recovery provenance](runs/deepseek-v4-flash/one-fact-patching-filler-random/report-recovery/validation.json).

## 2026-09-12 — implementation and CPU preparation

Objective: test whether later filler residuals can be replaced by independent
random directions after cutoffs 0–5, preserving every destination's clean full
mHC tensor L2 norm. Retain all 96 discovery prompts with 20 fillers, including
baseline errors. Replace fillers j+1 through 19 at every post-block layer 0–42;
the answer prefix evolves through a fresh full-prompt forward pass.

The new `prepare --design filler-random` CLI design fixes one realization and seed
42. Historical CLI draw defaults remain five. A SHA-256-derived seed for each
prompt/layer/absolute-token position initializes a local CPU Torch generator.
Directions and normalization use float32 before casting to the clean residual
dtype; zero norms produce zeros, nonfinite values fail, and cast norm error must
not exceed 0.5%. All cutoff conditions sharing a runtime baseline reference the
same checksummed bank. All four ranks receive the same stored tensor. Restarted
work uses fresh runtime baselines and banks with the same position seeds; completed
trials retain their original baselines and banks.

New captures include the last question token, all 20 fillers, `Answer`, `:`, and
the trailing space, separately, at all 43 layers. Preparation validates actual
token IDs/text and saves this coverage in the manifest. Baselines retain the full
prompt. Native mHC readout, raw logits, identity controls, and final-layer-only
invariance retain the existing score tolerances; replacement tensors must agree
exactly. Bank reconstruction and saved tensor checks supplement hook acknowledgements.

Uninterrupted workload: 96 baselines + 576 randomized trials + 576 matching-position
identities + 24 final-layer-only diagnostics = **1,272 forwards**. The existing
launcher installs residual and raw-logit hooks before model loading, validates
four A100 80 GB GPUs, unsets `SGLANG_PORT`, and owns server cleanup. No Slurm
submission is authorized yet.

Outputs: [campaign directory](runs/deepseek-v4-flash/one-fact-patching-filler-random).
Preparation freezes source/input hashes, source snapshots, masks, seeds, required
positions, historical repeat references, and storage accounting. Reporting exports
per-example CSVs, accuracy/transitions, correct-answer log probabilities and raw
logits, and PNG/PDF plots. All comparisons use 2,000 shared whole-panel bootstrap
resamples, seed 42; an independent CSV-based reconstruction checks statistics.
Historical repeat sources 0/5 and 1–4 keep their own runtime baselines and have
explicitly labeled legacy capture coverage. Intervals are conditional on the
single noise realization, and cutoff comparisons also change replacement count.

Validation is being completed in `.venv-sglang` with CPU thread counts limited
to one. The first five new tests passed, including a real 43-layer hook execution
on a small causal model with four simulated ranks, restart recovery, downstream
propagation, norm/seed/rank checks, and tampering. Actual GPU and native DeepSeek
validation remains pending an explicitly approved allocation. These CPU tests
do not substitute for GPU rank checks or empirical model results.

Working tree already contained substantial user changes and untracked experiment
code. This implementation extends the existing patching modules and adds
`filler_random.py`, `random_analysis.py`, and `test_filler_random.py`; no commit
has been made. Final preflight details and the launch command will be recorded
in the campaign directory before requesting approval.

### 2026-09-12T02:29:50.138165+00:00 — CPU preparation complete

Final preflight: **131 tests passed**. Frozen configuration
`264bccd0af09aea9f35957d514dcb1955f91d5bee98b554a94505e1fd660460f`. Source snapshots and current sources independently
rehash exactly. Actual prompts have 24 required capture positions, with separate
Answer/colon/space tokens, and 78,432 distinct seeded streams. Estimate
194.50 GiB; reserve
243.12 GiB.

[Prepared command and review artifacts](runs/deepseek-v4-flash/one-fact-patching-filler-random/PREPARED.md).
The synthetic twelve-condition comparison reproduces statistics to 6.94e-18;
PNG layouts were inspected and PDF exports exist. The sandbox emitted MUNGE
socket messages after successful CPU commands; no scheduler operation was
requested. Scheduler execution must use the approved host path after explicit
Slurm approval. No empirical results, allocation, or commit yet.

### 2026-09-12T02:32:19.139199+00:00 — execution authorized

User approved the prepared four-hour campaign ("go"). Frozen source/configuration checks passed immediately before submitting the prepared salloc/srun command through the host scheduler. Allocation session 12254 owns the automated workload and cleanup.

### 2026-09-12T05:28:27.004762+00:00 — results complete; reporting recovered

All **1,272 forwards** completed in one runtime on job **58219082**, with no
restarts or excluded prompts. All 576 randomized trials, 576 identities, and 24
layer-42-only diagnostics passed full artifact verification on the compute node.
All four ranks and all 43 layers were checked. Native readout residual/rank errors
were exactly zero; maximum audited native log-probability error was 0.125 (tolerance
0.15), with zero checked raw-logit and normalizer error. All 600 score controls
had zero checked log-probability and raw-logit errors. Maximum cast norm error
over 99,168 rank/layer checks was **0.0099003%**, below the 0.5% requirement.

Clean performance: **83/96 (86.46%)**, identical greedy labels across the new and
historical runtime baselines.

| Cutoff j | Random correct | Random accuracy | Change from clean (pp) | Correct→incorrect / incorrect→correct | Mean Δlog probability | Mean Δraw logit |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 70/96 | 72.92% | -13.54 | 14 / 1 | -0.5155 | +36.4935 |
| 1 | 74/96 | 77.08% | -9.38 | 10 / 1 | -0.4421 | +28.6849 |
| 2 | 61/96 | 63.54% | -22.92 | 22 / 0 | -0.6983 | +23.2786 |
| 3 | 60/96 | 62.50% | -23.96 | 23 / 0 | -0.7423 | +29.8919 |
| 4 | 63/96 | 65.62% | -20.83 | 20 / 0 | -0.8493 | +25.4714 |
| 5 | 76/96 | 79.17% | -7.29 | 7 / 0 | -0.2841 | +19.8424 |

Accuracy and target log probability fall for all six random conditions. The
cutoff pattern is non-monotonic; cutoff 5 has the smallest observed accuracy loss,
and cutoff 3 the largest. Raw correct-answer logits rise in every random condition,
while log probabilities decline because the vocabulary normalizer also shifts.
Raw-logit increases therefore do not indicate better answer performance here.

| Cutoff j | Random accuracy | Historical repeat accuracy | Random − repeat (pp), 95% CI |
|---|---:|---:|---:|
| 0 | 72.92% | 64.58% | +8.33 [-3.12, +18.75] |
| 1 | 77.08% | 32.29% | +44.79 [+33.33, +56.25] |
| 2 | 63.54% | 41.67% | +21.88 [+10.42, +34.38] |
| 3 | 62.50% | 51.04% | +11.46 [+1.04, +21.87] |
| 4 | 65.62% | 66.67% | -1.04 [-11.46, +8.33] |
| 5 | 79.17% | 72.92% | +6.25 [-2.08, +15.62] |

Random replacements outperform historical repetition most clearly at cutoffs 1–3
in these descriptive comparisons. All intervals use 2,000 shared whole-panel
bootstrap resamples, seed 42; they are conditional on one noise realization, not
between-noise or between-runtime uncertainty. Cutoff comparisons change the
number of replaced fillers. Historical repeat and random conditions also use
different norm policies (source norm versus destination-clean norm), so their
contrast compares the complete replacement policies.

**Reporting interruption and recovery:** at 02:47:38 UTC, concurrent workspace
edits added multi-token-generation support to `patching_campaign.py`,
`patching_logits.py`, and `scripts/dsv4/one_fact_patching.py`. The controller had
already imported its campaign code before those edits. Lazy logits validation
imports could see the updated module; its added branches preserve one-token
semantics. Every frozen one-token score check also passed during recovery.
After full tensor verification passed, the live-source checksum guard stopped
reporting. Slurm records **FAILED 1:0**, start **02:32:10 UTC**, end/release
**03:32:52 UTC**, elapsed **1h 00m 42s**. This is a reporting/provenance failure,
not a failed model trial or failed tensor validation.

Live edits were preserved. Reporting ran from the checksummed frozen source
snapshot, using a bounded scoring-metadata reader to avoid reprocessing large
hidden-state JSON arrays on the login node. The original full response/tensor
verification had already passed on the compute node; recovery checked journal
envelopes, unchanged response timestamps, small response hashes, all raw-logit
rank hashes and identities, and exact stored scores. It verified **2,040** unique
response metadata records across the new and historical campaigns. No extra
model forwards or allocations were used. Source diffs and the recovery script
are preserved under `runs/deepseek-v4-flash/one-fact-patching-filler-random/report-recovery/`.

Independent reconstruction from exported CSVs matched all means, standard errors,
and percentile intervals: maximum absolute discrepancies **9.77e-15** (random
report) and **1.42e-14** (historical comparison). All six final PNG figures were
visually inspected and all PNG/PDF files validated. The clean legend was moved
outside the axes to expose covered error bars; numerical exports were unchanged.
Rendering used `.venv-sglang` and the existing standalone report style; no notebook
was changed or claimed verified. Observed campaign size was **196.77 GiB**, within
the 243.12 GiB reserve. No commit was made.
