# One-fact filler-repeat experiment

**Completed:** [results, plots and CSVs](runs/deepseek-v4-flash/one-fact-patching-filler-repeat/analysis/REPORT.md). Baseline 86.46%; repeat filler_5 72.92%; repeat filler_0 64.58%. All 488 forwards and integrity checks passed; allocation released.

## 2026-09-11T23:31:11.388349+00:00 — prepared in isolation

[Preparation, validation and exact launch command](runs/deepseek-v4-flash/one-fact-patching-filler-repeat/PREPARED.md). [Dated lab log](runs/deepseek-v4-flash/one-fact-patching-filler-repeat/LAB_LOG.md).

All 96 discovery prompts; clean filler_5→filler_6–19 versus clean filler_0→filler_1–19 at layers 0–42, full downstream recomputation on every TP rank. Primary accuracy/transition counts and secondary correct-answer logits/log probabilities; 2,000 shared whole-panel resamples, seed 42. Direct comparison combines source and replacement-count effects.

89 CPU tests passed in the final installed-workspace preflight. The frozen uninterrupted workload is 488 forwards; estimate 80.63 GiB, reserve 100.79 GiB. Configuration `73a4d19733cfab312b33ced7ddbf99dabab9da0ba0dbdb2a96b9af9b48f0efdb`. Implementation was staged while redundancy job 58214506 was active, then installed after it exited FAILED. The final preflight includes the rank-logits publication-race fix. The guarded installer, source snapshot and reviewable patch are in the campaign directory. No empirical results, new Slurm submission or commit yet.

## 2026-09-12T00:13:18.356491+00:00 — completed

Job **58215459** completed with exit code **0:0** and released **nid008285** at **2026-09-12 00:08:40 UTC**, elapsed **32m22s**. One uninterrupted runtime completed exactly **488 forwards**: 96 baselines, 192 interventions, 192 identities and 8 final-layer diagnostics. All 24 panels and all baseline errors were retained. Final checks verified source/destination tensors, all 43 layers, all four ranks, frozen provenance, raw responses and matching baselines. All 200 identity/final-layer score controls had zero checked log-probability and raw-logit error.

Results: clean **83/96 (86.46%)**; repeat filler_5 **70/96 (72.92%)**, change **−13.54 pp**, 95% panel-bootstrap interval **[−21.88, −6.25]**; repeat filler_0 **62/96 (64.58%)**, change **−21.88 pp**, interval **[−31.25, −12.50]**. Correct→incorrect counts were **13** and **21**; incorrect→correct counts were **0** for both. Correct-answer mean Δlog probability was **−0.4847** and **−0.6527**; mean Δraw logit **−10.3483** and **−2.0534**, respectively. Raw logits also depend on shifts in the full-vocabulary normalizer and are not alone a performance measure.

The paired repeat_0 minus repeat_5 accuracy change was **−8.33 pp**, 95% interval **[−17.71, +2.08]**. Thus both interventions degraded performance relative to clean, while their direct difference is uncertain; it also combines source-position and 19-versus-14 replacement-count effects. All uncertainty uses **2,000 shared whole-panel resamples, seed 42**; resampling added no model runs.

[Report and plots](runs/deepseek-v4-flash/one-fact-patching-filler-repeat/analysis/REPORT.md), [per-example CSV](runs/deepseek-v4-flash/one-fact-patching-filler-repeat/analysis/per_example.csv), [paired examples](runs/deepseek-v4-flash/one-fact-patching-filler-repeat/analysis/paired_examples.csv), [paired summaries](runs/deepseek-v4-flash/one-fact-patching-filler-repeat/analysis/paired_comparisons.csv), [full summary](runs/deepseek-v4-flash/one-fact-patching-filler-repeat/analysis/summary.json), [environment](runs/deepseek-v4-flash/one-fact-patching-filler-repeat/environment.json), [reproducibility metadata](runs/deepseek-v4-flash/one-fact-patching-filler-repeat/analysis/reproducibility.json), and [export audit](runs/deepseek-v4-flash/one-fact-patching-filler-repeat/analysis/export-validation.json). Independent post-export reconstruction matched every point estimate, SE and percentile interval exactly (maximum absolute discrepancy **0.0**), and all exported scores/accuracy labels matched raw responses and journal checksums. All three PNG figures were visually inspected; PNG and PDF files exist. The Markdown report received editorial clarifications and percentages after automatic export; numerical exports and frozen source files were unchanged.

Final installed-source preflight: **89 tests passed**. No commit was created. The eight-file implementation diff remains in [implementation.patch](runs/deepseek-v4-flash/one-fact-patching-filler-repeat/implementation.patch); its staging snapshot preserves the executed source. Substantive experiment work is complete.

## 2026-09-12 — Repeat sources 1–4 and five-shot k=5

Sources 1–4 now have completed results on the same 96 k=20 discovery prompts, with all-layer/full-residual copying into every later filler. Clean accuracy was 83/96; patched counts were **31, 40, 49, 64 / 96**, respectively. All 880 passes and integrity gates passed. The separate five-shot k=5 supplement scored **186/262 (70.99%)**. Job 58216945 completed 0:0 and released its four A100-80GB GPUs after 48m05s.

See the [extension lab log](runs/deepseek-v4-flash/one-fact-patching-filler-repeat-1to4/LAB_LOG.md), [combined source comparison](runs/deepseek-v4-flash/one-fact-patching-filler-repeat-1to4/combined/REPORT.md), and [updated accuracy notebook](notebooks/addition_accuracy.ipynb). The core-code installer waits for redundancy job 58215423 to exit; its [status](runs/deepseek-v4-flash/one-fact-patching-filler-repeat-1to4/installation-status.json) records completion or conflicts. Sources 0 and 5 remain historical, and source comparisons combine position and replacement-count effects.
