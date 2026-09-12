# One-fact filler redundancy lab log

## 2026-09-11 — implementation and CPU preparation

Objective: test resilience/nonlinear responses to early-block and random later-filler replacement, retaining the original 24 discovery panels and all 96 targets. All-layer full-downstream patching uses both historical donor types and saves actual logits as well as log probabilities.

[Preparation, methods, validation, storage and launch command](runs/deepseek-v4-flash/one-fact-patching-filler-redundancy/PREPARED.md). [Frozen manifest](runs/deepseek-v4-flash/one-fact-patching-filler-redundancy/manifest.json). Configuration `19a2b4e17be684ce4ffd350b25afc624fab49028fefe687148cfd011e4eb148a`. [CPU preflight](runs/deepseek-v4-flash/one-fact-patching-filler-redundancy/preflight.json).

Implemented frozen independent subset draws, raw pre-sampling logits capture, per-rank/request/artifact checks, raw-logit runtime gates, restart-safe single-mode baselines, and panel-level analysis/exports. Initial test failure was an incorrect expectation of immediate within-block downstream propagation; corrected to the next block. Historical-design tests are included in preflight. No substantive model execution, scheduler submission or commit yet; GPU execution awaits explicit approval. No empirical redundancy claim is made from CPU simulations.

Working-tree review: modified the existing patching, campaign-hook, campaign-controller, analysis router and one_fact_patching CLI files; added patching_logits.py, redundancy_analysis.py and test_filler_redundancy.py. All were kept separate from unrelated pre-existing changes. New durable records are this file and the campaign PREPARED.md.

### 2026-09-11T23:10:18.175346+00:00 — final preflight passed

75 tests passed in 48.66 seconds, including historical designs, native residual/readout checks and new raw-logit controls. Manifest/source/checkpoint metadata verified again after tests; prepared configuration `19a2b4e17be684ce4ffd350b25afc624fab49028fefe687148cfd011e4eb148a`. Exact counts: 4,992 trials, 2,496 identities, 96 baselines and 200 first-panel diagnostics (7,784 forward passes). Estimated 211.75 GiB, recommended reserve 264.69 GiB. Launch command is ready in PREPARED.md; empirical execution and results remain pending explicit scheduler approval.

### 2026-09-11T23:11:09.530111+00:00 — approved launch

The explicit tool approval request granted Slurm job **58214506**, node **nid008237**, account **m5258_g**, actual QOS **gpu_interactive**, constraint **gpu&a100&hbm80g**, 128 logical CPUs. Four NVIDIA A100-SXM4-80GB GPUs verified, each 85,093,777,408 bytes. Allocation starts 2026-09-11 23:10:31 UTC and ends 2026-09-12 03:10:31 UTC. Exec session **20501** controls the existing salloc/srun process. Runtime `58214506-13c1e329e61f`; automatic server startup and campaign are active. Both residual and logits hooks are in the actual launch command. No additional submission or cancellation is needed.

### 2026-09-11T23:30:19.505772+00:00 — first allocation failed and was released

Job 58214506 / exec session 20501 exited 1 after about 19 minutes. Native final-layer checks passed (requested raw logits and normalizer exact, hidden/rank errors zero, max top logprob error 0.125); all 104 pilot identities passed, and 62 layer-42 diagnostics completed. No substantive trials ran. Failure: HTTP response returned before rank 1 atomically published its logits JSON. All four rank files subsequently appeared and contained identical values, confirming a publication race in the new collector rather than a numeric validation failure. The controller immediately stopped its owned server and salloc relinquished the allocation.

Original manifest, journal, preflight, launch record and changed source snapshots preserved in `runs/deepseek-v4-flash/one-fact-patching-filler-redundancy/attempts/58214506`. Original rank artifacts remain at their absolute paths under runtimes/. Because there are zero substantive trials, the corrected implementation can be prepared at the same requested campaign root with a new source/config hash, while explicitly checking scientific design equality. No completed substantive observation will be discarded. A new allocation requires a new explicit approval.

### 2026-09-11T23:34:45.762725+00:00 — isolated retry ready

Publication wait and delayed-write/timeout tests passed. 89 passed in 60.90s (0:01:00). Corrected manifest `dfe460ea4a8c4e38e74d2325fb3f595ef86b06edb687cbc6f88bf6544a70583f` preserves every panel, trial, identity, mask, input hash, seed and draw count exactly; [compatibility](/pscratch/sd/m/marocco/sandbox/mech_int/runs/deepseek-v4-flash/one-fact-patching-filler-redundancy/retry_compatibility.json). Concurrent source edits to shared files were detected after the previous preflight, so this campaign now runs from `/pscratch/sd/m/marocco/sandbox/mech_int/runs/deepseek-v4-flash/one-fact-patching-filler-redundancy/source_snapshot`; shared user edits were preserved. Snapshot-only path normalization supports existing data links. Final snapshot source/checkpoint verification passed. The interrupted diagnostic independently matches clean raw logits, normalizer and log probabilities exactly; [audit](/pscratch/sd/m/marocco/sandbox/mech_int/runs/deepseek-v4-flash/one-fact-patching-filler-redundancy/attempts/58214506/PUBLICATION_RACE_AUDIT.json). New allocation awaits approval.

### 2026-09-11T23:36:17.134555+00:00 — approved isolated retry launched

Job **58215423**, exec session **48725**, runtime **58215423-21239d23be70**, node **nid008256**. Account m5258_g, actual QOS gpu_interactive, one node/four verified A100-SXM4-80GB GPUs (85,093,777,408 bytes each), 128 logical CPUs, constraint gpu&a100&hbm80g. Allocation start 2026-09-11T23:34:59 UTC; deadline 2026-09-12T03:34:59 UTC. Executing from source_snapshot/ with configuration dfe460ea4a8c4e38e74d2325fb3f595ef86b06edb687cbc6f88bf6544a70583f. Automatic load, validation and campaign active.

### 2026-09-12T00:00:27.891043+00:00 — retry pilot passed

The isolated runtime passed native final-layer equivalence (requested raw logits and full-vocabulary normalizer exact; hidden/rank differences zero; max top logprob error 0.125), all 104 pilot identities, all 200 layer-42 diagnostics, all 208 substantive pilot trials, and the pilot artifact/delta integrity audit. The controller automatically continued the remaining discovery panels on the same server. Publication synchronization has passed the previous failure point. Early throughput is about 2.1 passes/s; full artifact verification adds material CPU time because retained baseline responses include all hidden states (~26 MiB each).

### 2026-09-12T01:04:53.025623+00:00 — all inference complete; final audit active

All 4,992 substantive trials, 2,496 identities, 96 baselines and 200 diagnostics are durably recorded on the isolated retry runtime. All online rank/logit/identity/diagnostic gates passed. Clean baselines contain 83 correct and 13 incorrect answers, all retained; [baseline summary](runs/deepseek-v4-flash/one-fact-patching-filler-redundancy/NATIVE_BASELINE_SUMMARY.json). The controller is running full artifact/delta verification before automatic analysis and allocation release. No final empirical report is claimed until these checks pass.

### 2026-09-12T01:51:50.375214+00:00 — completed, independently audited and allocation released

Job **58215423** completed the complete planned dataset and exited 0; salloc reported relinquishing the allocation. Reports finished **2026-09-12T01:50:10.660448+00:00**, about **2 h 15 min** after allocation start. All inference finished at 01:04:53 UTC; final integrity audit and analysis took **45 min 18 s**, slightly beyond the initial 30–45 minute estimate. The failed first allocation added about 19 minutes and contributed no substantive observations.

All 4,992 substantive trials, 2,496 identities, 96 baselines and 200 diagnostics passed final integrity checks. Native requested raw logits/normalizer and hidden/rank comparisons were exact; native top/candidate logprob max error 0.125 passed the existing 0.15 tolerance. Every identity and layer-42 diagnostic had exactly zero candidate logit/logprob/normalizer error and identical output IDs. All 96 targets were retained (83 correct, 13 incorrect clean answers). CPU preflight: 89 tests passed. An independent export audit reproduced 324 condition and 432 paired statistics from the CSVs with 20,000 shared panel resamples to 1e-12; all six PNG/PDF files exist and all three PNGs were visually inspected.

Result: later subset sizes 1–5 progressively increase donor-relative preference while mean target-answer log probability stays near clean. At five later fillers, target Δlog p is −0.0634 [−0.2013, +0.1104] for same-addend donors and −0.0292 [−0.1491, +0.1311] for different-addend donors. Different-addend donor-minus-mixed Δgap is +1.3923 [+0.9611, +1.8294] logits; the corresponding same-addend gap is zero by token identity. Five early fillers have substantially weaker donor effects than five later fillers. Intervals are descriptive 95% whole-panel bootstrap intervals, without multiplicity adjustment. Resilience is relevant to redundancy but does not establish active error correction or isolate an addend-specific representation. Mean resilience does not imply invariance of individual examples.

[Interpretation](runs/deepseek-v4-flash/one-fact-patching-filler-redundancy/analysis/INTERPRETATION.md), [full numerical report](runs/deepseek-v4-flash/one-fact-patching-filler-redundancy/analysis/REPORT.md), [execution provenance](runs/deepseek-v4-flash/one-fact-patching-filler-redundancy/EXECUTION_SUMMARY.json), [independent audit](runs/deepseek-v4-flash/one-fact-patching-filler-redundancy/EXPORT_AUDIT.json), and [changes for inspection](runs/deepseek-v4-flash/one-fact-patching-filler-redundancy/CHANGE_REVIEW.md). Configuration remains `dfe460ea4a8c4e38e74d2325fb3f595ef86b06edb687cbc6f88bf6544a70583f`. The executed snapshot is preserved. Shared concurrent work was not overwritten; no commit has been made.
