# DeepSeek V4 Flash J-Lens compatibility lab log

## 2026-09-09: implementation and preflight

Objective: test Xiangchen Song's published rectangular Jacobian lens against
this workspace's saved DeepSeek-V4-Flash-0731 activations. The selected scope
is a small compatibility pilot: the Paris prompt plus the corrected Atatürk
addition pilot (`A=57`, `X=11`, `A+X=68`), at the last question token, ten filler
tokens, and answer prefix. No lens fitting or full model loading is required.

### Inputs and provenance

- [Published lens](https://huggingface.co/xiangchensong/jacobian-lens-deepseek-v4-flash-0731):
  revision `a5841da3565b88a79665a418e79d1641d346f667`, default `lens.pt`,
  2,818,579,540 bytes, SHA-256
  `429d6f3810392cf6af7df5f81058eba3619bba34901108e4cb257b76d2efdfff`.
- [Author's code](https://github.com/xiangchensong/jacobian-lens-open-frontier):
  revision `b8b840caaa246ad04d354b4650848f30519184b6`. A source snapshot of
  the library, license, README and package metadata lives in
  `ports/jacobian-lens-open-frontier/`; `SOURCE.json` hashes every copied file.
- Local checkpoint: `model/DeepSeek-V4-Flash-0731-MoE-MXFP4-BF16`.
- Existing captures: `runs/deepseek-v4-flash/logit-lens/57804608/captures/`,
  Paris pass 0 plus passes 26, 28, ..., 48 identified by
  `filler-pilot-v2/manifest.json` in the same run. The earlier `filler-pilot/`
  captures are excluded because of their documented trigger/pass ambiguity.
- Environment: `.venv-sglang`, Python 3.12, Torch 2.11.0, Transformers 5.8.1.
  The fork declares Transformers >=5.14, but its loader, transport and readout
  adapter import successfully with the existing environment. No packages were
  installed or upgraded; the command scopes the pinned source import path.

Downloaded only the default fit; source and model pins can be restored with:

```bash
.venv-sglang/bin/python -m scripts.dsv4.prepare_jlens
```

### Method and implementation

`filler/dsv4/jlens.py` loads the tensor-only checkpoint with `weights_only=True`
and memory mapping. It validates dimensions, layer metadata, fitting count,
matrix dtype, finiteness and nonzero content. The real matrix scan runs on
compute; small synthetic fixtures run on the login node with one math thread.

For zero-based layers 19–39, flatten each complete `[4,4096]` residual in
row-major order and compute `h.float() @ J.float().T`. Matrices are stored
fp16 and transported in fp32. The resulting 4096-vector is already in the
collapsed basis; never apply mHC collapse again. Cast to the local vocabulary
head's BF16 dtype, apply the HF final RMSNorm, and unembed.

Precision detail verified from the installed HF implementation: HF rounds the
normalized vector before multiplying the norm weight. The existing native
SGLang lens multiplies the weight in fp32 before rounding. J-Lens deliberately
matches the published HF path; ordinary Logit Lens keeps its validated native
rounding. The reference check uses the author's actual adapter with the actual
HF `DeepseekV4RMSNorm`, holding only the local norm and vocabulary head.

`filler/dsv4/jlens_validation.py`, exposed by
`python -m scripts.dsv4.validate_jlens`, validates all four capture ranks and
the saved Paris native readout before comparing the lenses. It processes one
layer at a time, checking both transported vectors and complete readouts
against the reference with `rtol=1e-5`, `atol=1e-5`. The Paris gate retains
exact hidden agreement, native argmax equality, top-ten overlap 10/10 and
maximum native-top-token log-probability error <=0.15. There is no supplied
layer-42 Jacobian; this gate checks the ordinary final readout.

Outputs contain 273 paired comparisons (13 positions × 21 layers), full-vocabulary
target rank/logit/log-probability/log-odds, both top-ten decoded readouts, CSV,
paired rank heatmaps in PNG/PDF, reference errors and a Markdown report.
Partial layer results are retained. A failed or completed output directory
cannot be silently overwritten; reruns need a fresh `--output-dir`.

### Preflight validation

All 13 selected passes were inspected: all 43 layers exist on all four ranks,
all residuals are finite BF16 `[4,4096]`, and ranks agree exactly.
Checkpoint download size and SHA-256 match the published metadata.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 .venv-sglang/bin/python -m pytest -q \
  tests/test_dsv4_jlens.py tests/test_deepseek_v4_logit_lens.py \
  tests/test_dsv4_factorial.py
```

Result: **30 passed in 25.02 seconds**. Checks include rectangular checkpoint
load/reference transport, shape and metadata rejection, missing layers,
nonfinite inputs, BF16 and FP32 reference readout, batch-size-four ambiguity,
HF versus fused norm rounding, native gate failures, exact target metrics,
and a small complete comparison with CSV/PNG/PDF/Markdown export.
The login-node environment emitted an NVML warning and sandbox MUNGE messages;
the test process exited zero. No GPU computation or scheduler operation occurred.
Shell syntax and `git diff --check` passed.

### Prepared execution

Compute calculations use one CPU node, 16 threads and a 30-minute allocation.
The launcher captures UTC Slurm metadata and performs the comparison offline.
The Python runner refuses real-checkpoint execution outside a compute-node
Slurm environment. Prepare/download/checks precede submission approval.

```bash
salloc --account=m5258 --qos=interactive --nodes=1 \
  --constraint=cpu --time=00:30:00 \
  srun --ntasks=1 --cpus-per-task=16 --cpu-bind=cores \
  bash /pscratch/sd/m/marocco/sandbox/mech_int/run_jlens_smoke_allocation.sh
```

Explicit user approval is required before submitting this command under
[AGENTS.md](AGENTS.md). Output root:
`runs/deepseek-v4-flash/jlens-smoke/`. Actual results remain pending until the
approved run completes; preflight is not evidence that the real J-Lens passed.

This pilot validates software compatibility on converted SGLang activations.
It does not establish equality with the author's original inference backend,
causal validity, or population performance. It does not select intervention
sites or change `should_run_jlens`. Existing runs, active model servers and
unrelated working-tree changes are preserved. No commit was made.

## 2026-09-09 23:53 UTC: first CPU attempt and top-k diagnostic

Approved job `58132933` ran on `nid004222`, account `m5258`, QOS
`interactive`, with the prepared 16-thread step. Slurm start was 23:53:36 UTC;
the 30-minute limit ended at 00:23:36 UTC on September 10. The task stopped
after 6.51 seconds and the allocation was automatically relinquished.

The native gate reported exact hidden/rank agreement, matching Paris argmax,
and maximum log-probability error **0.130479**, below 0.15. It rejected a raw
top-ten overlap of **9/10**, so no J-Lens comparison was run or accepted.
Original diagnostics remain in `runs/deepseek-v4-flash/jlens-smoke/`.

Investigating an exact BF16 tie at the top-ten cutoff: top-k implementations
may select different token IDs at equal scores. The gate now records the
actual cutoff, boundary tie count, selected top ten, and unmatched native
tokens. It permits a differing selection only when all native top-ten IDs
meet the *exact* cutoff and every strictly higher-scoring token is included.
No numerical tolerance was increased. A synthetic regression verifies an
exact tied replacement passes and a lower-scoring replacement fails. The
real tie explanation remains pending verification on compute.

Also preserve full provenance before the native gate, including on failures.
The launcher accepts an explicit fresh output directory for retries. A second
30-minute CPU allocation will keep an interactive shell open through validation
and diagnosis, avoiding automatic release before an unexpected failure can
be inspected. Retry outputs will use `jlens-smoke/retry-1/`; the first attempt
will not be overwritten. Another Slurm submission requires explicit approval.

## 2026-09-10 01:53 UTC: approved retry and exact-tie confirmation

User approved another 30-minute CPU allocation. Job `58136361` started at
01:52:30 UTC on `nid004184`, account `m5258`, QOS `interactive`, deadline
02:22:30 UTC. The 16-thread process runs inside a retained interactive compute
shell. All **31 CPU tests passed in 19.43 seconds** before this submission.

The real diagnostic confirmed the suspected exact tie: CPU top-k chose token
7854 where the saved native top ten included token 223; both have log
probability **-6.2783355712890625**, exactly at the tenth-place cutoff. There
are two tied tokens. All strictly higher-scoring tokens agree. The native
gate now passes while retaining raw overlap 9/10 and the unchanged maximum
log-probability error 0.1304793357849121. No tolerance was relaxed.

The reference comparison is running at
`runs/deepseek-v4-flash/jlens-smoke/retry-1/`. Its `native_validation.json`
preserves the cutoff and unmatched-token evidence; `provenance.json` includes
source/input hashes and exact readout tensor hashes. Final comparisons and
reports remain pending.

### Manifest-label fix within the retained allocation

The retry passed the first layer's actual transport and full-readout reference
checks, then stopped while constructing output rows: the saved pilot uses
`label`, whereas the normalized comparison interface uses `position_label`.
The initial small comparison fixture had supplied normalized rows directly,
so it missed this file-to-interface boundary. Failure occurred after 41.75
seconds; no completed J-Lens results were accepted.

Fixed the manifest conversion and added a compact on-disk fixture exercising
all thirteen passes/four ranks, label normalization and response-ID mismatch
rejection. Final relevant suite: **32 passed in 36.88 seconds**. Original
failure artifacts remain in `retry-1/`. Restarted the corrected command inside
the same approved job `58136361`, using fresh output `retry-2/`; no new Slurm
submission or allocation modification was needed.

## 2026-09-10 01:57 UTC: completed compatibility pilot

The corrected run completed in **38.04 seconds** inside job `58136361`.
All **273 paired comparisons** (13 positions × 21 fitted layers) passed.
The maximum absolute difference from the pinned author's implementation was
**0.0 for transported vectors and 0.0 for full vocabulary readouts** across
every layer. All original input/source hashes were unchanged at completion.
Independent output inspection verified 273 unique case/position/layer keys,
all layers 19–39, and exactly 1,554 target-score CSV rows. PNG/PDF exports
were generated and the PNG was visually inspected.

Results:

- For the exact native Paris token (space-prefixed token 11111), best rank
  within the fitted band is 1 for ordinary Logit Lens and 2 for J-Lens, both
  at layer 31. Other Paris tokenizations are shown in the top-ten output;
  this metric tracks the exact native token, not a semantic equivalence class.
- Both lenses rank the sum token `68` first at filler_5, layer 38, and at the
  answer prefix, layer 38. Across all 252 addition position/layer pairs, each
  has three rank-one sum readouts. The factual token `57` is rank one at five
  ordinary-lens sites and four J-Lens sites; `11` is never rank one.
- J-Lens gives weaker sum ranks at many other positions in this example:
  the best filler_0 sum rank is 49,119 versus 80 for ordinary Logit Lens,
  and the best filler_9 sum rank is 886 versus 13. These are descriptive
  observations from one addition prompt, not a population comparison.

Compatibility therefore passed. This pilot does **not** demonstrate a
readout advantage or establish transfer equivalence between the author's
fitting backend and the local converted SGLang model. No model reload,
backward pass, discovery-panel expansion, or causal experiment was performed.

Artifacts:

- [Comparison report](runs/deepseek-v4-flash/jlens-smoke/retry-2/REPORT.md)
- [Validation results](runs/deepseek-v4-flash/jlens-smoke/retry-2/validation.json)
- [All readouts and target metrics](runs/deepseek-v4-flash/jlens-smoke/retry-2/results.json)
- [Target CSV](runs/deepseek-v4-flash/jlens-smoke/retry-2/target_scores.csv)
- [PNG heatmaps](runs/deepseek-v4-flash/jlens-smoke/retry-2/target_ranks.png)
- [PDF heatmaps](runs/deepseek-v4-flash/jlens-smoke/retry-2/target_ranks.pdf)
- [Provenance](runs/deepseek-v4-flash/jlens-smoke/retry-2/provenance.json)

After integrity checks and report inspection, exited the retained compute
shell to release the allocation. Implementation and documentation remain
uncommitted for user inspection. A broader scientific comparison would need
a separately specified discovery/confirmation protocol and controls for
backend/precision transfer.

## 2026-09-10: camilablank workspace square J-Lens

Objective: implement the requested
[workspace J-Lens](https://huggingface.co/camilablank/workspace-lenses/tree/main)
for DS V4 Flash by averaging the four complete post-decoder-block mHC streams
before applying each square matrix. This is a separate artifact and reduction
from the rectangular 0731 lens above; its earlier validation results do not
validate this square release.

### Artifact and method

Pinned `camilablank/workspace-lenses` revision
`d740106d1e0f95456dc8718fba2895e9c8ffd6ef`, file
`deepseek-v4-flash/j-lens/lens.pt`, size **1,409,295,893 bytes**, SHA-256
`8b010eef8b2b08efb1b07601e5203ff5d215b1fcae63704847fdf001e61e0efc`.
The public Hub API supplied the pin. A bounded 32 KiB HTTP range and
`pickletools` inspection (no pickle execution) established the actual schema:
`J` is a dictionary of **42 fp16 [4096,4096] matrices**, keyed by zero-based
post-block layers **0–41**, alongside `n_prompts=25`, `source_layers`,
`d_model=4096` and provenance. The provenance identifies model
`deepseek-ai/DeepSeek-V4-Flash`, dataset `NeelNanda/pile-10k`, target 41,
`skip_first=4`, `t_max=128`, and standard estimator/arm `std`.
`skip_first` excludes fitting token positions, not decoder layers.

The pinned download completed at
`model/workspace-lenses/deepseek-v4-flash/j-lens/lens.pt`; its downloader wrote
`SOURCE.json` after size and SHA-256 verification. The system `python3` is 3.6
and cannot run this project; use `.venv-sglang/bin/python` (3.12). No packages
or shared installations were changed.

For each token independently, with `h` shaped `[...,4,4096]`:

```python
mean = h.float().mean(dim=-2)
transported = mean @ J[layer].float().T
logits = unembed_transported(transported, weights)
```

Averaging and transport use fp32; batch and token axes remain intact. Flattened
`[...,16384]` captures are reshaped before averaging. The readout casts to the
head dtype, applies the actual HF-style final RMSNorm rounding and unembeds.
It never applies the learned mHC head to the mean or transported vector.
There is no supplied layer-42 matrix and no identity fallback for absent
layers. Layer 41 is required to be exactly identity during full value
validation. That anchor tests mean-only readout equivalence; it is **not** a
claim of equivalence to the native final logits, which use a learned collapse.

### Implementation and use

- `filler/dsv4/jlens.py`: `load_workspace_jlens`, `mean_mhc_streams`, and an
  explicit `JLens.stream_reduction` field. Existing rectangular loading and
  flattening remain the default. Square loading validates shapes, layer keys,
  prompt count, model/fit metadata, estimator, finiteness, nonzero values and
  identity anchor. The equally shaped R-Lens is rejected.
- `filler/dsv4/workspace_jlens_artifact.py` and thin command
  `scripts/dsv4/prepare_workspace_jlens.py`: pinned, resumable download with
  checksum verification and protection of existing files.
- `filler/dsv4/jlens_top_object.py`: `--lens-format workspace-mean` uses the
  new loader/readout for existing captures, retains complete-vocabulary and
  numerical argmaxes, records artifact and pooling provenance, and chooses a
  separate output directory. Rows use `lens="jlens_workspace_mean"`.
  Existing rectangular rows keep `lens="jlens"`.
- `tests/test_workspace_jlens.py` and additions to
  `tests/test_jlens_top_object.py`: arithmetic, precision, shape, artifact,
  saved-grid, download-integrity and login-node-boundary regressions.

Download and metadata-only preflight:

```bash
.venv-sglang/bin/python -m scripts.dsv4.prepare_workspace_jlens
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-sglang/bin/python -m scripts.dsv4.prepare_jlens_top_object \
  --lens-format workspace-mean --preflight
```

Use the reusable API for complete outputs from `NativeResidualHooks`:

```python
from filler.dsv4.jlens import load_workspace_jlens, project_jlens_logits
from filler.dsv4.lens import load_checkpoint_readout
from filler.dsv4.workspace_jlens_artifact import DEFAULT_PATH

# On an approved compute node; no full model load is needed for saved captures.
lens = load_workspace_jlens(DEFAULT_PATH)
weights = load_checkpoint_readout("model/DeepSeek-V4-Flash-0731-MoE-MXFP4-BF16")
logits = project_jlens_logits(post_block_states, lens, layer, weights)
```

After explicit allocation approval, saved-grid scoring can run automatically
on one CPU node with 16 threads:

```bash
salloc --account=m5258 --qos=interactive --nodes=1 --constraint=cpu --time=00:30:00 \
  srun --ntasks=1 --cpus-per-task=16 --cpu-bind=cores \
  env OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 \
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv-sglang/bin/python -m scripts.dsv4.prepare_jlens_top_object \
  --lens-format workspace-mean --device cpu --threads 16 --batch-size 64
```

No allocation was submitted in this session. This command validates all real
matrices, then scores saved activations; it does not allocate GPUs or load the
full model. Default output is
`runs/deepseek-v4-flash/workspace-jlens-top-object-heatmap/`; an existing run
requires a fresh `--output-dir`. Full scoring/validation refuses execution
outside a Slurm compute node. FP32 matrix caching requires about 2.8 GB in
addition to the vocabulary head and bounded capture/logit batches.

The existing `export_jlens_top_object` command and executed J-Lens notebook
are specific to the rectangular release and intentionally reject square rows.
For new square rows, use the generic `scripts.dsv4.plot_top_object_heatmap`
command with `--rows`, `--filler-length`, `--cohort`, `--output` and `--stats`,
or the existing `aggregate_rows`/`plot_stats` API.

### Validation and execution boundary

The first synthetic regression run had **79 passes and one plotting failure**
in 55.10 seconds. The failure was the existing rectangular export test:
local Matplotlib settings enabled TeX, whose `cmr8.tfm` font was unavailable.
Rerunning that one test with an isolated `MPLCONFIGDIR` passed in 46.06 seconds;
no plotting or numerical code was changed to work around it. The square
readout tests already passed against the pinned HF adapter with both fp32 and
bf16 weights, including batch/token dimensions equal to four. Nonsymmetric
matrices catch transposition mistakes, unequal streams catch pooling errors,
and poisoned mHC weights catch accidental learned collapse.

The suite command was:

```bash
MPLCONFIGDIR=/tmp/mech-int-workspace-jlens-mpl \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-sglang/bin/python -m pytest -q \
  tests/test_workspace_jlens.py tests/test_dsv4_jlens.py \
  tests/test_jlens_top_object.py tests/test_deepseek_v4_logit_lens.py \
  tests/test_dsv4_factorial.py
```

Actual-file metadata preflight passed without touching matrix values:
**4,128 captures**, **173,376 planned rows**, all layers **0–41**, filler
lengths **0,5,10,20**, with **96 examples per filler length**. It used
`torch.load(weights_only=True, mmap=True)` and `validate_values=False`.
No complete real matrix scan, vocabulary projection or model forward pass was
run. Full value/anchor validation and evaluation on the converted 0731
checkpoint remain pending on compute; dimensional agreement alone does not
establish transfer performance between checkpoints/backends.

The user asked whether this work belonged on a login node. Confirmed no
remaining download/test process after the interruption, and kept login work
to edits, small single-threaded synthetic tests and metadata inspection.
[NERSC policy](https://docs.nersc.gov/policies/resource-usage/#nersc-login-node-policy)
permits short Python work on small datasets and directs compute/memory
intensive work to compute nodes. Real lens scoring remains behind the Slurm
compute-node guard. All changes are uncommitted for inspection; unrelated
working-tree work was preserved.

Final boundary checks: both new CLI tests passed in **11.94 seconds**. They
verify the square preflight plan and that non-preflight login execution fails
before loading readout weights or evaluating matrices. Across the initial
suite, isolated plotting rerun and these two additions, **82 tests passed**.
All six changed/new Python files passed AST parsing under project Python 3.12;
`git diff --check` passed. The implementation and metadata preflight are ready;
real-matrix numerical validation remains unrun.

## 2026-09-10: paired square/rectangular comparison implementation

Objective: compare both published artifacts on all 4,128 saved one-fact captures,
using layers 19–39 for 86,688 direct position/layer pairs and retaining all square
layers 0–41 (173,376 rows). No model reload or lens fitting is needed.

Added `scripts.dsv4.compare_jlenses`, shared batch scoring with optional exact
A/X/A+X ranks and full-vocabulary log-probabilities, strict pairing and baseline
provenance validation, and `notebooks/one_fact_jlens_comparison.ipynb`. The notebook
reads saved results with filler-length, original-correctness cohort, metric and
example selectors. Exports include paired CSV/JSON, summaries with agreement and
exclusive-selection counts, shared-scale PNG/PDF panels, square-only panels,
an executed notebook and a Markdown report. Differences are square minus
rectangular. Rank counts strictly greater logits; lowest-ID argmax tie-breaking
is preserved, so a rank-one target is not necessarily selected.

The scorer deserializes each capture once for both lenses and records the SHA-256
of those exact bytes. Both use the same BF16 head, HF norm, fp32 transport,
numeric vocabulary, batch size 64 and 16 threads. Both artifact checksums,
full matrix/identity validation, independent Paris/addition transport/HF pilots
and the existing native final-readout gate precede scoring. Any baseline argmax
mismatch stops acceptance with a diagnostic file. Input hashes are rechecked.
The historical rectangular baseline lacks capture/head hashes; exact argmax
reproduction is a retrospective compatibility check, not proof of historical
byte identity. Different fitting recipes prevent attributing differences solely
to stream averaging versus flattening.

Login-node checks used small synthetic tensors with one math thread:
`tests/test_jlens_comparison.py`, `test_jlens_top_object.py`,
`test_workspace_jlens.py`, `test_dsv4_jlens.py`: **100 passed in 39.18 s**.
Metadata-only preflight confirmed 4,128 captures, 96 examples per filler length,
173,376 square rows and 86,688 rectangular rows. No real matrix values or readout
were evaluated by preflight. Notebook dependency inspection found no nbclient
or nbformat, so execution uses the established local in-process export pattern,
embedding static figures/tables and leaving selectors rerunnable without any
package installation. An additional synthetic notebook-export test was added.

User explicitly authorized the two-hour CPU allocation in chat after requesting
implementation. Prepared launch (automatically runs and relinquishes allocation
on workload exit or failure):

```bash
salloc --account=m5258 --qos=interactive --nodes=1 --constraint=cpu --time=02:00:00 \
  srun --ntasks=1 --cpus-per-task=16 --cpu-bind=cores \
  bash /pscratch/sd/m/marocco/sandbox/mech_int/run_jlens_comparison_allocation.sh
```

Output: `runs/deepseek-v4-flash/jlens-comparison/JOB_ID/`.
UTC scheduler/start/end logs: `runs/deepseek-v4-flash/jlens-comparison-execution/JOB_ID/`.
The sandbox shell/file editor cannot see the mounted workspace in this session;
scoped host execution is being used. Existing unrelated changes were preserved;
no commit is authorized or made. Actual compute results are pending launch.

### Approved allocation 58175818: real-artifact gates passed

Job `58175818` started at **2026-09-10 22:50:41 UTC** on `nid004192`, account
`m5258`, QOS `interactive`, two-hour deadline **2026-09-11 00:50:41 UTC**.
The workload step uses 16 CPU threads; no GPUs were requested. Allocation metadata
is in `runs/deepseek-v4-flash/jlens-comparison-execution/58175818/allocation.txt`.

The notebook-export regression and comparison suite passed **29 tests in 15.84 s**
(101 unique tests across the initial suite and the added export test). Shell syntax
and `git diff --check` passed. The local exporter required no new dependencies.

Both real artifact checksums, all matrix value checks and the square identity anchor
passed. All 42 square and 21 rectangular layers passed the 13-position Paris/addition
pilot with **0.0 maximum transport error and 0.0 maximum full-readout error** for both
lenses. Saved native validation passed: exact hidden agreement, matching argmax,
maximum top-token log-probability error 0.1304793358 <= 0.15, and the previously
validated exact top-ten boundary tie (raw overlap 9/10; equivalence up to ties).

Full-grid scoring is in progress at
`runs/deepseek-v4-flash/jlens-comparison/58175818/`; paired acceptance and exports
remain pending until `COMPLETE.json` reports success with an executed notebook.

## 2026-09-11: completed paired comparison and validation-loop recovery

All 4,128 captures were scored successfully in job `58175818`: **173,376 square
rows**, **86,688 rectangular rows**. Final row validation exposed a performance
bug: this installed tokenizer reconstructs vocabulary metadata when asked for its
full size. A 100-call timing check on the allocated CPU node measured **0.0240 s
per call**. Repeating that lookup for every target would take hours, despite all
expensive scoring having finished. The implementation now caches the vocabulary
size and numeric-ID set once and validates through `validate_token_metrics`.

Ran `scripts.dsv4.recover_jlens_comparison` in an overlapping 16-thread step inside
the **same allocation**, using the complete saved scores. No projections were
repeated. Recovery verified original source/input/capture hashes, rechecked both
artifact checksums, validated the complete row grids and target bounds, and
required exact reproduction of both rectangular argmax scopes. It hard-linked
finished immutable scores into a fresh result directory, preserving the original
run. Both scopes matched at **all 86,688 rows**. Recovery plus all exports completed
successfully in **240.82 seconds**. The original slow validation process was then
interrupted; Slurm relinquished the allocation at **00:28:05 UTC**, before its
00:50:41 deadline. The original step's signal/exit status reflects stopping the
redundant validator after recovery succeeded; it is not a failed scoring result.

Accepted results:

- [Report](runs/deepseek-v4-flash/jlens-comparison/58175818-recovered/REPORT.md)
- [Executed notebook](runs/deepseek-v4-flash/jlens-comparison/58175818-recovered/one_fact_jlens_comparison.executed.ipynb)
- [Paired CSV](runs/deepseek-v4-flash/jlens-comparison/58175818-recovered/paired.csv)
- [Paired JSON](runs/deepseek-v4-flash/jlens-comparison/58175818-recovered/paired.json)
- [Acceptance and counts](runs/deepseek-v4-flash/jlens-comparison/58175818-recovered/COMPLETE.json)
- [Final artifact inspection](runs/deepseek-v4-flash/jlens-comparison/58175818-recovered/FINAL_INSPECTION.json)

There are **86,688 matched position/layer pairs**, **260,064 paired target rows**,
**96 PNGs plus 96 PDFs** (paired and square-only views for each filler length,
cohort and metric), and a successfully executed notebook. The default top-1 figure
was visually inspected: axes, shared absolute scales, zero-centered difference
scales, target labels and cohort denominator are correct. All retained hashes
were verified before recovery exports. The original baseline lacks capture/head
tensor hashes, so exact baseline readout reproduction is the historical check.

Saved cohort counts (all = 96 each): filler 0 correct/wrong **80/16**; filler 5
**72/24**; filler 10 **72/24**; filler 20 **83/13**.

For filler 20, all examples, equal weighting over the 44,352 matched
position/layer pairs for each target:

| Target | Square top-1 | Rectangular top-1 | Square MRR | Rectangular MRR | Square mean log-p | Rectangular mean log-p |
|---|---:|---:|---:|---:|---:|---:|
| A | 1.9007% | 1.1183% | 0.027711 | 0.014540 | -17.0872 | -35.1921 |
| X | 2.4148% | 0.0383% | 0.039605 | 0.001647 | -16.0857 | -34.2413 |
| A+X | 0.8861% | 0.8861% | 0.013277 | 0.011310 | -18.5619 | -35.4504 |

Full-vocabulary argmax agreement across these pairs is **7.8779%**. The square
artifact has higher factual/addend top-1 readability and higher MRR and mean
log-probability for all three targets in this pooled view; sum top-1 is equal.
Per-position/layer differences, numeric-only rates, agreement rates and exclusive
selection counts remain available in the notebook and tables. These averages do
not isolate effects of stream pooling: fitting recipes, sample sizes, target
bases and model/backend provenance differ, and no causal conclusion follows.

After allocation release, fixed the repeated vocabulary lookup permanently in
both runners and made `COMPLETE.json` publication follow successful notebook
execution. Regular readers now reject missing notebook exports; only the internal
executor can read already-validated results before completion. These changes
preserve scoring and metric arithmetic. **51 focused tests passed in 14.79 s**,
covering the updated exporter, token bounds and existing scorer (106 unique
passing tests across the session). Project-Python compilation, shell syntax and
`git diff --check` passed. An earlier compile-only invocation used system Python
3.6 and rejected future annotations; the actual recovery used project Python 3.12
and completed normally. No dependencies were installed.

The completed artifact provenance records the source hashes actually used for
scoring and recovery. Subsequent source edits are the validation performance and
completion-marker fixes above; they do not alter accepted numeric outputs.
All changes remain uncommitted for user inspection; unrelated work is preserved.

## 2026-09-11: complete answer-prefix coverage requested

The user explicitly requires `Answer`, `:` and trailing space positions for lens
artifacts in general, and requested them for both k=0 and k=20 square J-Lens
views. Added durable project guidance in `AGENTS.md`. Both published J-Lenses
will be rescored for all displayed positions from the same full-prompt capture
per example. The historical last-token grid cannot supply Answer/colon, and
mixing it with later full-prompt residuals would change capture provenance
within a heatmap. Its k=5/k=10 outputs remain available as labeled historical
artifacts; the complete-prefix run covers k=0/k=20.

No new forward passes are needed. Reuse the completed ordinary-lens capture
run `answer-token-heatmap/runtimes/58169491-f5aaee6f`: 96 k=0 captures from that
runtime and 96 k=20 baselines from the earlier full-prompt patching campaign.
Native CUDA final-readout equivalence passed for both lengths with maximum
candidate/top logprob error 0.0; all 192 final-answer checks passed. Each
example's cohort is taken from its own saved native response (80/16 correct/wrong
for k=0, 83/13 for k=20), consistently across all its positions.

Added `lens_positions.py`, `jlens_full_prompt.py` and `jlens_answer_tokens.py`.
`python -m scripts.dsv4.compare_jlenses` now defaults to full-prompt inputs;
`--legacy-grid` explicitly reproduces incomplete historical coverage. The old
single-lens preparation CLI requires `--allow-incomplete-answer-prefix` rather
than silently generating another incomplete artifact. The comparison notebook
derives filler lengths and position selectors from its selected result and
renders Answer/colon/space separately; historical incomplete runs are labeled.

Metadata preflight checks the completed capture pointer/configuration, planned
cells, exact prompt token IDs and suffix, unpatched controls, all TP
acknowledgements, response identity, native validation records, checkpoint
configuration/stat metadata, cohort counts and every required file. It does
not load activation tensors or scan Jacobian values. Compute rechecks capture
hashes across all ranks and both artifact checksums, preserves the Paris/native
and published-reference gates, tests the new full-prompt Answer/colon readouts
against independent references, and runs a fixed old-capture rectangular
regression probe. The new capture source is not required to reproduce the old
activation grid. Exact source snapshots will accompany outputs.

Actual metadata preflight passed: **192 capture files**, **2,688 selected
positions**, **112,896 square rows**, **56,448 rectangular rows**, k=0 four
positions and k=20 24 positions, all 96 examples per length. Initial regression
suite: **75 passed in 15.20 s**, including metadata-only preflight, token-offset
selection, per-capture batching, checksum/identity failures and plotting. An
additional complete-prefix export/notebook integration test is being run before
submission. No Slurm job has yet been submitted for this refresh.

Prepared CPU-only command (new explicit allocation approval required):

```bash
salloc --account=m5258 --qos=interactive --nodes=1 --constraint=cpu --time=01:30:00 \
  srun --ntasks=1 --cpus-per-task=16 --cpu-bind=cores \
  bash /pscratch/sd/m/marocco/sandbox/mech_int/run_jlens_comparison_allocation.sh
```

The launcher automatically validates, scores, exports and releases the allocation
on workload exit. Output uses a fresh job-ID directory under `jlens-comparison`.
The previous allocation 58175818 was already released; its approval is not
being treated as authorization for an additional allocation.

### 2026-09-11 05:09 UTC: approved complete-prefix run

The user approved the prepared CPU submission. Job **58185137** started at
**04:30:31 UTC** on **nid004161**, account `m5258`, QOS `interactive`, with a
90-minute limit (deadline 06:00:31 UTC), 16 CPU threads and batch size 64.
The additional export/notebook test passed: **17 tests in 16.42 s** in the new
module, **76 unique passing tests** across this refresh. Shell syntax, project
Python compilation and `git diff --check` also passed before submission.

Both artifact checksums, matrix-value checks, square identity anchor, all saved
native gates, the original Paris/addition pilot, and the new eight-position
full-prompt pilot passed. The historical rectangular regression probe reproduced
both argmax scopes for all **1,344 rows** (64 old capture positions × 21 layers).
This is a fixed historical regression check; the refreshed full-prompt results
use different saved activations and are not claimed to reproduce that old grid.
The earlier completed comparison already reproduced all 86,688 historical rows.

All **2,688 positions** have now been scored: **112,896 square rows** and
**56,448 rectangular rows**. Final integrity checks and exports are in progress
at `runs/deepseek-v4-flash/jlens-comparison/58185137/`. Acceptance still requires
successful notebook export and `COMPLETE.json`; the launcher releases the
allocation automatically on exit. No new model forward passes or GPUs were used.

### 2026-09-11: complete-prefix results accepted

Job **58185137** completed with **exit status 0** and automatically relinquished
its CPU allocation. The full workflow took **2,351.31 seconds (39.2 minutes)**.
All 192 captures / 2,688 positions passed final input-integrity checks:
**112,896 square rows**, **56,448 rectangular rows**, **56,448 matched pairs**
and **169,344 paired target rows**. Both independent pilots passed all 63
lens/layer checks with **0.0 maximum transport and full-readout error**.

Final inspection verified the complete, unique position/layer/target/cohort grids
in both summary exports: all three answer-prefix positions at every supported
layer, for both k=0 and k=20. Every cohort denominator matches the saved model
labels (96 total per length; correct/wrong 80/16 and 83/13). There are **48 PNGs
and 48 PDFs**. All four notebook code cells executed without errors. Visually
checked the k=20 paired top-1 figure and k=0 all-layer square top-1 figure;
Answer, colon and space appear separately, with correct axes and color scales.

- [Interactive comparison notebook](notebooks/one_fact_jlens_comparison.ipynb)
- [Executed notebook](runs/deepseek-v4-flash/jlens-comparison/58185137/one_fact_jlens_comparison.executed.ipynb)
- [Report](runs/deepseek-v4-flash/jlens-comparison/58185137/REPORT.md)
- [Paired CSV](runs/deepseek-v4-flash/jlens-comparison/58185137/paired.csv)
- [Paired JSON](runs/deepseek-v4-flash/jlens-comparison/58185137/paired.json)
- [Completion](runs/deepseek-v4-flash/jlens-comparison/58185137/COMPLETE.json)
- [Final inspection](runs/deepseek-v4-flash/jlens-comparison/58185137/FINAL_INSPECTION.json)

The interactive notebook automatically selects the newest completed result;
choose square-only for layers 0–41, paired for common layers 19–39. k=5/k=10
remain separately available in the labeled historical artifact and have not
been expanded to full-prefix coverage. Published fitting recipes and backend
provenance differ; readability differences cannot be attributed solely to
stream averaging versus flattening. The all-prefix preference is now durable
in `AGENTS.md`. Changes remain uncommitted for user inspection.
