# DeepSeek V4 Logit Lens lab log

## 2026-08-31: launch sequencing failure

The first server was launched without activation hooks or hidden-state return.
It paid 757.7 seconds for expert preparation and then successfully served, but
could not export intermediate mHC residuals because SGLang registers hooks at
model construction and has no runtime HTTP operation for adding them. Existing
`dsv4_factorial.py` already contained `NativeResidualHooks` and native mHC
projection but was not discovered before launch. After explicit authorization,
the server was stopped and idle allocation `57803794` was relinquished.

Corrective rule: search for existing integration, define and CPU-test the exact
end-to-end validation, and verify every launch-time hook/flag before any future
expensive load. A warm process that cannot expose required experiment state is
not a successful launch.

## 2026-08-31: instrumented relaunch preflight

Prepared `run_deepseek_v4_logit_lens.sh`, which launches the A100 port with:

- SGLang forward hooks on all `model.layers.*` modules;
- `--enable-return-hidden-states` as an independent final-state check;
- an explicit `SGLANG_OPT_FUSE_MHC_POST_PRE=0` invariant;
- one-shot first-prefill capture on every TP rank, after which hooks are inert;
- a trigger-file mtime mechanism for later one-shot captures without restart.

`probe_dsv4_logit_lens.py` defines the deterministic first request and preserves
the native token, top log probabilities, and returned final state.
`validate_dsv4_logit_lens.py` requires all four TP captures and checks, in order:

1. exact final residual agreement across TP ranks;
2. exact agreement between layer-42 hook state and SGLang's returned pre-readout
   hidden state;
3. lens/native argmax token equality;
4. top-10 overlap and native-top-token log-probability error (maximum allowed
   error 0.15 by default).

Preflight validation completed before requesting compute: shell syntax and
Python compilation passed, launch arguments were expanded with a mock base
launcher, `git diff --check` passed, and the relevant CPU suite reported
`10 passed`.

### Second launch failure: wildcard hook expansion

Allocation `57804283` loaded the model, but live registration logs showed that
the configured target `model.layers.*` matched every descendant module under
each block because SGLang uses `fnmatch`. Hooks were incorrectly attached to
attention projections, MLP components, norms, and other leaves in addition to
the 43 decoder blocks. No inference request was sent and no invalid capture was
used. The server was stopped and the idle allocation was relinquished.

This should have been caught before launch by resolving the hook spec against a
representative module tree. The corrected launcher now generates the 43 exact
names `model.layers.0` through `model.layers.42`. A test invokes SGLang's actual
`register_forward_hooks`, asserts the registered-module list equals those 43
names with zero descendants, runs one dummy pass through all registered blocks,
and verifies one `[4, hidden_size]` state per layer in the saved capture.

## 2026-08-31: validated live lens and filler-position pilot

Allocation `57804608` on `nid008281` loaded the corrected fast-profile server.
The deterministic prompt `The capital of France is` produced native token
`11111` (` Paris`). Final-layer validation passed:

- lens/native argmax equal (`11111`);
- native/lens top-10 overlap 10/10;
- layer-42 hook versus returned pre-readout hidden state max absolute difference
  0.0;
- final residual max absolute difference across four TP ranks 0.0;
- maximum native-top-token log-probability difference 0.13051, below the
  predefined 0.15 tolerance.

Artifacts are under `runs/deepseek-v4-flash/logit-lens/57804608/`, including
`native_response.json`, `validation.json`, and rank-specific captures. The
instrumented server remained alive after validation.

An exploratory one-prompt filler pilot used the existing verified Atatürk-age
task: `A=57`, `X=11`, `A+X=68`, with ten forced dot tokens. Exact token-ID
prefixes captured the last-question position (65), filler positions 68--77,
and answer-prefix position 80 at all 43 layers. Causal prefix truncation is
valid here because the residual at a position cannot depend on later tokens.

The first trigger attempt was excluded: rapid shared-file rewrites did not
produce distinct mtimes for every request, and SGLang internal forward pass IDs
were incorrectly assumed to equal request IDs. The clean rerun assigned unique
nanosecond mtimes and discovered the new capture filename after every request.
It produced passes 26, 28, ..., 48; all 12 captures were complete and exactly
equal across TP ranks.

Exploratory findings (one prompt only):

- at the last-question position, the best sum rank was 1,623 at layer 40;
- the factual value 57 reached rank 1 at layer 40 on filler 0;
- the sum first reached rank 1 at filler 5, layer 38, but was non-monotonic
  across positions/layers;
- from filler 6 onward the best late-layer sum rank was at most 8;
- at filler 9 the sum was rank 1 at layers 41 and 42 and was the model's next
  generated token;
- at the answer prefix the sum was rank 1 from layer 38 onward and was generated
  correctly.

Treat this only as a mechanism/pipeline pilot. It is not factorial evidence and
does not justify population-level claims. Next, use aligned discovery panels
from the manifest, freeze candidate layer-position sites using discovery data,
and reserve confirmation panels for the predefined tests.

## 2026-08-31: setup and final-readout reconstruction

Objective: build a Logit Lens for the local DeepSeek-V4-Flash checkpoint and
validate it by reproducing the model's native next-token prediction at the final
layer.

Environment and inputs:

- Perlmutter interactive allocation `57803794`, node `nid008396`, four
  NVIDIA A100-SXM4-80GB GPUs, tensor parallel size 4.
- Converted checkpoint:
  `model/DeepSeek-V4-Flash-0731-MoE-MXFP4-BF16`.
- SGLang A100 port and launcher documented in `DEEPSEEK_V4_A100_AUDIT.md`.
- Server launched with `DEEPSEEK_STARTUP_PROFILE=fast`, which disables CUDA
  graphs but leaves the approximately 770-second MXFP4-to-INT8 preparation
  unchanged.

Architecture finding: DeepSeek V4's layer residual is an mHC tensor with four
4096-wide streams, not a conventional 4096-wide residual. Its native final
readout is:

1. learned `hc_head` gating and sum from `[4, 4096]` to `[4096]`;
2. final RMSNorm;
3. untied vocabulary head (`head.weight`).

Accordingly, an RMSNorm-plus-unembedding lens would be incorrect. The faithful
projection is implemented in `deepseek_v4_logit_lens.py`. It loads only
`hc_head_fn`, `hc_head_base`, `hc_head_scale`, `norm.weight`, and `head.weight`
from checkpoint shard 45. Unit validation:

```text
.venv-sglang/bin/python -m pytest -q tests/test_deepseek_v4_logit_lens.py
2 passed
```

Current limitation: the first live server was started without activation hooks
or `--enable-return-hidden-states`. Preserve it after warmup as requested. The
remaining end-to-end validation requires obtaining a real final mHC state from
that process, then comparing the lens argmax/top-k with native SGLang logits.
Intermediate-layer use additionally requires capturing each decoder layer's
post-block mHC output. Avoid restarting the warmed server without explicit
coordination because repeated startup pays the expert preparation cost again.

Planned performance measurement: record eager warm TTFT, inter-token latency,
single-request decode throughput, and batched throughput. A future controlled
throughput-profile run can compare CUDA graphs; its measured startup cost from
the earlier audit is approximately 268 seconds, giving break-even as
`268 / per-request seconds saved`.

## 2026-08-31: factorial capture and transplant implementation

Objective: implement the complete crossed one-fact/two-fact protocol for reuse
by a later session that has a warm DeepSeek worker.

Added files:

- `dsv4_factorial.py`: absolute-position bookkeeping, 2x2 panel construction
  with all four target rotations, exact native residual projection, exact
  numeric rank/logit/log-odds scoring, discovery-only site selection,
  activation hooks, factorial contrasts, the J-Lens eligibility gate, runtime
  precision metadata, and packed KV-record copying.
- `prepare_dsv4_factorial.py`: deterministic, fact-disjoint 18x18 discovery and
  26x26 confirmation manifests for both tasks (an approximately fivefold
  expansion over the original 8x8/12x12 pilot).
- `tests/test_dsv4_factorial.py`: CPU tests for design rotation, repeated-sum
  rejection, scoring, leakage prevention, packed-cache copies, and contrasts.

Generate a manifest without loading the model:

```bash
.venv-sglang/bin/python prepare_dsv4_factorial.py \
  --facts runs/deepseek-v4-flash/fact-knowledge/known_facts.json \
  --output runs/deepseek-v4-flash/factorial/design.json
```

The default manifest has 2,000 cells: 324 discovery and 676 confirmation cells
for each of the two tasks. All four cells in every 2x2 basic panel rotate
through the target role, giving 8,000 predefined target/donor directions and
500 independent 2x2 panels.

Worker integration procedure:

1. Render every cell with the same official chat template, few-shot prefix,
   filler text, and filler length. Use `locate_positions` on the exact rendered
   pieces, then `assert_aligned_and_single_token` separately within each panel.
   Do not discard failures: attach clean generation correctness to every cell.
2. Instantiate `NativeResidualHooks(model)`. It fails if SGLang's deferred
   cross-layer mHC mode is active. In the checked-out port this optimization is
   off by default (`SGLANG_OPT_FUSE_MHC_POST_PRE=0`). Capture every requested
   position on every layer while pre-filling one prompt at a time.
3. Set `capture.metadata["hook_dtype"]` from an actual captured tensor and add
   `runtime_metadata(...)`. Save the captures with `ResidualCapture.save`.
4. For ordinary Logit Lens, call `score_numeric_targets` with `A`, `X`, and
   `A+X` token IDs for one-fact trials, or `A1`, `A2`, and `A1+A2` for two-fact
   trials. This applies native mHC collapse, final RMSNorm, and unembedding.
5. Build discovery rows at layer-position-role granularity and call
   `candidate_sites`. It rejects confirmation rows. Freeze this site list
   before touching confirmation results.
6. Use `NativeResidualHooks.transplant` for activation patches. Run target
   prompts clean and patched and score their answer-position logits with
   `logits_metrics`; store `causal_effect` plus panel ID. Test only frozen sites
   on confirmation panels.
7. For KV interventions, prefill donor and target requests in the same worker,
   resolve their physical token locations from SGLang's request-to-token pool,
   and call `copy_cache_rows` for the selected layer/positions before target
   continuation. The known DSV4 packed layout copies the complete 584-byte
   record: FP8 values, scales, padding, and BF16 RoPE values. Do not copy these
   bytes across pools, workers, backends, or layouts. If using the separate DSA
   index cache, transplant it through its own `SetKAndS` accessor; the generic
   copier intentionally does not guess that layout.
8. Aggregate at the factorial-panel level and report main/interaction effects
   with `factorial_contrasts`. Use matched patch directions only as repeated
   measurements within panels, not as independent observations.
9. Call `should_run_jlens` before any J-Lens analysis. It permits J-Lens only
   where ordinary lens evidence is non-positive and causal evidence is
   positive.

The manifest lists same-sum/different-decomposition, operand-order, and
same-value/different-cue controls as required controls. They need factual-cue
curation after the tokenizer/model knowledge checks; they are not fabricated
automatically. Carry/no-carry balance, magnitude balance, one-token sums, and
absolute alignment likewise remain explicit validation gates because these
depend on the selected facts and the actual official tokenizer.

Validation on the login node (no model load or Slurm action):

```text
9 passed
manifest: 2,000 cells, 8,000 donor directions, 500 panels
```

### 2026-08-31 enlarged default design

At the user's request, the default grids were increased from 8x8 discovery and
12x12 confirmation to 18x18 and 26x26. The intended manifest therefore has
2,000 clean prompts across the two tasks, 500 independent panels, and 8,000
rotated donor directions including identity controls. Causal execution cost
still scales by the number of frozen candidate sites, so an initial 5--10-site
confirmation sweep is recommended before broadening it.

## 2026-09-01: expanded filler-length discovery and confirmation

The knowledge-filtered input contains 262 facts (78 age, 99 atomic, 85 static),
so the original 18/26-axis manifest was not an appropriate limit on independent
one-fact panels. `prepare_dsv4_filler_panels.py` created the token-aware,
fact-disjoint design at
`runs/deepseek-v4-flash/factorial/filler-design-expanded.json`: 24 discovery
panels (48 facts) and 60 confirmation panels (120 different facts), aligned and
single-token-valid at filler lengths 0, 5, 10, and 20.

Allocation `57804608` captured every filler position plus the final question
and answer-prefix positions on all 43 layers for the 24 discovery panels. It
produced 96 manifests, 4,128 unique captures, and 177,504 scored rows at
`runs/deepseek-v4-flash/logit-lens/57804608/filler-grid-expanded/lens_rows.jsonl`.
Clean correctness was recorded and never used to discard cells.

Discovery selected final-filler layer 41 as the primary site. The frozen rule
is `runs/deepseek-v4-flash/factorial/frozen_filler_sites.json`. Discovery
rank-1 rates for `A+X` were 1.0%, 43.8%, and 61.5% at lengths 5, 10, and 20;
mean log-odds were -8.90, -2.50, and -0.94. Layers 40/42 were frozen as adjacent
robustness sites and layer-41 answer-prefix readout as a positive control.

The untouched 60-panel confirmation split was then queried only at the frozen
final-filler and answer-prefix positions. It produced 240 manifests and 1,680
unique captures. The primary result replicated: paired panel-mean log-odds
changed by +9.93 from length 5 to 10 (95% panel-bootstrap CI [9.11, 10.76]) and
+1.99 from 10 to 20 (95% CI [1.29, 2.69]). Artifacts are under
`runs/deepseek-v4-flash/logit-lens/57804608/filler-confirmation-frozen/`.
Allocation `57804608` was released after projection and summary.

### Failed unattended length-50/100 attempt and exact recovery

Allocation `57806912` launched the correctly instrumented fast server and
`run_dsv4_long_filler_unattended.sh`. The standard validation prompt generated
`Paris`, but the pipeline assumed it corresponded to capture `pass00000`.
The returned final hidden state and that capture differed
(`max_abs=96764.734375`), so validation failed closed before any length-50/100
experimental request. The failure marker is
`runs/deepseek-v4-flash/logit-lens/57806912/long-filler-pipeline/FAILED.json`.
The allocation was automatically released; no long-filler data were produced.

Most likely an internal startup/profile forward consumed pass 0. Do not weaken
validation or choose a capture after seeing similarity. Recovery steps:

1. Snapshot rank-0 capture filenames before the validation prompt.
2. Arm `CAPTURE_NEXT` with a unique nanosecond mtime, exactly as in
   `capture_dsv4_factorial_grid.py`.
3. Send the validation request, discover the exact single new rank-0 filename,
   and require that filename on ranks 1--3.
4. Pass its numeric ID to `validate_dsv4_logit_lens.py --pass-id`.
5. Add a regression test proving validation does not assume pass 0 when older
   captures or startup forwards exist.
6. Only after exact final-state equivalence passes, run the already-rendered
   `filler-{discovery,confirmation}-rendered-k{50,100}.json` inputs. Discovery
   saves every filler token; confirmation stays restricted to the previously
   frozen final-filler and answer-prefix sites.
7. Preserve automatic completion/failure markers and allocation release.

Relevant files are `run_dsv4_long_filler_unattended.sh`,
`capture_dsv4_factorial_grid.py`, `analyze_dsv4_factorial_grid.py`, and
`summarize_frozen_filler.py`. A replacement should request four A100 80-GB GPUs
for four hours on account `m5258_g`.

### 2026-09-01: length-50/100 validation fix and unavailable allocation

Corrected the unattended recovery so validation no longer assumes capture pass
0. `probe_dsv4_logit_lens.py` now snapshots rank-0 captures, arms `CAPTURE_NEXT`
with a distinct nanosecond mtime, sends the deterministic validation request,
requires exactly one new filename, verifies that filename on all four TP ranks,
and writes its numeric pass ID. `run_dsv4_long_filler_unattended.sh` passes that
explicit ID to `validate_dsv4_logit_lens.py`. It still fails closed before any
experimental request if final-state equivalence fails.

Added regression coverage with pre-existing startup captures and an incomplete
TP capture case. Shell/Python syntax, `git diff --check`, and the targeted tests
passed (`2 passed`). Added `run_dsv4_long_filler_allocation.sh` to keep server
startup, logs, pipeline execution, and cleanup together inside the allocation.

Authorized interactive request `57824046` asked for one node, four A100 80-GB
GPUs, four hours, and account `m5258_g`. It remained pending for the bounded
600-second immediate window and ended with `Unable to allocate resources:
Connection timed out`. It never entered RUNNING state: no model was loaded, no
GPU time was consumed, and no length-50/100 data were produced. The corrected
pipeline remains ready for a later allocation attempt.

### 2026-09-01: split launch blocked by NERSC Scratch hardware outage

Split the long-filler pipeline into independent length-specific runs. The
allocation wrapper now requires a filler length of 50 or 100, and each run
performs its own exact-pass validation, discovery capture and projection,
confirmation capture and projection, summary, markers, and cleanup. The planned
requests are two hours for length 50 and four hours for length 100; each pays
the model preparation cost but has no cross-allocation capture dependency.

Authorized interactive requests `57824624` (length 50, two hours) and
`57824633` (length 100, four hours) both remained pending with reason
`Licenses`. `scontrol show job` showed `Licenses=scratch:1`; the license pool
showed all 1,000,000 lowercase `scratch` licenses reserved. NERSC reported that
Perlmutter Scratch suffered a hardware failure shortly before 06:00 and part of
the filesystem would remain inaccessible until vendor replacement of the
failed component. Both requests reached their bounded 600-second immediate
timeouts and were revoked without entering RUNNING. No model load, GPU use, or
experimental request occurred. Do not bypass the Scratch hold: the checkpoint,
inputs, captures, and outputs are all Scratch-resident. Resubmit the two split
runs only after NERSC reports Scratch restored.

### 2026-09-01: browser-based plotting starter

Added `addition_accuracy_plot.py` and `notebooks/addition_accuracy.ipynb` as a
minimal NERSC Jupyter workflow. The notebook plots exact-answer accuracy against
filler lengths 0, 10, 20, 50, and 100 for the completed one-fact and two-fact
addition experiments. Inputs are the existing `one-fact-addition-full-batched`
and `two-fact-addition-full-under-999` summaries. Error bars are two-sided 95%
Wilson binomial intervals, using 262 facts and 130 fact pairs as the respective
units. Plot logic stays in an autoreloaded Python module so notebook iteration
does not require transferring figures to a laptop.

Registered the existing `.venv-sglang` environment as the user Jupyter kernel
`mech-int-sglang`, displayed as `mech-int (sglang)`. Targeted tests passed
(`2 passed`), notebook JSON parsed successfully, and a headless 800x500 PNG
smoke render succeeded. The smoke image was written only to `/tmp`; the notebook
exports final PDF/PNG artifacts under `runs/deepseek-v4-flash/plots/` on demand.

### 2026-09-01: two-fact accuracy expansion goal after Scratch recovery

Goal for the next available inference run: evaluate approximately 500
additional independent two-fact addition pairs. Preserve paired evaluation of
the same pair at baseline and each filler condition; do not select or discard
pairs based on observed correctness. Evaluate filler lengths 10, 20, 50, and
100 for every pair. None is primary: treat the four baseline comparisons as one
prespecified family and report multiplicity-adjusted inference across all four.
With all four filler lengths plus baseline, 500 new pairs imply approximately
2,500 generation requests.

Motivation and power basis: the completed 130-pair experiment at length 50 had
6 baseline-correct/filler-incorrect pairs and 13
baseline-incorrect/filler-correct pairs, an accuracy change of +5.38 percentage
points and exact two-sided McNemar p=0.1671. Assuming the observed discordance
rates persist, roughly 394 total pairs (264 additional) target 80% power at
two-sided alpha 0.05. Roughly 560 total pairs (430 additional) target 80% power
using alpha 0.0125 as a conservative four-comparison correction. Adding about
500 pairs gives about 630 total pairs, exceeding both targets for the observed
length-50 effect while allowing some eligibility or integrity losses. Power can
differ by length because the observed effects and discordance rates differ; do
not imply that 630 pairs guarantees equal power for all four effects.

Before execution, construct a fact-disjoint expansion, validate one-token
targets and the under-999 constraint, freeze the manifest and four-comparison
analysis, and calculate exact paired McNemar results, multiplicity-adjusted
p-values, and paired accuracy-change intervals for every filler length.
Do not submit compute until NERSC reports Perlmutter Scratch restored and the
required inputs and output path pass bounded read/write checks.

### 2026-09-01: paired accuracy-change plot

Extended `addition_accuracy_plot.py` and the notebook with a second plot that
pairs every filler-condition result with the same fact or fact pair at
baseline. It reports the mean exact-accuracy change and a seeded 95% percentile
bootstrap interval obtained by resampling the 262 facts or 130 pairs. The
returned plot data also records wrong-to-right, right-to-wrong, and unchanged
counts. This directly estimates the filler effect and avoids treating the two
conditions as independent samples.

At 50 and 100 dots, the one-fact paired intervals exclude zero: +6.49 points
(95% bootstrap CI +1.15 to +11.83) and +8.02 points (+2.29 to +13.36),
respectively. None of the two-fact intervals exclude zero. Tests passed in the
notebook environment (`3 passed`), and the modified notebook remains valid
JSON. No model inference or Slurm work was required.

### 2026-09-01: five-shot addition prompt revision

Revised the one- and two-fact addition protocols from zero-shot to five-shot.
Each task now has five fixed, elementary, task-matched user/assistant
demonstrations. Demonstration user turns end in `Answer:` and their assistant
turns contain only the integer. The target fillers and `Answer:` are likewise
inside the final user turn; the final assistant turn is empty and begins only
after the official encoder inserts `<｜Assistant｜></think>` in `chat` mode.
The shared system prompt is now: `Solve each addition problem. After 'Answer:'
respond with only the integer answer. No explanation, no words, no reasoning,
just the number.` Run manifests use schema version 2 and record the complete
demonstration set and prompt protocol.

Updated factorial prompt splitting and absolute-position calculation so
`answer_prompt` identifies the final user-turn `Answer:` token rather than the
later assistant transition token. Added focused tests for five demonstration
turns, user-turn filler placement, and generation-prefix-aware position
indexing. A model-free render check with the pinned DeepSeek V4 encoder verified
six user turns, six assistant transitions, and the exact target suffix `. . .
. .\nAnswer:<｜Assistant｜></think>` for both tasks at `k=5`. Python syntax checks
passed under Python 3.11. Removed eager imports of the PyTorch-based fact
evaluation modules from both addition CLIs, because they made prompt-only runs
stall during irrelevant framework initialization on a login node. Both actual
prompt-only CLIs then completed at `k=5`, and all 17 one- and two-fact unit
tests passed using the lightweight Python 3.11 environment. The separate
torch-dependent factorial suite was not run; no model load, inference request,
Slurm action, or GPU work was performed.

Follow-up prompt correction: for a condition with filler length `k`, the same
`k` space-separated literal periods are now inserted immediately before
`Answer:` in all five demonstration user turns as well as the final target user
turn. There is no `Filler:` label. The official encoder remains solely
responsible for inserting `<｜Assistant｜></think>` after each user turn. At
`k=5`, actual prompt-only rendering for both tasks verified exactly six copies
of `. . . . .\nAnswer:<｜Assistant｜></think>`, no filler label, and generation
starting immediately after the final native non-thinking transition. All 17
addition tests passed.
