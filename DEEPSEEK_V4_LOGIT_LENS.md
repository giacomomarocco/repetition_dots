# DeepSeek V4 Logit Lens lab log

## 2026-08-31: launch sequencing failure

The first server was launched without activation hooks or hidden-state return.
It paid 757.7 seconds for expert preparation and then successfully served, but
could not export intermediate mHC residuals because SGLang registers hooks at
model construction and has no runtime HTTP operation for adding them. Existing
`filler/dsv4/factorial.py` already contained `NativeResidualHooks` and native mHC
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

`scripts/dsv4/probe_logit_lens.py` defines the deterministic first request and preserves
the native token, top log probabilities, and returned final state.
`scripts/dsv4/validate_logit_lens.py` requires all four TP captures and checks, in order:

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
projection is implemented in `filler/dsv4/lens.py`. It loads only
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

- `filler/dsv4/factorial.py`: absolute-position bookkeeping, 2x2 panel construction
  with all four target rotations, exact native residual projection, exact
  numeric rank/logit/log-odds scoring, discovery-only site selection,
  activation hooks, factorial contrasts, the J-Lens eligibility gate, runtime
  precision metadata, and packed KV-record copying.
- `scripts/dsv4/prepare_factorial.py`: deterministic, fact-disjoint 18x18 discovery and
  26x26 confirmation manifests for both tasks (an approximately fivefold
  expansion over the original 8x8/12x12 pilot).
- `tests/test_dsv4_factorial.py`: CPU tests for design rotation, repeated-sum
  rejection, scoring, leakage prevention, packed-cache copies, and contrasts.

Generate a manifest without loading the model:

```bash
.venv-sglang/bin/python -m scripts.dsv4.prepare_factorial \
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
one-fact panels. `scripts/dsv4/prepare_filler_panels.py` created the token-aware,
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
   `scripts/dsv4/capture_factorial_grid.py`.
3. Send the validation request, discover the exact single new rank-0 filename,
   and require that filename on ranks 1--3.
4. Pass its numeric ID to `python -m scripts.dsv4.validate_logit_lens --pass-id`.
5. Add a regression test proving validation does not assume pass 0 when older
   captures or startup forwards exist.
6. Only after exact final-state equivalence passes, run the already-rendered
   `filler-{discovery,confirmation}-rendered-k{50,100}.json` inputs. Discovery
   saves every filler token; confirmation stays restricted to the previously
   frozen final-filler and answer-prefix sites.
7. Preserve automatic completion/failure markers and allocation release.

Relevant files are `run_dsv4_long_filler_unattended.sh`,
`scripts/dsv4/capture_factorial_grid.py`, `scripts/dsv4/analyze_factorial_grid.py`, and
`scripts/dsv4/summarize_frozen_filler.py`. A replacement should request four A100 80-GB GPUs
for four hours on account `m5258_g`.

## 2026-09-04: transplant runner gates, answer scores, and allocation attempts

### 2026-09-04: validated single-site factorial patching run

Allocation `57927452` ran on `nid008192` with four A100-SXM4-80GB GPUs,
interactive QOS, account `m5258_g`, and the `gpu&hbm80g` constraint.  The
instrumented server disabled fused mHC post/pre, returned hidden states, and
loaded both native capture and transplant hooks.  Final-layer equivalence
passed with zero returned-hidden error, argmax agreement, top-10 overlap 10/10,
and maximum native top-logprob error 0.13051 (threshold 0.15).

Ordinary cross-request radix reuse failed for the 85--87-token prompts because
the hybrid-SWA server reported zero cached tokens.  The runner was changed to
use SGLang streaming sessions: patch-boundary requests append true prompt
chunks with zero generated tokens, and the final append generates the answer.
Boundary-matched unpatched baselines are required because chunked and
monolithic execution can differ numerically.

Donor captures from job `57804608` were not sufficiently invariant on the new
runtime, so all 24 panels (96 cells) were recaptured on the active server and a
new manifest was built at
`runs/deepseek-v4-flash/transplants/57927452-streaming/one-fact-activation-patching-current.json`.
With current-runtime captures, all 1,920 single-site identity controls passed
exact-token invariance with zero target-logprob error.  The completed outputs
are under `runs/deepseek-v4-flash/transplants/57927452-streaming/full/`:
`chunked-clean.jsonl` has 288 target/site baselines,
`identity-single-site.jsonl` has 1,920 controls, and
`substantive-single-site.jsonl` has 3,840 unique patches across layers 33--42,
sites `filler_5` and `filler_10`, and donor roles `left` and `both`.

The two-position `filler_5+filler_10` condition remains excluded: a layer-38
identity control reproducibly changed the generated token from 7207 to 2875,
despite target-logprob error 0.08373 being within tolerance.  Partial outputs
from failed gates are retained for diagnosis but are not valid experiment data.
Preliminary mean target-logprob deltas versus matched baselines range from
-0.09047 (`both`, `filler_10`, layer 33) to +0.05459 (`both`, `filler_5`,
layer 39).  Formal uncertainty estimates and plots remain next steps.

Extended `scripts/dsv4/run_activation_patching.py` before model launch so the
same controller can write fresh clean baselines, select an explicit pilot by
target/site/layer/donor role, and fail closed when an identity patch changes
the generated token or changes the target-answer log probability by more than
0.15. Every patched-prefix replay and final replay now requests and records the
log probabilities of both the target cell's answer token and the donor cell's
answer token, together with their token IDs and integer answer values. This
supports direct likelihood-shift analysis rather than only exact-answer
accuracy. The focused factorial/transplant/runner suite passed (13 tests), and
Python compilation plus `git diff --check` passed.

The planned preflight pilot is one target at layer 42: one fresh clean
baseline, identity patches at `filler_5`, `filler_10`, and both positions, and
both `left` and `both` donor roles at those same three sites. It therefore
covers every requested intervention category in 10 evaluations before the
full 96 clean, 2,880 identity, and 5,760 substantive evaluations.

After confirming that no interrupted allocation existed, authorized requests
`57921311` and `57921370` each asked for one interactive node, four GPUs, the
quoted `gpu&hbm80g` constraint, four hours, and account `m5258_g`. Both stayed
pending for priority until `salloc` timed out and revoked them. Neither entered
RUNNING, so no node was inspected, no model was loaded, and no experimental
request ran. The next attempt must retain the exact 80-GB constraint and begin
with `nvidia-smi` before server launch.

## 2026-09-03: one-fact residual-transplant preparation

The activation-patching design was frozen to filler length 20, positions
`filler_5`, `filler_10`, and both positions, with independent patches at each
of layers 33--42. Substantive donors are the factorial `left` role (different
fact, same/addend-matched addend) and `both` role (different fact and addend).
The discovery manifest at
`runs/deepseek-v4-flash/factorial/one-fact-activation-patching-k20-layers33-42.json`
contains 96 targets, 5,760 substantive runs, 2,880 identity controls, and 96
clean baselines. It resolves donor capture pass IDs before model launch and
records that layers are swept independently.

`filler.dsv4.hooks.make_layer_transplant_hook` implements an externally
triggered, one-pass post-block mHC transplant. Each TP rank loads its matching
saved donor tensor, changes only the final token at the requested layer, and
writes a rank-specific acknowledgement. The controller must truncate the
request at the intervention token, then replay all later prompt tokens through
the patched cache; a full parallel prefill is not a valid substitute. Ten CPU
tests covering the transplant hook and factorial utilities passed. Before an
expensive launch, the remaining controller must enforce cache flushing between
conditions, all-rank acknowledgements, final-layer equivalence, identity-patch
invariance, and observed cache reuse during continuation.

### 2026-09-01: length-50/100 validation fix and unavailable allocation

Corrected the unattended recovery so validation no longer assumes capture pass
0. `scripts/dsv4/probe_logit_lens.py` now snapshots rank-0 captures, arms `CAPTURE_NEXT`
with a distinct nanosecond mtime, sends the deterministic validation request,
requires exactly one new filename, verifies that filename on all four TP ranks,
and writes its numeric pass ID. `run_dsv4_long_filler_unattended.sh` passes that
explicit ID to `scripts/dsv4/validate_logit_lens.py`. It still fails closed before any
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

Added `filler/addition/accuracy_plot.py` and `notebooks/addition_accuracy.ipynb` as a
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

Extended `filler/addition/accuracy_plot.py` and the notebook with a second plot that
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

### 2026-09-03: five-shot one- and two-fact accuracy evaluation

Objective: test whether forced dot filler increases exact-answer accuracy under
the corrected five-shot protocol. Allocation `57906667` used four A100 80-GB
GPUs on `nid008476` with the `m5258_g` account and interactive QOS. DeepSeek V4
Flash was served by `run_deepseek_v4_a100.sh` with the `throughput` startup
profile and the existing A100 port. The endpoint smoke test succeeded before
evaluation. Focused evaluator tests passed (`19 passed`), and full prompt-only
preflights produced the expected 1,310 one-fact and 1,300 two-fact prompts.

Commands (both used greedy decoding, seed 42, and filler lengths 0, 10, 20, 50,
and 100):

```bash
.venv-sglang/bin/python -m scripts.addition.one_fact \
  --facts runs/deepseek-v4-flash/fact-knowledge/known_facts.json \
  --max-facts 262 --filler-lengths 0 10 20 50 100 \
  --output-dir runs/deepseek-v4-flash/one-fact-addition-5shot-full

.venv-sglang/bin/python -m scripts.addition.two_fact \
  --facts runs/deepseek-v4-flash/two-fact-addition-full-under-999/eligible_facts.json \
  --max-pairs 260 --filler-lengths 0 10 20 50 100 \
  --output-dir runs/deepseek-v4-flash/two-fact-addition-5shot-cyclic
```

All 2,610 results completed. Accuracy and paired changes from the no-filler
condition were:

| Task | Dots | Correct | Accuracy | Paired change | Paired bootstrap 95% CI | Exact McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| One fact | 0 | 137/262 | 52.3% | — | — | — |
| One fact | 10 | 194/262 | 74.0% | +21.8 pp | +15.6 to +27.9 pp | 1.70e-11 |
| One fact | 20 | 182/262 | 69.5% | +17.2 pp | +11.1 to +23.3 pp | 1.59e-7 |
| One fact | 50 | 206/262 | 78.6% | +26.3 pp | +20.2 to +32.4 pp | 2.77e-15 |
| One fact | 100 | 214/262 | 81.7% | +29.4 pp | +22.9 to +35.9 pp | 6.61e-17 |
| Two facts | 0 | 20/260 | 7.7% | — | — | — |
| Two facts | 10 | 13/260 | 5.0% | -2.7 pp | -6.2 to +0.8 pp | 0.210 |
| Two facts | 20 | 17/260 | 6.5% | -1.2 pp | -5.0 to +2.7 pp | 0.701 |
| Two facts | 50 | 15/260 | 5.8% | -1.9 pp | -5.4 to +1.5 pp | 0.383 |
| Two facts | 100 | 17/260 | 6.5% | -1.2 pp | -5.0 to +2.7 pp | 0.690 |

The paired bootstrap used 200,000 resamples of facts or cyclic fact pairs with
a fixed NumPy RNG seed per task and condition. McNemar p-values are two-sided
exact binomial tests on discordant matched outcomes. One-fact gain/loss counts
were 67/10, 60/15, 77/8, and 86/9 at 10, 20, 50, and 100 dots. Two-fact counts
were 8/15, 12/15, 8/13, and 11/14. Thus filler robustly increased one-fact
accuracy in this protocol, but there is no evidence it increased two-fact
accuracy; all two-fact point estimates were lower than baseline. This is an
accuracy result for forced dot filler repeated in all five demonstrations and
the target turn, not a general result for arbitrary extra computation tokens.

Durable outputs include `run_config.json`, `prompts.json`, fsynced
`results_progress.jsonl`, `results.json`, `summary.json`, and `evaluation.log`
in each output directory above. Server logs are in
`runs/deepseek-v4-flash/server-57906667/server.log`.

### 2026-09-03: exact-count heads-up rerun and residual-capture preflight

Prepared a paired rerun of the established one- and two-fact addition
experiments in which every nonzero filler condition tells the model the exact
number of upcoming dots in the system message. The baseline retains the
original system message. The one-fact grid contains 262 facts and 1,310 prompts;
the two-fact grid contains the existing 200 deterministic atomic-number pairs
and 1,000 prompts. Both use lengths 0, 10, 20, 50, and 100, seed 42, the local
five-shot scaffold, greedy generation, and one-token target scoring.

`run_dsv4_addition_headsup_allocation.sh` is the end-to-end allocation workflow.
It launches the instrumented server with native post-block mHC hooks, fails
closed unless final-layer equivalence passes, runs both behavioral grids, then
retains stratified trajectories for 16 deterministic pairs per experiment.
Each trajectory covers all 43 layers at the final question token, first/last
filler tokens, every fifth filler token, and the answer prefix. This is about
1,472 tensor-parallel captures (roughly 8--9 GB at the measured prior capture
size), providing repeated examples at every filler length without duplicating
the full residual stream for all 2,310 behavioral prompts.

Raw tensors and request manifests will be under
`runs/deepseek-v4-flash/logit-lens/$SLURM_JOB_ID/addition-headsup/trajectories/`;
`scripts/dsv4/analyze_addition_headsup.py` projects them with the native mHC
readout and writes layer/position target ranks, logits, and log-odds to
`lens_rows.jsonl`. Prompt-only preflight verified all 2,310 prompts and the
focused one-/two-fact suites passed (24 tests). No Slurm job was submitted in
this preflight; submission still requires explicit approval.

### 2026-09-03/04: exact-count heads-up run completed

Ran the approved end-to-end workflow as Slurm job `57909905` on four 80 GB
GPUs (`nid008633`, interactive QOS, account `m5258_g`). The native final-layer
equivalence gate passed before experimental inference: the argmax token agreed,
top-10 overlap was 10/10, returned-hidden maximum absolute difference was zero,
and maximum native top-logprob absolute error was 0.13051 against the 0.15
threshold.

The one-fact run completed 1,310/1,310 prompts (262 paired facts). Exact-answer
accuracy was 137/262 (52.3%) at baseline, 189/262 (72.1%) at 10 dots, 189/262
(72.1%) at 20 dots, 209/262 (79.8%) at 50 dots, and 200/262 (76.3%) at 100
dots. Mean target log probability changes versus baseline were +1.196, +0.887,
+1.372, and +1.449 respectively.

The atomic two-fact run completed 1,000/1,000 prompts (200 paired fact pairs).
Exact-answer accuracy was 28/200 (14.0%) at baseline, 37/200 (18.5%) at 10
dots, 24/200 (12.0%) at 20 dots, 24/200 (12.0%) at 50 dots, and 29/200 (14.5%)
at 100 dots. Mean target log probability changes versus baseline were +0.243,
+0.136, -0.720, and +0.317 respectively. These descriptive results are not yet
accompanied by paired confidence intervals or significance tests.

Trajectory capture completed for 16 deterministic pairs per experiment and all
five conditions. The exact sampling rule produced 1,600 capture requests (50
per pair), correcting the earlier rough estimate of 1,472. Native projection
processed 160 manifests and wrote 68,800 rows. Integrity checks found 1,310 and
1,000 durable progress rows, 160 trajectory manifests, 68,800 lens rows, and a
`COMPLETE.json` marker. Behavioral outputs are in
`runs/deepseek-v4-flash/one-fact-addition-full-headsup/` and
`runs/deepseek-v4-flash/two-fact-addition-5shot-atomic-200-headsup/`; validation,
trajectory manifests, and projected rows are in
`runs/deepseek-v4-flash/logit-lens/57909905/addition-headsup/`, with 8.6 GB of
raw tensor-parallel captures in the sibling `captures/` directory.

### 2026-09-09: simultaneous one-fact patching campaign implemented and preflighted

Objective: distinguish transport of fact information from answer information
by replacing complete post-block mHC residuals at separate zero-based
`filler_5` and `filler_10` positions. Cross simultaneous layer sets 33–42 and
0–42, full downstream versus answer-only recomputation, and different-fact
donors with either the same or a different addend. Score target `A_t+X_t`,
donor `A_d+X_d`, and mixed `A_d+X_t` sums with explicitly requested,
full-vocabulary-normalized natural log probabilities. Preserve donor/mixed
aliases as separate labels while requesting their shared token only once.

Inputs are the existing
`runs/deepseek-v4-flash/factorial/filler-discovery-rendered-k20.json` and local
`model/DeepSeek-V4-Flash-0731-MoE-MXFP4-BF16` checkpoint. Preserve the historical
saved prompts, including their earlier assistant-filler format; do not rerender
them with the later five-shot scaffold. All 24 panels and 96 cells are retained,
without filtering on clean correctness. Prompt lengths are 85–87 tokens. The
first sorted panel is `discovery-filler-0743614aafcd1ceb0a07`; its filler sites
are absolute positions 67 and 72 and its answer-predicting prompt position is
84. The controller always derives that last position as `len(input_ids)-1`.
The saved candidates use 73 distinct vocabulary tokens, all verified as
canonical single-token integers with the actual local tokenizer.

Implementation:

- `NativeResidualHooks.transplant_layers` in `filler/dsv4/factorial.py` and
  `filler/dsv4/campaign_hook.py` share the residual replacement operation. The
  existing single-layer API and runner remain available. Answer-only passes
  restore clean post-block states at every layer and every position except
  the chosen filler and final prompt position. Intermediate computations
  execute before restoration.
- `filler/dsv4/patching.py` defines the frozen grid, candidate mappings, scoring
  and checksummed journal. `filler/dsv4/patching_campaign.py` controls both
  phases without a server restart at pilot completion. Every active panel
  receives fresh clean captures and two mode-matched baselines per target;
  all 32 panel identity controls must pass before its substantive trials.
- Clean captures contain all positions and all 43 layers on all four ranks.
  Every intervention retains the filler and answer residuals at all layers,
  plus per-layer exact replacement/restoration acknowledgements on every
  rank. Captures are fsynced, atomically renamed and checksummed; workers
  cache the four panel captures for reuse.
- The first active panel in each runtime additionally runs 32 layer-42-only
  filler diagnostics (four targets × two sites × two modes × two donor types).
  These are extra validation requests, outside the substantive/control totals.
  Native final-layer validation checks returned-hidden equality, agreement
  across TP ranks, argmax, and top/requested candidate log probabilities
  against the native mHC readout, with maximum error 0.15. Identity and
  layer-42 controls require unchanged generated tokens and candidate errors
  at most 0.15. Any failure stops progression and preserves diagnostics.
- Every completed result is flushed and fsynced before an atomic progress
  update. Recovery reconciles checksummed result IDs, quarantines incomplete
  trailing records, preserves complete records even if progress was not
  updated, and reruns unfinished work. New runtimes repeat validation and
  obtain fresh baselines/captures; completed results keep their original
  baseline references. Even previously completed identity controls are
  rechecked before unfinished panel trials in a new runtime.
- `filler/dsv4/patching_analysis.py` recomputes stored deltas from raw responses,
  validates referenced artifacts, and produces panel means, condition tables,
  paired factor comparisons, PNG/PDF plots for all three sums and a Markdown
  report. Intervals use 20,000 paired panel bootstrap resamples, seed 42.
  `COMPLETE.json` and a completion lab-log entry are written only after
  inference, integrity checks and reporting succeed.

The campaign grid is 64 substantive trials, 32 identity controls and eight
baseline cells in the pilot; full discovery includes these results in totals
of 1,536 trials, 768 identity controls and 192 baseline cells. Restarts may add
baseline and recheck records without duplicating planned result IDs. An
uninterrupted run therefore makes 2,528 inference passes including the 32
extra layer-42 diagnostics. Remaining walltime and projected work use measured
pass durations, including baselines and rechecks; reporting time is excluded
from the projection. Actual throughput and four-hour completion remain unknown
until inference runs.

Model-free preflight command:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  TOKENIZERS_PARALLELISM=false .venv-sglang/bin/python \
  -m scripts.dsv4.one_fact_patching prepare
```

Preflight passed all 33 focused CPU tests, including native residual semantics,
single-layer compatibility, numerical equivalence gate behavior, all-rank
acknowledgements, candidate aliases, automatic pilot-to-discovery progression,
failed-gate stops, interrupted trials, interrupted identity validation,
result/progress write interruption and incomplete trailing records. A simulated
two-panel campaign also verified the analysis tables and plots. Actual-tokenizer
render alignment, all 48 checkpoint shard metadata records, Python compilation,
shell syntax and working-tree whitespace checks passed. The first real-data
preflight exposed a legacy schema difference: split labels exist on cells but
not at the rendered manifest's top level. Preparation now validates those cell
labels directly; a regression test rejects any confirmation cell. The environment
emitted its existing Transformers rope-parameter warning and MUNGE socket
messages on interpreter exit; the tests themselves succeeded.

Prepared artifacts: [manifest](runs/deepseek-v4-flash/one-fact-patching-discovery/manifest.json),
[preflight report](runs/deepseek-v4-flash/one-fact-patching-discovery/preflight.json),
and [test log](runs/deepseek-v4-flash/one-fact-patching-discovery/preflight-tests.log).
Frozen configuration hash:
`4e28d749d46bebb0142db341808030c7f6e70572c85c6a8a079f49f178aa43d8`.
The manifest hashes relevant workspace/runtime sources, tokenizer/config files,
and records checkpoint shard sizes/mtimes. Workspace revision is `10763c9`;
SGLang is `1c0019da7579db73223195f25b0eed3882dff24e`, the A100 port is
`d3987e718f0f0d835e97b1ff43a95af282494c85`, and the tested PyTorch is
`2.11.0+cu130`. Pre-existing working-tree changes are retained and no commit
has been made.

Launch, only after explicit Slurm approval:

```bash
salloc --account=m5258_g --qos=interactive --nodes=1 --gpus=4 \
  --constraint='gpu&hbm80g' --time=04:00:00 \
  srun --ntasks=1 --cpus-per-task=128 --cpu-bind=none --gpus=4 bash \
  /pscratch/sd/m/marocco/sandbox/mech_int/run_one_fact_patching_allocation.sh
```

This explicitly requests the GPU counterpart of the default project account
(corrected after the launch attempt below). Four hours is the documented interactive
GPU limit; the quoted memory constraint selects 80 GB nodes
([NERSC QOS policy](https://docs.nersc.gov/jobs/policy/#perlmutter-gpu)).
The controller verifies the allocation, visible A100 capacities and actual
server configuration. The launcher uses TP=4 with prefix caching, prefill
chunking, CUDA graphs, piecewise graphs and overlap scheduling disabled;
the base launcher sets `SGLANG_OPT_FUSE_MHC_POST_PRE=0` before model construction.
All hooks, tensor capture logic and hidden-state return are enabled at startup.
The controller owns the server through both phases and shuts down its own
server process group only when the campaign finishes or stops. It does not
submit, cancel, requeue or modify Slurm jobs.

For recovery, rerun the same launcher and campaign root inside a newly approved
allocation. Configuration changes after results exist require a new root;
source hashes prevent silently mixing implementations. Once inference is
complete, reporting alone can be resumed with
`.venv-sglang/bin/python -m scripts.dsv4.one_fact_patching analyze` on a compute
node. No Slurm submission, model load, GPU validation or scientific trial was
performed during this implementation session. Those steps await allocation
approval; the scientific campaign is not yet complete.

### 2026-09-09: approved launch and durable NERSC request corrections

The user approved the four-hour, four-A100 campaign and asked that routine GPU
allocation knowledge be documented so these failures do not recur. Added a
standard GPU allocation procedure to `AGENTS.md`, with an executable command,
the required GPU account, internal QOS alias, GPU visibility/CPU binding,
UTC scheduler timestamps and read-only account inspection commands. The
durable setup record `~/docs/PERLMUTTER_CODEX_SETUP.md` was read before editing
the project guidance; no authentication or installation settings were changed.

The first submission omitted an account and Slurm rejected it without creating
a job. Read-only associations confirmed default CPU account `m5258` and its GPU
counterpart `m5258_g`. NERSC requires the `_g` account for GPU requests
([interactive job instructions](https://docs.nersc.gov/jobs/interactive/)).
The corrected request was granted as job `58128140` on `nid008653`, with four
A100 GPUs, 128 logical CPUs and a four-hour limit. It ran from 20:23:06 to
20:23:13 UTC and exited before any model load: the controller incorrectly
rejected NERSC's internal `QOS=gpu_interactive` name for the requested public
`--qos=interactive` queue.

Fixed the controller to accept both names, and to request `TZ=UTC` from
`scontrol` before parsing `EndTime`. Verified on the real completed job that
`TZ=UTC` changes Slurm's 13:23 Pacific timestamp to 20:23 UTC. Added three
regression tests covering the alias, rejection of other QOS/time limits and
explicit UTC queries. Updated preflight and launcher examples to name
`m5258_g` and give the single TP=4 server step 128 logical CPUs with
`--cpu-bind=none`. All 36 focused tests passed before retrying.

The new frozen configuration hash is
`c39f97284646a6c0df6a63be4e533acc885132cdbf0759bd2c0fa1d9ce89bbab`.
The experimental grid is unchanged and no scientific result existed when this
configuration was regenerated. Early attempts are recorded in
[launch-attempts.json](runs/deepseek-v4-flash/one-fact-patching-discovery/launch-attempts.json).

The corrected retry was granted as job `58128414` on `nid008340`, starting
2026-09-09 at 20:39:03 UTC with an end time of 2026-09-10 at 00:39:03 UTC.
The controller accepted the internal QOS and verified four
`NVIDIA A100-SXM4-80GB` devices, each reporting 85,093,777,408 bytes.
Runtime `58128414-a80df2f1f227` owns the instrumented model server; its
[runtime metadata](runs/deepseek-v4-flash/one-fact-patching-discovery/runtimes/58128414-a80df2f1f227/runtime.json)
and [server log](runs/deepseek-v4-flash/one-fact-patching-discovery/runtimes/58128414-a80df2f1f227/server.log)
are durable. Scientific validation and trials follow server startup.

Job `58128414` finished weight processing in about 758 seconds and registered
all 43 hooks on all four ranks, but exited at 20:53:14 UTC before any scientific
request. The HTTP bind failed with `address already in use` on port 30002.
Root cause: the new controller exported `SGLANG_PORT=30002` to choose the HTTP
port; the pinned SGLang `shm_broadcast.py` calls `get_open_port()`, which also
uses that environment variable. The broadcast channel therefore claimed the
HTTP port during initialization. This also explains why readiness polling
began timing out, rather than receiving connection refusal, during loading.

Fixed the controller to pass `--port` explicitly and remove `SGLANG_PORT` from
the server environment, including inherited values. Port preflight now tests
binding, not merely connecting, and the added regression test exercises the
actual pinned `get_open_port()` function: it reproduces the collision with
the exported variable and verifies independent port selection in the corrected
environment. Added both relevant SGLang networking sources to manifest hashes.
All 37 preflight tests passed. The new configuration hash is
`ddbcae2e18042debfd094ae81dba85647a9b73e15f65b3e30084f552e0987e5f`.
No experiment result existed before this correction.

In response to the user's question about recurrence, searched seven historical
server logs (jobs 57794970, 57906667, 57909905, 57912784, and three 57927452
variants). None contained an earlier `address already in use` / port-bind
failure. This collision was introduced by the new campaign controller;
the cause and prevention are now documented in `AGENTS.md`.

At the user's request, paused for a fresh context window on 2026-09-09 at
20:58 UTC. The last retry tool call was interrupted. A subsequent read-only
`squeue --me` returned no queued/running jobs; no new runtime directory or
scientific result journal existed. Verified the latest frozen source hashes
and the removal of `SGLANG_PORT` from the server environment. Full continuity
and the next launch command are in [ONE_FACT_PATCHING_HANDOFF.md](ONE_FACT_PATCHING_HANDOFF.md).

### 2026-09-09 21:16 UTC: resumed approved pilot and discovery campaign

Read the handoff and project guidance, verified frozen configuration
`ddbcae2e18042debfd094ae81dba85647a9b73e15f65b3e30084f552e0987e5f`,
all recorded source hashes and checkpoint shard metadata, and confirmed no
existing Slurm allocation or scientific result journal. Repeated the five
focused preflight test modules: **37 passed in 11.28 seconds**. Inspected the
native residual replacement, full-prompt campaign hooks and native mHC readout.
The launcher removes `SGLANG_PORT`, passes the HTTP port explicitly, enables
hidden-state return and all 43 layer hooks, and disables fused mHC post/pre,
prefix reuse, prefill chunking, graphs and overlap scheduling.

Resumed the already approved standard four-hour command from `AGENTS.md`
through host scheduler access. Job **58129149**, account `m5258_g`, actual
QOS `gpu_interactive`, started on **nid008272** at **21:10:10 UTC** and expires
on **2026-09-10 at 01:10:10 UTC**. The single TP=4 step has 128 logical CPUs,
binding disabled and four `NVIDIA A100-SXM4-80GB` devices, each reporting
85,093,777,408 bytes. Runtime metadata and server logs are under
`runs/deepseek-v4-flash/one-fact-patching-discovery/runtimes/58129149-6a8c8f39011a/`.
At 21:16 UTC all 48 shards had been read and weight processing remained active;
no scientific validation or results had completed yet. Native final-layer
equivalence, panel identity gates and first-panel layer-42 controls remain
required before progression; the controller continues pilot to full discovery
on this same server and allocation. No source implementation changes or
commits were made during this resume preflight.

At 21:24:32 UTC weight processing completed on all ranks (757.72–758.07 s),
and HTTP startup succeeded at 21:24:35 UTC. The first scientific baseline
passed native equivalence: returned hidden states and all four ranks matched
exactly, argmax matched, maximum top/candidate logprob error 0.125 against the
0.15 limit. By 21:29 UTC all 64 pilot trials, 32 identity controls, eight
baseline cells, 32 layer-42 diagnostics and pilot artifact checks passed.
The controller automatically entered full discovery on the same server.

### 2026-09-09: one-fact simultaneous patching discovery completed

Campaign `ddbcae2e18042debfd094ae81dba85647a9b73e15f65b3e30084f552e0987e5f`: all 1,536 substantive trials, 768 identity controls and 192 target/mode baseline cells completed. Pilot results were retained in discovery totals. Each runtime passed native final-layer equivalence; each panel passed identity controls before substantive trials. The first active panel per runtime also passed all 32 layer-42 filler-only controls. Every stored candidate delta was recomputed from its original raw response and matched runtime/mode baseline.

Manifest, journal and raw rank captures: `/pscratch/sd/m/marocco/sandbox/mech_int/runs/deepseek-v4-flash/one-fact-patching-discovery`. [Results and plots](/pscratch/sd/m/marocco/sandbox/mech_int/runs/deepseek-v4-flash/one-fact-patching-discovery/analysis/REPORT.md) include all three sums and paired panel bootstrap intervals (20,000 resamples, seed 42); no clean-correctness filtering. Interrupted runtimes retain their original baselines; unfinished work uses fresh captures. Discovery intervals are descriptive and are not adjusted for multiple comparisons.

Completion timestamp: **2026-09-09 21:56:27 UTC**. The controller exited zero
and Slurm released job 58129149 normally. All 2,528 scientific passes ran in
runtime `58129149-6a8c8f39011a` (including the 32 layer-42 diagnostics).
Inference finished around 21:49:46 UTC; artifact verification and reporting
then took about 6 minutes 41 seconds. Post-run checks confirmed unchanged
source hashes/checkpoint metadata, the completion/journal counts, 1,152 panel
means, 48 condition summaries and 96 paired comparisons. Visually inspected
the generated PNG; labels and intervals are readable. The plot is also saved
as PDF. Updated the handoff and launch-attempt record to completed status.

Patching `filler_5` raised donor-sum log probability more than `filler_10`
in all eight matched layer/mode/donor comparisons: mean paired differences
**1.71–2.49 nats**, with all descriptive 95% paired panel intervals above zero.
At `filler_5`, donor-sum increases range from **1.74 to 3.16 nats**; full
propagation exceeds answer-only restoration in all four matched comparisons
(differences **0.58–1.10 nats**, intervals above zero).

Target-sum changes are small in the panel average (**−0.078 to +0.028 nats**),
and all 16 target-sum intervals include zero. These intervals do not establish
an exactly zero effect. Donor and mixed scores coincide for same-addend donors
because the candidate token is shared. These are descriptive discovery results
with no adjustment for multiple comparisons.

### 2026-09-09: requested patch-effect table with one-sigma errors

Recomputed all 16 conditions from the completed campaign's saved journal for
the requested candidate order: target `A_t+X_t`, mixed `A_d+X_t`, donor
`A_d+X_d`. The [table](runs/deepseek-v4-flash/one-fact-patching-discovery/analysis/ONE_SIGMA.md)
reports mean paired patched-minus-clean log probabilities in nats, with
one-sigma errors defined as the standard deviation (`ddof=1`) of 20,000
whole-panel bootstrap means (24 panels, shared indices, seed 42). These are
standard errors of the mean, not individual-trial scatter. All 96 target
prompts remain included per condition, regardless of clean correctness.

Verified journal checksums/configuration, complete trial membership,
runtime/mode/target-matched baselines, paired score subtraction and four
trials per panel/condition. Reproduced all 1,152 existing panel means, 48
condition means and original 95% intervals to absolute tolerance 1e-12;
same-addend donor/mixed aliases agree exactly. The
[CSV](runs/deepseek-v4-flash/one-fact-patching-discovery/analysis/conditions_one_sigma.csv)
also retains clean/patched means, panel scatter and analytic panel standard
errors. A [reproduction script](runs/deepseek-v4-flash/one-fact-patching-discovery/analysis/summarize_one_sigma.py)
and exact command accompany the table. This was a brief local NumPy
calculation using `.venv-sglang/bin/python`; no model inference or Slurm
operation was needed. Existing campaign outputs and implementation remain
unchanged; no commit was made.

### 2026-09-09: donor-minus-target logit differences

At the user's request, computed `g(y) = z(y) - z(A_t+X_t)` for both donor-derived
candidates `A_d+X_t` and `A_d+X_d`, using the exact identity
`g(y) = ln p(y) - ln p(A_t+X_t)` within each forward pass. Reported clean gaps,
patched gaps, and paired transplant shifts `g_patched - g_clean` for all 16
conditions. Formed differences within trials before averaging the four
targets per panel; used shared 20,000 whole-panel bootstrap resamples
(24 panels, seed 42) and bootstrap standard deviations (`ddof=1`) for ±1σ
errors. This preserves donor/target and clean/patched covariance. No
clean-correctness filtering, model inference, or Slurm operation.

[Results and method](runs/deepseek-v4-flash/one-fact-patching-discovery/analysis/LOGIT_GAPS_ONE_SIGMA.md),
[full-precision CSV](runs/deepseek-v4-flash/one-fact-patching-discovery/analysis/logit_gaps_one_sigma.csv),
and [reproduction script](runs/deepseek-v4-flash/one-fact-patching-discovery/analysis/summarize_logit_gaps.py)
are saved alongside the original analysis. Execute the script with
`.venv-sglang/bin/python` from the workspace. Journal checksums/configuration,
complete trial membership, baseline matching, trial and panel difference
identities, bootstrap pairing, and same-X aliases all passed (numerical
tolerance 1e-12 where applicable).

Clean mean gaps are −10.772 ± 0.430 nats for the mixed candidate (and the
same-X donor candidate), and −10.452 ± 0.444 for different-X donor sums.
The largest shift is +3.176 ± 0.349 nats for same-X donors at `filler_5`,
layers 0–42 with full propagation; its patched gap is −7.596 ± 0.361.
All patched condition mean gaps remain negative, so the mean relative score
still favors the target even where transplanting shifts it toward the donor.

An initial count assertion conflated 2,528 scientific passes with journal
records. The journal has 2,554 records, including 26 native/panel/pilot gate
records. Corrected the assertion and the earlier one-sigma report's count;
both analyses verify every journal checksum. Scientific counts and previous
numeric results are unchanged. No commit was made.

### 2026-09-09: does the transplant preferentially carry the full donor answer?

Computed the direct different-X donor contrast
`h = z(A_d+X_d) - z(A_d+X_t)` and its paired transplant shift, using the
same saved trials and shared 24-panel bootstrap (20,000 resamples, seed 42).
Clean mean `h` is +0.320 ± 0.206 nats. At `filler_5`, the patched gap is
+1.320 to +1.563 nats. Baseline-adjusted shifts are +1.015 ± 0.282
(layers 33–42, full), +1.001 ± 0.352 (0–42, full), +1.243 ± 0.214
(33–42, answer only), and +1.241 ± 0.208 (0–42, answer only). All four
descriptive 95% paired intervals exclude zero. `filler_10` has no corresponding
positive preferential shift: full propagation shifts are −0.188 ± 0.118 and
−0.295 ± 0.134; answer-only shifts are approximately −0.023 in both layer sets.

This supports transfer of donor-specific information relevant to the full
answer beyond the simplest pure-fact-transplant interpretation. It does not
distinguish a precomputed sum from operands or other answer-related features
used downstream. Whole residuals are patched at multiple layers simultaneously,
so these results do not localize an explicit sum representation to one layer.
Mean log-probability advantages do not establish an advantage on every trial.

[Direct comparison and interpretation](runs/deepseek-v4-flash/one-fact-patching-discovery/analysis/DONOR_VS_MIXED.md)
and [CSV](runs/deepseek-v4-flash/one-fact-patching-discovery/analysis/donor_vs_mixed_one_sigma.csv)
are generated by the extended `analysis/summarize_logit_gaps.py` script under
the campaign root. Candidate pairing, baseline matching, journal checksums,
original panel means and gap identities all passed. No model inference,
Slurm operations or commits were performed.

Follow-up clarification: inspected the first saved target/donor panel's exact
prompts. The fact question and explicit addend both precede the assistant
filler positions (e.g. donor asks zinc's atomic number plus 25 before the
patched filler). A whole-residual transplant can therefore transfer distinct
fact/value and addend features together, without those features already being
combined into a sum. The final prompt row remains free to compute in
answer-only mode, so that control still permits downstream addition. This is
the concrete reason that favoring `A_d+X_d` alone does not identify whether
the sum was computed before transplantation or downstream from operand features.

### 2026-09-09: four-intervention filler-coverage follow-up prepared

Implemented the requested repeat to compare full-downstream and answer-only
recomputation for (1) all 20 fillers patched together at layers 0–42 and
(2) filler_5 patched at layers 32–37 inclusive, using both original donor
types. Indices are zero-based; complete post-block mHC residuals are replaced
on every TP rank. The existing hooks and native mHC readout are reused.
Answer-only restoration runs at every block, including after layer 37;
selected fillers and the final answer-prediction row remain free to evolve.
All-fillers mode differences concern the two intervening non-filler rows.
Comparisons across intervention families change both position and layer scope
and cannot identify an isolated position effect.

Preparation at 23:18 UTC froze configuration
`5853c9f001996c60a4473e36be00f75659d66e54171b51805636d3dd86abf3ec`
in [one-fact-patching-filler-coverage](runs/deepseek-v4-flash/one-fact-patching-filler-coverage/PREPARED.md).
Exact saved prompts, all 24 panels/96 targets, input/tokenizer/config hashes,
and the 48 checkpoint shards' size/mtime metadata match the completed original
campaign. Every saved prompt was checked against the actual local tokenizer,
with canonical single-token candidates. No clean-correctness filtering.

Totals: **768 trials, 384 identities, 192 target/mode baselines**, plus
**32 layer-42 diagnostics per runtime**. Pilot contributes 32/16/8 to those
totals. A fresh runtime requires 1,376 forwards; the fourth condition adds
296 relative to the three-condition plan (27.4%, without another model load).
Elapsed-time impact remains an estimate. Diagnostics now derive distinct
target/donor/position-set/mode combinations from the first active panel.
Counts, progress, completion records and reports are manifest-derived.

The [frozen preflight](runs/deepseek-v4-flash/one-fact-patching-filler-coverage/preflight.json)
passed **51 CPU tests in 29.49 seconds** using `.venv-sglang/bin/python`,
one CPU math thread and disabled pytest plugin autoload. Expanded checks cover
20-position/four-rank replacement, exact layers 32–37, restoration and
continued propagation through 42, diagnostic membership, CLI roots, and
interrupted controls/pilot/discovery recovery with original baseline retention.
A nonzero synthetic-data check verifies 20,000 shared paired bootstrap
resamples (seed 42). The [compatibility check](runs/deepseek-v4-flash/one-fact-patching-filler-coverage/compatibility.json)
replayed saved token IDs to reproduce all original scientific manifest fields;
the default schema-2 design remains supported. Source/checkpoint guards pass.

Analysis now exports trial clean/patched/difference scores, panel means, eight
intervention/donor conditions and matched mode/donor contrasts, with Markdown,
CSV, PNG/PDF and JSON outputs. Expected rows: 2,304 trial/candidate scores,
576 panel/candidate means, 24 condition/candidate summaries and 24 matched
contrast/candidate summaries. Full-vocabulary-normalized explicit candidate
logprobs and aliases are retained. Bootstrap settings: 20,000 shared whole-panel
resamples, seed 42, descriptive 95% intervals without multiplicity adjustment.

Implementation changes are in `filler/dsv4/{patching,patching_campaign,patching_analysis}.py`,
`scripts/dsv4/one_fact_patching.py` and `tests/test_one_fact_patching.py`.
The instrumented launcher retains the HTTP `--port`/unset `SGLANG_PORT` fix.
The controller now records actual allocation account/QOS/node/start/end, and
rechecks frozen sources after inference. Git revisions and 28 SHA-256 input/source
hashes are recorded in the manifest; changes are uncommitted. Existing unrelated
changes and prior campaign outputs were preserved.

No Slurm job was submitted or model loaded. The next action requires explicit
submission approval under project AGENTS.md: one four-hour interactive node,
account m5258_g, four A100 80 GB GPUs, constraint `gpu&hbm80g`. The exact command
and pending native/identity/diagnostic gates (logprob tolerance 0.15) are in
PREPARED.md and the updated handoff. GPU results are pending. CPU commands exited
zero despite sandbox MUNGE socket messages; use the approved host scheduler
path after approval. No system configuration was changed.

### 2026-09-09 23:22 UTC: approved filler-coverage execution started

User approved the four-hour submission with `go`, then requested no further
approvals during execution. Frozen hashes/preflight were verified immediately
before submission. Job **58132427** started at **23:22:33 UTC** on **nid008265**,
account **m5258_g**, actual QOS **gpu_interactive**, four **A100-SXM4-80GB**
GPUs (85,093,777,408 bytes each), 128 logical CPUs. Deadline:
**2026-09-10 03:22:33 UTC**. Submitted the exact PREPARED.md salloc/srun command
through host Slurm/MUNGE access; no system configuration changes.

Runtime `58132427-9f78bd47b97d` is starting the instrumented server with the
frozen command and HTTP-port fix. Native equivalence, identity/diagnostic
gates, discovery, and final reporting remain pending. Durable
[launch record](runs/deepseek-v4-flash/one-fact-patching-filler-coverage/launch-attempts.json),
[allocation log](runs/deepseek-v4-flash/one-fact-patching-filler-coverage/allocation.log),
and [runtime metadata](runs/deepseek-v4-flash/one-fact-patching-filler-coverage/runtimes/58132427-9f78bd47b97d/runtime.json)
record the approved environment. Keep this allocation/server alive through
both experiment phases and integrity verification.

### 2026-09-09 23:42 UTC: filler-coverage pilot passed; discovery active

Job 58132427 loaded weights in **753.46 seconds**; HTTP startup succeeded at
**23:36:53 UTC**. All 43 layer hooks are installed on each TP rank. Native
final-layer validation passed with exact returned-hidden and rank equality,
equal argmax and maximum logprob error **0.125** (limit 0.15). All **16 pilot
identities, 32 layer-42 diagnostics and 32 pilot trials** passed their gates
and integrity check. The controller automatically continued discovery on the
same loaded server; no additional submission or approval was needed.

### 2026-09-10: one-fact simultaneous patching discovery completed

Campaign `5853c9f001996c60a4473e36be00f75659d66e54171b51805636d3dd86abf3ec`, design `filler-coverage`: all 768 substantive trials, 384 identity controls and 192 target/mode baseline cells completed. Pilot results were retained in discovery totals. Each runtime passed native final-layer equivalence; each panel passed identity controls before substantive trials. The first active panel per runtime also passed all 32 layer-42 filler-only controls. Every stored candidate delta was recomputed from its original raw response and matched runtime/mode baseline.

Manifest, journal and raw rank captures: `/pscratch/sd/m/marocco/sandbox/mech_int/runs/deepseek-v4-flash/one-fact-patching-filler-coverage`. [Results and plots](/pscratch/sd/m/marocco/sandbox/mech_int/runs/deepseek-v4-flash/one-fact-patching-filler-coverage/analysis/REPORT.md) include all three sums and paired panel bootstrap intervals (20,000 resamples, seed 42); no clean-correctness filtering. Interrupted runtimes retain their original baselines; unfinished work uses fresh captures. Discovery intervals are descriptive and are not adjusted for multiple comparisons.

### 2026-09-10: filler-coverage final audit and results

Job **58132427** completed at **00:00:48 UTC** after approximately **38 minutes
16 seconds** from allocation start, then relinquished the allocation normally
(controller exit 0). All **768 trials, 384 identities, 192 baselines and 32
layer-42 diagnostics** are complete on the original server/runtime. No additional
submission or approval was needed. Native hidden/rank and argmax equality passed;
maximum logprob error 0.125. All identity/diagnostic requested scores matched
exactly (maximum error 0). All 5,504 rank captures (~115.38 GiB) passed controller
integrity checks, and frozen source/checkpoint guards passed before and after
inference. The last trial finished at 23:55:09 UTC; final checks/reporting took
about 340 seconds. No live server remains.

The [independent audit](runs/deepseek-v4-flash/one-fact-patching-filler-coverage/analysis/INDEPENDENT_AUDIT.json)
passed at 00:01:47 UTC: 1,402 checksummed journal records, 1,344 raw responses,
1,152 candidate-alias equalities, all CSV contents, and all condition/paired
bootstrap statistics reproduced to 1e-12. It uses 20,000 shared whole-panel
resamples, seed 42, and reuses the controller's completed tensor integrity
check. Reproduce from the workspace with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1 .venv-sglang/bin/python runs/deepseek-v4-flash/one-fact-patching-filler-coverage/audit_results.py`.
The audit script's hash is in launch-attempts.json. PNG layout was inspected;
CSV/Markdown/PNG/PDF/JSON artifacts are complete and linked in the
[results and interpretation](runs/deepseek-v4-flash/one-fact-patching-filler-coverage/analysis/INTERPRETATION.md).

All-fillers/layers 0–42 donor-sum gains: **7.5044 nats** (same X) and **6.8536**
(different X) with full downstream, versus **5.9091/5.4582** with answer-only.
Matched full-minus-answer-only gains are **1.5953 [1.1639, 2.0516]** and
**1.3954 [1.1032, 1.7377]** nats (95% paired intervals). Because fillers are
clamped at every layer in both modes, this comparison tests propagation through
the two intervening non-filler rows before the answer-prediction row.

For filler_5/layers 32–37, donor-sum gains are **2.7887/2.3899** nats with full
downstream and **1.9285/1.8596** with answer-only. Matched mode gains are
**0.8602 [0.4953, 1.3026]** and **0.5303 [0.3342, 0.7457]**. All-fillers/full
reduces target-sum scores by **0.7911/0.8567** nats (both intervals below zero).
Other target-sum intervals include zero, which does not establish exactly zero
effects. Same-X donor/mixed aliases agree exactly. Mean patched target scores
remain above mean donor scores in all eight conditions.

All 24 panels remain included. Intervals are descriptive without multiplicity
adjustment. Position set and layer range change together between families, so
the family comparison cannot isolate a position effect. Whole-residual patches
can transfer operands as well as answer-related features and do not establish
a precomputed sum representation. The handoff and campaign status records are
updated. Existing unrelated changes were retained; no commit was made.

### 2026-09-10 01:57 UTC: filler_5 layer/recomputation comparison with errors

At the user's request, combined the saved filler_5 interventions for inclusive
zero-based layers **0–42, 32–37, and 33–42**, each under full-downstream and
answer-only recomputation, retaining both donor types and all three candidate
sums. The first/third ranges come from original runtime 58129149-6a8c8f39011a;
32–37 comes from follow-up runtime 58132427-9f78bd47b97d. Saved panels, prompts
and positions agree exactly across runs; each score uses its original matched
runtime/mode baseline. All 24 panels/96 targets per condition are retained.

The [comparison table](runs/deepseek-v4-flash/one-fact-patching-filler-coverage/analysis/FILLER5_LAYER_COMPARISON.md)
reports mean patched-minus-clean log probabilities in nats with **±1σ
bootstrap standard errors**, calculated as the sample standard deviation
(ddof=1) of 20,000 whole-panel bootstrap means using shared indices and seed 42.
Errors describe panel-sampling uncertainty and do not independently estimate
between-runtime variability. [CSV](runs/deepseek-v4-flash/one-fact-patching-filler-coverage/analysis/filler5_layer_comparison.csv)
also retains clean/patched means and descriptive 95% intervals;
[JSON/provenance](runs/deepseek-v4-flash/one-fact-patching-filler-coverage/analysis/filler5_layer_comparison.json)
and the [reproduction script](runs/deepseek-v4-flash/one-fact-patching-filler-coverage/analysis/filler5_layer_comparison.py)
are saved alongside it. Execute with `.venv-sglang/bin/python` and one CPU math
thread from the workspace. This brief NumPy analysis required no inference
or scheduler action and left original campaign artifacts unchanged.

Validation passed: 3,956 journal checksums/configuration IDs, all 1,152 selected
trials and their baseline matches, four targets per panel/condition, and exact
same-addend candidate aliases. Reproduced 864 prior panel means, 36 condition
means/95% intervals, and 24 prior one-sigma estimates to 1e-12. Uses saved scores
and prior completed raw/tensor integrity checks. No commit was made.

### 2026-09-10: filler_10 missing-range campaign implemented and CPU-preflighted

Objective: match the filler_5 analyses for filler_10 by adding only inclusive layers
32–37, reusing original filler_10 layers 0–42 and 33–42 and the existing filler_5
reference. No new inference has run; explicit allocation approval is pending.

Design `filler10-coverage`, root `runs/deepseek-v4-flash/one-fact-patching-filler10-coverage`, configuration
`062d2db00d1e38420d8c8ec58a0abb1538a25481a436cade81e51bb53bacaa18`. Fresh runtime: 384 trials, 192 identities, 192 baseline cells,
16 layer-42 diagnostics; automatic first-panel continuation. Original saved prompts,
24 panels/96 targets, both donors and native whole-mHC replacement/restoration
semantics retained. HTTP port explicit; SGLANG_PORT removed; 0.15 gates unchanged.

Added cross-campaign `compare --root` and automatic completion-time reporting.
Historical journals are read without repair or progress writes. Scores retain
original runtime/mode baselines. All candidate scores, three logit gaps and 60
matched mode/layer/donor/site contrasts use within-panel differences and 20,000
shared bootstrap draws, seed 42, ddof=1 standard errors and descriptive 95%
intervals; errors do not separately estimate between-runtime variability.

Final CPU preflight: 62 tests passed in 34.17 seconds. Verified 3,956 historical
journal records, 3,840 raw responses, 1,152 exact same-addend aliases, saved
panel/prompt/tokenizer/checkpoint agreement and 4,264 old scalar statistics
(maximum error 8.881784197001252e-16, tolerance 1e-12), including old paired
contrasts. A new test initially used the wrong raw-response fixture key; corrected
before the successful preflight. Actual GPU equivalence and rank-capture gates
remain to run. Historical full tensor checks are reused from completed campaigns.

[Prepared design, implementation, validation and exact allocation command](runs/deepseek-v4-flash/one-fact-patching-filler10-coverage/PREPARED.md).
[Preflight](runs/deepseek-v4-flash/one-fact-patching-filler10-coverage/preflight.json), [compatibility](runs/deepseek-v4-flash/one-fact-patching-filler10-coverage/compatibility.json).
New report paths will be `runs/deepseek-v4-flash/one-fact-patching-filler10-coverage/comparison/` and `runs/deepseek-v4-flash/one-fact-patching-filler10-coverage/analysis/`.
No Slurm submission or commit was made; pre-existing unrelated work is retained.

### 2026-09-10 15:37 UTC: approved filler_10 coverage launched

The user approved the prepared campaign with `go`. Combined salloc/srun job
58163841 started at 15:37:23 UTC on nid008513, account m5258_g, actual QOS
gpu_interactive, deadline 19:37:23 UTC, 128 CPUs and four A100-SXM4-80GB GPUs
(85,093,777,408 bytes each). The prepared controller started immediately and
created runtime `58163841-a8e672cde1a2`. Instrumentation is supplied at launch,
with explicit HTTP --port and SGLANG_PORT removed. Exec session 26929 owns the
allocation. New native/identity/diagnostic gates and campaign completion are
pending; monitor the existing session and durable runtime/progress records.

Root: `runs/deepseek-v4-flash/one-fact-patching-filler10-coverage/`.

### 2026-09-10: one-fact simultaneous patching discovery completed

Campaign `062d2db00d1e38420d8c8ec58a0abb1538a25481a436cade81e51bb53bacaa18`, design `filler10-coverage`: all 384 substantive trials, 192 identity controls and 192 target/mode baseline cells completed. Pilot results were retained in discovery totals. Each runtime passed native final-layer equivalence; each panel passed identity controls before substantive trials. The first active panel per runtime also passed all 16 layer-42 filler-only controls. Every stored candidate delta was recomputed from its original raw response and matched runtime/mode baseline.

Manifest, journal and raw rank captures: `/pscratch/sd/m/marocco/sandbox/mech_int/runs/deepseek-v4-flash/one-fact-patching-filler10-coverage`. [Results and plots](/pscratch/sd/m/marocco/sandbox/mech_int/runs/deepseek-v4-flash/one-fact-patching-filler10-coverage/analysis/REPORT.md) include all three sums and paired panel bootstrap intervals (20,000 resamples, seed 42); no clean-correctness filtering. Interrupted runtimes retain their original baselines; unfinished work uses fresh captures. Discovery intervals are descriptive and are not adjusted for multiple comparisons.

[Three-range filler_10/filler_5 comparison](/pscratch/sd/m/marocco/sandbox/mech_int/runs/deepseek-v4-flash/one-fact-patching-filler10-coverage/comparison/REPORT.md) includes clean/patched candidate scores, target-relative and donor-minus-mixed logit gaps, and matched mode/layer/donor/site contrasts. Uses 20,000 shared panel resamples, seed 42, ±1σ standard errors (ddof=1) and descriptive 95% intervals. Historical tables reproduce to 1e-12. Errors do not separately estimate between-runtime variability.

### 2026-09-10 16:08 UTC: filler_10 three-range analysis completed and audited

Job **58163841** completed on **nid008513**, runtime `58163841-a8e672cde1a2`,
account **m5258_g**, actual QOS **gpu_interactive**, four A100-SXM4-80GB GPUs
(85,093,777,408 bytes each), 128 CPUs. Allocation start: **15:37:23 UTC**;
completion marker: **16:07:49 UTC**, before the 19:37:23 deadline. The controller
exited zero and salloc relinquished the allocation normally; no live server or
allocation remains from this run. All work used one uninterrupted runtime.

Weight loading took 754.30 seconds on TP0 (754.29–754.51 across ranks); HTTP ready
at 15:51:30 UTC. SGLang 0.5.18, the frozen checkpoint/configuration and all launch
instrumentation were verified. Native hidden states and TP ranks agreed exactly,
argmax matched, maximum candidate/top log-probability error 0.125 versus 0.15 limit.
All 192 identity controls and 16 layer-42 diagnostics matched baseline scores
exactly (maximum error 0). The pilot continued automatically into discovery.

Completed **384 substantive trials, 192 identities, 192 baseline cells, 16
diagnostics, all 24 panels/96 targets**, without clean-correctness filtering.
The controller validated raw responses, original runtime/mode baselines, complete
trial membership, hook acknowledgements and **3,136 rank captures** totaling
**53,790,974,080 bytes** (50.10 GiB). The journal has 810 checksummed records.
[Execution summary](runs/deepseek-v4-flash/one-fact-patching-filler10-coverage/EXECUTION_SUMMARY.json) and
[completion](runs/deepseek-v4-flash/one-fact-patching-filler10-coverage/COMPLETE.json) retain exact counts and allocation metadata.

The comparison joins 2,304 selected trials into 24 conditions and 60 matched
condition pairs. All 4,264 overlapping historical statistics reproduced within
1e-12 (maximum error 8.881784197001252e-16). An independent CSV/provenance audit
recomputed **16,416 scalar statistics exactly**, including all paired contrasts,
using the same 20,000 panel draws, seed 42, ddof=1; 1,152 same-addend aliases were
exact. PNG/PDF files passed format checks and the gap figure was visually checked.
Historical journals, baselines and results were reused without writes.

At the new filler_10/layers 32–37 condition, donor-sum changes (nats, ±1σ) are:

| Donor | Full downstream | Answer only |
|---|---:|---:|
| Different fact / same addend | +0.4445 ± 0.1693 | +0.1306 ± 0.0577 |
| Different fact / different addend | +0.1542 ± 0.0751 | +0.0601 ± 0.0576 |

The matched full-minus-answer changes are +0.3139 nats (95% interval
0.0914–0.6435) for same-addend donors and +0.0941 (0.000189–0.2088) for
different-addend donors. The latter lower bound is very close to zero. All four
32–37 versus 33–42 donor-sum contrasts have descriptive intervals containing
zero; this does not establish equivalence. All new target-sum intervals include
zero. Donor-minus-mixed shifts with different-addend donors are −0.1947 ± 0.1196
(full; 95% −0.4564–0.0072) and −0.0319 ± 0.0471 (answer only; −0.1204–0.0638).

Across all three ranges, filler_5 donor-sum increases exceed filler_10 in all
12 matched conditions, each with a positive descriptive 95% interval. At 32–37,
the site differences are 1.7979–2.3443 nats. These whole-residual interventions
do not distinguish stored sums from operand/features used downstream. Errors
describe panel sampling and do not separately estimate between-runtime variability;
intervals are descriptive and have no multiplicity adjustment.

[Full report](runs/deepseek-v4-flash/one-fact-patching-filler10-coverage/comparison/REPORT.md),
[filler_10 table](runs/deepseek-v4-flash/one-fact-patching-filler10-coverage/comparison/FILLER_10_THREE_RANGES.md),
[gap tables](runs/deepseek-v4-flash/one-fact-patching-filler10-coverage/comparison/LOGIT_GAPS.md),
[matched contrasts](runs/deepseek-v4-flash/one-fact-patching-filler10-coverage/comparison/MATCHED_CONTRASTS.md),
[independent audit](runs/deepseek-v4-flash/one-fact-patching-filler10-coverage/comparison/INDEPENDENT_AUDIT.json).
Full-precision CSVs, PNG/PDF plots, provenance JSON and reproducible commands
are linked from the report. Reproduce the lightweight comparison with
`.venv-sglang/bin/python -m scripts.dsv4.one_fact_patching compare --root runs/deepseek-v4-flash/one-fact-patching-filler10-coverage`
using one CPU math thread. Independently audit via `runs/deepseek-v4-flash/one-fact-patching-filler10-coverage/independent_audit.py`.
No further model launch is needed. No commit was authorized or made.

### 2026-09-11T17:35:48.893916+00:00: filler_10 full confirmation rerun prepared

User requested repeating the three-range filler_10 transplant on the full
confirmation dataset. The new `filler10-confirmation` design retains all 60
panels/240 targets/120 discovery-disjoint facts and scores A_t+X_t, A_d+X_d and
A_d+X_t at inclusive layers 0–42, 32–37 and 33–42, both propagation modes and
both donor types. Planned: 2,880 trials, 1,440 identities, 480 baselines and 16
layer-42 diagnostics. Exact dataset membership and tokenization passed; all
68 CPU preflight tests passed in 48.57 seconds. Scoped TeX-independent plotting
fixed a missing local cmr10.tfm failure, including the compatibility report.

Configuration `8a2887a0778c5a63a27d185efc7dcebb243dbbea0320cef10115a7fdf5f55457`. Native final-layer, identity and layer-42
GPU gates remain pending. No inference or allocation submission has occurred;
explicit approval of one four-hour interactive allocation with four 80 GB A100s
is the remaining launch step. The prepared command immediately runs the automated
pilot, full campaign, validation and reporting, then releases the allocation.

[Preparation and exact launch](runs/deepseek-v4-flash/one-fact-patching-filler10-confirmation/PREPARED.md),
[preflight](runs/deepseek-v4-flash/one-fact-patching-filler10-confirmation/preflight.json),
[review patch](runs/deepseek-v4-flash/one-fact-patching-filler10-confirmation/PREPARATION_CHANGES.patch).
New results will be written only to the confirmation campaign root; discovery
artifacts are preserved. No commit was made.

### 2026-09-11T17:38:11.068102+00:00: approved filler_10 confirmation launch

User approved with “Go”. Submitted job **58202743**, exec session **64738**, account m5258_g, interactive QOS, four GPUs with gpu&hbm80g, four-hour limit. Allocation is queued; the prepared controller starts automatically when granted and runs validation, the pilot, full confirmation and reporting before releasing the node. Campaign root: `runs/deepseek-v4-flash/one-fact-patching-filler10-confirmation`.

### 2026-09-11: one-fact simultaneous patching confirmation completed

Campaign `8a2887a0778c5a63a27d185efc7dcebb243dbbea0320cef10115a7fdf5f55457`, design `filler10-confirmation`: all 2,880 substantive trials, 1,440 identity controls and 480 target/mode baseline cells completed. Pilot results were retained in full-split totals. Each runtime passed native final-layer equivalence; each panel passed identity controls before substantive trials. The first active panel per runtime also passed all 16 layer-42 filler-only controls. Every stored candidate delta was recomputed from its original raw response and matched runtime/mode baseline.

Manifest, journal and raw rank captures: `/pscratch/sd/m/marocco/sandbox/mech_int/runs/deepseek-v4-flash/one-fact-patching-filler10-confirmation`. [Results and plots](/pscratch/sd/m/marocco/sandbox/mech_int/runs/deepseek-v4-flash/one-fact-patching-filler10-confirmation/analysis/REPORT.md) include all three sums and paired panel bootstrap intervals (20,000 resamples, seed 42); no clean-correctness filtering. Interrupted runtimes retain their original baselines; unfinished work uses fresh captures. Reported intervals are descriptive and are not adjusted for multiple comparisons.

### 2026-09-11T21:36:11.583608+00:00: filler_10 full confirmation completed and audited

Job **58202743**, runtime **58202743-975fbbd1ffd2**, finished successfully on
nid008340 (m5258_g, gpu_interactive, four verified 80 GB A100s). Start 17:39:44 UTC;
completion 2026-09-11T18:55:20.775030+00:00; scheduler elapsed 01:15:40, COMPLETED, exit
0:0. Exec session 64738 exited zero and salloc released the allocation. Weight
loading took 756.42–756.57 seconds; HTTP was ready at 17:53:52 UTC.

All **60 panels/240 targets**, **2,880 trials**, **1,440 identities**, **480
baselines** and **16 diagnostics** completed on one uninterrupted runtime. Native
hidden/rank/argmax equality was exact; maximum log-probability error 0.124974
(limit 0.15). Identities and diagnostics had zero error. Raw response/capture
integrity passed. An independent exported-data audit checked 8,640 score rows and
504 statistics with maximum discrepancy 8.33e-17 (tolerance 1e-12).

Same-addend donor/mixed changes were positive in all six descriptive 95% intervals.
At layers 32–37: donor changes +0.4810 ± 0.1215 (full), +0.1754 ± 0.0396
(answer only) for same-addend donors; +0.2448 ± 0.0656 and +0.1137 ± 0.0352
for different-addend donors. Different-addend mixed changes were +0.4828 ±
0.1238 and +0.1936 ± 0.0421. Full-downstream target changes at 32–37 and 33–42
were −0.056 to −0.064 nats with descriptive intervals below zero for both donors;
other target-change intervals include zero. Different-addend mixed mean increases
exceed donor mean increases in all six conditions; no paired significance claim
is made from this ordering. No discovery observations were pooled into confirmation.

[All three sums, all conditions](runs/deepseek-v4-flash/one-fact-patching-filler10-confirmation/analysis/FILLER_10_CONFIRMATION.md),
[full report](runs/deepseek-v4-flash/one-fact-patching-filler10-confirmation/analysis/REPORT.md),
[execution summary](runs/deepseek-v4-flash/one-fact-patching-filler10-confirmation/EXECUTION_SUMMARY.json),
[export audit](runs/deepseek-v4-flash/one-fact-patching-filler10-confirmation/EXPORT_AUDIT.json).
Errors use 20,000 shared panel resamples, seed 42, ddof=1; intervals are descriptive,
without multiplicity adjustment or a separate between-runtime variance estimate.
No model launch remains necessary. No commit authorized or made.
