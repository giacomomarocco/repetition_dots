# Two-fact addition analysis

## 2026-09-03 — Upstream-format comparison preflight

Objective: isolate the behavioral effect of adopting the prompt scaffold from
`kaleybrauer/filler-token-reasoning` while preserving the 200 atomic target
pairs used in the local-format run. Audited upstream revision
`4ba4c75d5d9f04248749ec46b8bed8661b746715`, specifically
`scripts/data/generate_2fact_dataset.py` and
`scripts/eval/evaluate_2fact_vllm.py`.

Added `--prompt-variant upstream` as an opt-in alternative; `local` remains the
default, so existing behavior and outputs are not overwritten. The upstream
variant reproduces its condition-specific system message, five atomic-number
demonstration pairs, phrasal `Question:` wording, blank-line separators, and
literal `Filler: ` label before space-separated dots. Baseline user turns omit
the filler line entirely. The run configuration records the upstream repository,
revision, and source path.

Prepared a separate prompt-only manifest with:

```bash
.venv-sglang/bin/python -m scripts.addition.two_fact \
  --facts runs/deepseek-v4-flash/two-fact-addition-full-under-999/eligible_facts.json \
  --fact-kind atomic --max-pairs 200 --filler-lengths 0 10 20 50 100 \
  --prompt-variant upstream \
  --output-dir runs/deepseek-v4-flash/two-fact-addition-5shot-atomic-200-upstream-format \
  --prompt-only
```

The preflight produced exactly 1,000 prompts (200 pairs × five conditions), and
the focused one- and two-fact suites passed all 22 tests. This is a full-format
comparison, not a `Filler:`-label-only ablation. It intentionally keeps the
local run's target pairs and filler lengths (including 20 rather than upstream's
usual 25) so the two result sets remain paired. No model load, inference, or
Slurm action was performed for this preflight.

## 2026-09-03 — 200-pair atomic-only five-shot run

Objective: expand the both-atomic analysis to 200 paired observations at each
filler length. The eligible-fact manifest contains 99 facts with `kind: atomic`,
so `filler/addition/two_fact.py` was extended with exact `--fact-kind` filtering
and deterministic cyclic-offset rounds. These produce additional distinct
directed pairs after the original successor round, without self-pairs or
duplicate directed pairs.

The prompt-only preflight and generation command used seed 42, greedy decoding,
and filler lengths 0, 10, 20, 50, and 100:

```bash
.venv-sglang/bin/python -m scripts.addition.two_fact \
  --facts runs/deepseek-v4-flash/two-fact-addition-full-under-999/eligible_facts.json \
  --fact-kind atomic --max-pairs 200 --filler-lengths 0 10 20 50 100 \
  --output-dir runs/deepseek-v4-flash/two-fact-addition-5shot-atomic-200
```

Allocation `57909465` used four A100 80-GB GPUs on `nid008209` with account
`m5258_g` and interactive QOS. The existing DeepSeek V4 Flash A100 launcher used
the `throughput` startup profile. Startup took about 17 minutes, including the
MXFP4-to-INT8 conversion and CUDA-graph capture; the endpoint smoke test passed.
The 1,000 generations then completed in about three minutes with per-prompt
checkpointing.

| Dots | Correct | Accuracy | Mean target log probability | Change from baseline |
|---:|---:|---:|---:|---:|
| 0 | 28/200 | 14.0% | -6.104 | — |
| 10 | 26/200 | 13.0% | -6.575 | -0.471 |
| 20 | 25/200 | 12.5% | -5.955 | +0.149 |
| 50 | 30/200 | 15.0% | -6.371 | -0.267 |
| 100 | 18/200 | 9.0% | -7.224 | -1.120 |

Thus 50 dots had the highest exact accuracy, one percentage point above
baseline, while 20 dots had the best mean target log probability. The 100-dot
condition was worst on both measures. These are descriptive paired results;
formal uncertainty and paired significance tests have not yet been added.

Integrity checks: 1,000 unique prompt IDs cover 200 unique directed pairs and
exactly 200 prompts per condition; all 99 selected source facts are atomic,
there are no self-pairs, no result references a non-atomic fact, and all target
log-probability fields are present. Outputs are in
`runs/deepseek-v4-flash/two-fact-addition-5shot-atomic-200/`. The focused one-
and two-fact test suites passed all 20 tests before launch.

## 2026-09-03 — Is the five-shot accuracy loss real?

Audited all 1,300 records in
`runs/deepseek-v4-flash/two-fact-addition-5shot-cyclic/results.json` to assess
whether the lower filler-condition accuracies establish a real degradation.
All 260 cyclic pairs are present exactly once in all five conditions, prompt
IDs are unique, targets equal the two source answers, recorded correctness
equals strict parsed-answer correctness, and target log probabilities are
finite. Each filler prompt has the expected six filler-plus-`Answer:` slots
(five demonstrations and one target). Conditions were evaluated consecutively
within each pair, limiting slow time drift, although condition order was fixed.

The exact-answer changes from baseline were -2.7, -1.2, -1.9, and -1.2
percentage points at 10, 20, 50, and 100 dots. Their matched gain/loss counts
were 8/15, 12/15, 8/13, and 11/14; exact McNemar p-values were 0.210, 0.701,
0.383, and 0.690. The corresponding pair-bootstrap 95% intervals were
-6.2 to +0.8, -5.0 to +2.7, -5.4 to +1.5, and -5.0 to +2.7 points. An omnibus
Cochran Q test across all five conditions also found no condition difference
(Q=2.804, 4 df, p=0.591). Averaging each pair's four filler outcomes gave a
-1.73-point change with a pair-bootstrap 95% interval of -5.0 to +1.35 points.
Circular moving-block bootstraps with block lengths 2 through 20, used as a
sensitivity check for adjacent cyclic pairs sharing facts, gave essentially
the same interval and always included zero.

The result is not caused by strict output parsing. Re-scoring a response as
correct when the target integer appeared at its start or anywhere in it
recovered no additional correct answer in any condition. However, answer-only
compliance varied: 227/260 baseline responses were numeric-only, versus
176/260, 164/260, 217/260, and 237/260 under 10, 20, 50, and 100 dots. This
non-monotonic variation is another reason not to interpret the small accuracy
ordering as a dose-dependent effect.

There is stronger evidence for degradation in the softer target-token metric.
Mean target log probability changed by -0.608, -0.294, -0.666, and -0.264
nats. The 10- and 50-dot paired tests had unadjusted p-values near 0.005; the
20- and 100-dot changes were not significant. Averaging the four filler
conditions within pair yielded -0.458 nats (ordinary pair-bootstrap 95% CI
-0.842 to -0.079; paired t p=0.019; Wilcoxon p=0.011). Circular block-bootstrap
intervals remained below zero for block lengths 2 through 20. A Friedman test
also detected an overall log-probability condition difference (p=0.0227), but
the effect was non-monotonic and 100 dots had essentially the same mean
within-pair rank as baseline.

Conclusion: the observed accuracy loss is real as a descriptive property of
this completed run, but the data do **not** establish a reproducible negative
effect on exact-answer accuracy. They are compatible with modest harm, no
effect, or a small benefit at individual lengths. The target-log-probability
analysis supports some filler-induced suppression, especially at 10 and 50
dots, without showing a monotonic length response. Independent reruns and/or
new pairing seeds are required to separate a reproducible prompt effect from
run-specific serving variation and this particular cyclic pairing.

## 2026-09-03 — Five-shot both-atomic subset

Restricted `two-fact-addition-5shot-cyclic/results.json` to pairs for which both
source facts have `kind: atomic` in the eligible-fact manifest. The cyclic
design contains 39 directed both-atomic pairs per condition. Exact-answer
accuracy was 6/39 (15.4%) at baseline, 3/39 (7.7%) at 10 dots, 6/39 (15.4%) at
20 dots, and 4/39 (10.3%) at both 50 and 100 dots. Mean target log probability
was -5.942, -6.682, -5.854, -7.719, and -7.383, respectively. Thus 20 dots
tied baseline accuracy and produced only a small +0.088 mean-log-probability
change; the other filler lengths were worse.

This is not a controlled zero-shot/five-shot comparison: the completed
zero-shot run placed filler in a forced assistant prefix after `</think>`, while
the five-shot run placed it between each question and `Answer:` in all
demonstrations and the target. After checking arXiv:2607.03502 and its released
repository, the latter placement is closer to the paper: its scaffold is
question, literal `Filler:` label plus filler, then `Answer:`, and its five
few-shot examples also contain filler. Therefore the zero-shot run, not the
five-shot run, has the location mismatch. The local five-shot run still omits
the paper's literal `Filler:` label and condition-specific system message, uses
a different model/checkpoint, and includes mixed fact types; it is not a close
replication of the paper's behavioral experiment.

## 2026-09-03 — Doubled five-shot rerun design

Changed `filler/addition/two_fact.py` from disjoint adjacent pairing to cyclic
successor pairing after the same seed-keyed ordering. On the existing 260-fact
`two-fact-addition-full-under-999/eligible_facts.json`, this produces 260 pairs:
each fact occurs once in each question position, no pair contains the same fact
twice, and five filler conditions produce 1,300 prompts. A prompt-only preflight
completed successfully; targets range from 13 through 654, with none above 999.
The focused one- and two-fact test suites pass (18 tests). Use a fresh output
directory because prior manifests encode the old disjoint-pairing protocol.
The one-fact evaluator was also given matching per-result durable checkpointing,
configuration-validated resume, and clean Ctrl-C partial-output handling.

## 2026-09-01 — Accuracy by fact-question type

Objective: determine whether DeepSeek V4 Flash is more accurate when both
questions ask for atomic numbers, and identify the filler condition where that
subset performs best.

Inputs:

- `runs/deepseek-v4-flash/two-fact-addition-full-under-999/results.json`
- `runs/deepseek-v4-flash/two-fact-addition-full-under-999/eligible_facts.json`
- Run configuration: seed 42, 130 disjoint pairs, filler lengths 0, 10, 20,
  50, and 100 dots, greedy decoding, exact parsed-answer scoring.

Method: joined result fact IDs of the form `source_file:source_index` to the
eligible-fact metadata, then partitioned the same 130 pairs into both atomic
number (18), exactly one atomic number (63), and neither atomic number (49).
Wilson 95% intervals were calculated for the binomial accuracy estimates.

| Pair type | Baseline | 10 dots | 20 dots | 50 dots | 100 dots |
|---|---:|---:|---:|---:|---:|
| Both atomic | 3/18 (16.7%) | 2/18 (11.1%) | **4/18 (22.2%)** | 3/18 (16.7%) | 2/18 (11.1%) |
| Exactly one atomic | 4/63 (6.3%) | 3/63 (4.8%) | 5/63 (7.9%) | 7/63 (11.1%) | **10/63 (15.9%)** |
| Neither atomic | 2/49 (4.1%) | 1/49 (2.0%) | 4/49 (8.2%) | **6/49 (12.2%)** | 2/49 (4.1%) |

Both-atomic Wilson 95% intervals were 5.8–39.2% at baseline, 3.1–32.8% at 10
dots, 9.0–45.2% at 20 dots, 5.8–39.2% at 50 dots, and 3.1–32.8% at 100 dots.

Observations:

- Both-atomic pairs performed best at 20 dots: 22.2%, one additional correct
  pair over baseline. Pairwise, 3 baseline failures became correct and 2
  baseline successes became incorrect.
- Both-atomic questions also had the highest baseline accuracy, but the sample
  is only 18 pairs and its confidence intervals are broad. The data therefore
  support a descriptive difference, not a strong claim that atomic-number
  pairing causes better performance.
- Long filler did not help both-atomic pairs: 100 dots fell to 11.1%, while
  exactly-one-atomic peaked at 100 dots and neither-atomic peaked at 50 dots.
- Target magnitude does not obviously explain the atomic split: mean targets
  were 116.3 for both-atomic pairs and 114.9 for all other pairs, although the
  non-atomic range was wider (21–654 versus 31–198).

Integrity checks: 650 results correspond to 130 unique pairs under five
conditions; each atomic-group denominator is constant across conditions and
the three groups sum to 130.

Limitation: there are too few examples in most of the detailed category-pair
cells to interpret them separately. A stratified or deliberately balanced
pairing design would be needed for a well-powered category comparison.
