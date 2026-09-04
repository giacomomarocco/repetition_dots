# Two-fact addition analysis

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
