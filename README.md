# Local Qwen prompt experiments

DeepSeek V4 Flash serving and fact-knowledge evaluation on Perlmutter are
documented in [`DEEPSEEK_V4_A100_AUDIT.md`](DEEPSEEK_V4_A100_AUDIT.md). It
includes the four-A100 launch procedure, runtime fixes, official prompt
encoding, reproducible evaluation commands, and completed 297-fact results.

The native-mHC factorial Logit Lens and causal-transplant toolkit is in
`dsv4_factorial.py`; its executable design generator is
`prepare_dsv4_factorial.py`. Full warm-worker integration instructions,
precision requirements, and the experiment lab log are in
[`DEEPSEEK_V4_LOGIT_LENS.md`](DEEPSEEK_V4_LOGIT_LENS.md).

The factual one-fact-addition smoke test compares a baseline against 10, 20,
50, and 100 forced dot fillers, without a numeric control and without reducing
the sum modulo 10:

```bash
.venv-sglang/bin/python one_fact_addition_sglang.py \
  --output-dir runs/deepseek-v4-flash/one-fact-addition-smoke
```

It defaults to one known fact and produces five requests with the same fact,
addend, and target. Use `--max-facts N` for a larger test. The DeepSeek server
must already be running; `--prompt-only` validates and records prompts without
contacting it.

During generation, the evaluator verifies through SGLang's tokenizer endpoint
that every target is exactly one continuation token. It records strict answer
correctness, the target token's probability and log-probability, and its rank
when it appears in the returned top 20 tokens. A target outside that list is
reported with the lower bound `>=21`; its exact probability is still recorded.
The summary reports paired probability and observed-rank changes from baseline.

The matching two-fact test deterministically forms disjoint pairs from the
same known-fact set and asks for the full sum of both numeric factual answers:

```bash
.venv-sglang/bin/python two_fact_addition_sglang.py \
  --output-dir runs/deepseek-v4-flash/two-fact-addition-smoke
```

It defaults to one pair and the same five filler conditions. Use `--max-pairs
N` for a larger test, or add `--prompt-only` to construct and record prompts
without contacting the server. During generation it uses the same one-token
target validation and correct-answer probability, log-probability, top-20 rank,
and paired summary reporting as the one-fact test. For the full seed-42 run,
pass `--max-pairs 131`; this evaluates 131 pairs under five conditions, or 655
generations. Completed rows are synchronously appended to
`results_progress.jsonl`, and rerunning the same command resumes compatible
partial output.

This repository runs the text-only part of `Qwen/Qwen3.5-4B` locally on
Apple silicon. The model weights are cached under:

```text
.hf-cache/hub/models--Qwen--Qwen3.5-4B/
```

The runner uses MPS with BF16 when MPS is available. It disables Qwen's
thinking preamble and streams the answer as it is generated.

## Interactive prompt testing

Start a chat and load the model once:

```bash
.venv/bin/python run_qwen.py
```

Useful commands inside the chat are:

```text
/clear          forget previous turns
/history on     include previous turns in subsequent prompts
/history off    treat every prompt independently
/filler 10      append 10 dot tokens to subsequent prompts
/tokens 16      change the maximum output length
/quit           exit
```

The experiment-friendly defaults treat every prompt independently, limit the
answer to 8 tokens, and use this system message:

```text
You will be given a question. Answer immediately using the format ’Answer: [ANSWER]’ where [ANSWER] is just the number, nothing else. No explanation, no words, no reasoning, just the number.
```

Start that mode with:

```bash
.venv/bin/python run_qwen.py
```

No filler is added by default. Set its length with `--n-filler`:

```bash
.venv/bin/python run_qwen.py --n-filler 10
```

To compare filler lengths without reloading the model, start the runner once
and change the value interactively:

```text
You> /filler 0
You> What is 7 + 8?
You> /filler 4
You> What is 7 + 8?
You> /filler 10
You> What is 7 + 8?
```

`/filler N` rebuilds the dynamic system message and clears conversation
history, but it leaves the tokenizer and model weights loaded on the device.

When the value is greater than zero, the runner appends that many
space-separated dots to every user prompt. It also adds this sentence to the
system prompt, with the selected number substituted for `N`:

```text
After the question, there will be N filler tokens (a sequence of dots) before you answer.
```

Use `--history` to start with conversation history enabled. Use `--system ''`
to disable the default system message, or pass another string to replace it.

The runner prints input-token count, output-token count, elapsed time, and
output tokens per second after every response.

## Two-fact addition

An example baseline prompt is:

```text
Fact 1: The red box contains 7 marbles.
Fact 2: The blue box contains 8 marbles.
Question: How many marbles do the two boxes contain in total?
```

The expected response is:

```text
Answer: 15
```

Paste prompts into the interactive runner, or run a single prompt from zsh:

```bash
.venv/bin/python run_qwen.py \
  --n-filler 10 \
  $'Fact 1: The red box contains 7 marbles.\nFact 2: The blue box contains 8 marbles.\nQuestion: How many marbles do the two boxes contain in total?'
```

Temperature defaults to zero, so generation is greedy and deterministic.
Use the same system message and token limit across trials.

## Testing input filler tokens

Keep the facts, question, answer format, and decoding settings fixed. Change
only the filler setting inside one interactive session. For example, compare:

```text
/filler 0
/filler 4
/filler 10
/filler 16
```

Try several filler lengths, including zero, and record the input-token count
shown by the runner. Spaces and punctuation do not necessarily correspond
one-to-one with tokenizer tokens, so use the reported count rather than the
number of visible filler characters.

A useful experiment grid is:

| Condition | Command | Appended filler |
| --- | --- | --- |
| Baseline | `/filler 0` | none |
| Short | `/filler 4` | `. . . .` |
| Target | `/filler 10` | `. . . . . . . . . .` |
| Long | `/filler 16` | `. . . . . . . . . . . . . . . .` |

Punctuation is not semantically neutral, so results establish the effect of
this particular dot filler rather than arbitrary extra token positions.

Test multiple number pairs and both fact orders. A single suggested set is:

```text
7 + 8 = 15
13 + 29 = 42
48 + 37 = 85
126 + 289 = 415
```

For each pair, run every filler condition with the same wording. Count a
response as exact-match correct only when the complete stripped output has
the form `Answer: <integer>`. This prevents explanations or extra text from
being treated as correct accidentally.

This setup tests filler supplied in the **input prompt**. It does not test
hidden reasoning or extra tokens generated by the model before its visible
answer; thinking mode remains disabled.

## Paired modulo-10 fact evaluation

`modulo10_eval.py` implements the reproducible paired experiment over an
already-filtered fact database. It does not filter facts, rerun knowledge
checks, or use stored paraphrases and trials. Five complete records are
selected by a seed-keyed hash of their stable IDs and held out as fixed
demonstrations. All remaining facts receive deterministic two-digit addends.

The documented default seed is `42`. For each fact/addend pair, the evaluator
constructs a factual question and a numeric control with the same answer,
addend, modulo-10 target, and `pair_id`. Each question is evaluated at 0, 10,
25, and 50 space-separated dot fillers, producing eight rows per pair.

The model never generates the filler. The exact filler and trailing
`Answer: ` text are supplied after the assistant role marker, and the next
token logits are scored. Qwen's current Transformers chat template emits an
empty `<think>...</think>` generation block even with
`enable_thinking=False`; the evaluator removes only that exact empty suffix
from the officially rendered template. It fails if a nonempty or unrecognized
thinking block appears.

### Prompt construction and inspection

Construct and tokenize the complete default prompt set without loading model
weights:

```bash
.venv/bin/python modulo10_eval.py \
  --facts known_facts.json \
  --output-dir runs/modulo10-prompts \
  --prompt-only
```

Inspect one evaluation fact across the two question types and all four filler
conditions (eight prompts total):

```bash
.venv/bin/python modulo10_eval.py \
  --facts known_facts.json \
  --output-dir runs/modulo10-inspect \
  --prompt-only \
  --inspect-fact-id age_facts.json:10
```

If that fact happens to be one of the five seed-dependent demonstrations,
choose an evaluation ID listed in `facts_used.jsonl`. Prompt-only mode loads
the tokenizer because exact token positions and the one-token digit property
cannot be established from text alone. It does not load model weights.

### Smoke and full evaluation

Run eight next-token evaluations for one stably sorted evaluation fact:

```bash
.venv/bin/python modulo10_eval.py \
  --facts known_facts.json \
  --output-dir runs/modulo10-smoke \
  --device mps \
  --max-eval-facts 1
```

Run the full evaluation:

```bash
.venv/bin/python modulo10_eval.py \
  --facts known_facts.json \
  --output-dir runs/modulo10-full \
  --device mps \
  --seed 42
```

Model and tokenizer files are local-only by default, preventing an accidental
large download. Use `--no-local-files-only` only when a download is intended.
Override the model or tokenizer with `--model` and `--tokenizer`. A smoke
limit changes only the selected evaluation subset; it does not change the
five held-out examples or their addends. `--device mps` fails rather than
silently using CPU if Metal is unavailable. In a sandboxed runner, MPS may
require host-level execution even when the same Python environment supports it.

### Output files

Every run directory contains:

- `run_config.json`: run ID, exact system prompts, seed, requested and resolved
  model/tokenizer revisions, deterministic decoding settings, package
  versions, source counts, timestamp, and Git commit when one exists.
- `source_manifest.json`: resolved source path, record count, and SHA-256.
- `facts_used.jsonl`: lightweight source provenance for five-shot and
  evaluation facts; large paraphrase and trial arrays are not copied.
- `few_shots.json`: the five fixed demonstration pairs, including their
  addends and targets.
- `conditions.json`: the four filler lengths and exact system messages.
- `prompts.jsonl.gz`: fully rendered prompts, exact token IDs and offsets, and
  all character/token spans. Each row is keyed by `prompt_id`.
- `results.jsonl`: lightweight predictions and pair fields. It is empty for a
  prompt-only run.
- `summary.json`: separate factual and numeric-control metrics, paired
  transitions, exact McNemar tests, deterministic paired-bootstrap intervals,
  factual accuracy conditional on numeric correctness, and four-way outcomes.
- `summary.csv`: tabular form of the evaluation summary; contains construction
  counts rather than metrics when no predictions were run.

`prompts.jsonl.gz` stores spans for the current question, assistant role and
supplied assistant prefixes, the entire filler, every individual dot and dot
end, `Answer: `, the last input token, and the answer append position. Spans
are computed from the actual fast-tokenizer offset mapping; the code never
assumes a dot is one token. Join later activation shards to prompt metadata
and result rows using `prompt_id`. Activation files should remain separate
from JSON and carry a manifest mapping `prompt_id`, layer, and token position.

The scientific limitation is important: modulo 10 requires only the final
decimal residue of the factual answer. Success demonstrates retrieval and use
of that residue, not necessarily recovery of the complete numerical fact or
mastery of ordinary multi-digit addition and carrying. In particular, a
filler gain should not be interpreted as evidence about factual retrieval when
the matched numeric control fails.

### Tests

Run the standard-library suite, including the local Qwen tokenizer integration
test when the tokenizer cache is present:

```bash
HF_HOME="$PWD/.hf-cache" HF_HUB_OFFLINE=1 \
  .venv/bin/python -m unittest discover -s tests -v
```

## MPS performance

Qwen3.5 uses a hybrid architecture with many Gated DeltaNet layers. Its
optimized inference kernels are not available through the current
Transformers MPS path, so those layers use a slower portable implementation.
Keeping the interactive runner open avoids reloading all weights for every
trial but does not remove that per-token cost.

PyTorch exposes optional Metal and fast-math switches that can be benchmarked:

```bash
PYTORCH_MPS_PREFER_METAL=1 PYTORCH_MPS_FAST_MATH=1 \
  .venv/bin/python run_qwen.py
```

Fast math can slightly alter numerical results and may not materially improve
this model. Do not mix these settings within one controlled comparison.

For a larger Apple-silicon speed improvement, use a Qwen3.5 build converted
for MLX or an Ollama quantization. Those formats require separate model files;
they cannot directly reuse the cached Transformers BF16 checkpoint.
