# DeepSeek V4 Flash: four-A100 audit

Audit date: 2026-08-31

## Result

The model served successfully on four NVIDIA A100-SXM4-80GB GPUs. A local
`/generate` request returned HTTP 200 and completed the prompt "The capital of
France is" with text beginning " Paris. The capital of France is Paris".

The interactive allocation (`57791821`, node `nid008428`) arrived in about 90
seconds. It was released after testing.

## Local installation

- Original checkpoint: `model/DeepSeek-V4-Flash-0731` (about 156 GiB)
- Converted checkpoint: `model/DeepSeek-V4-Flash-0731-MoE-MXFP4-BF16` (about 162 GiB)
- A100 port: `ports/deepseek-v4-a100-sglang`, commit `d3987e718f0f0d835e97b1ff43a95af282494c85`
- Pinned SGLang: `ports/sglang`, commit `1c0019da7579db73223195f25b0eed3882dff24e`
- Environment: `.venv-sglang`
- Tested launcher: `run_deepseek_v4_a100.sh`

The converted checkpoint has 48 shards. Conversion preserved 70,656 expert
tensors and dequantized 390 tensors.

## Timing audit

| Stage | Observed time | Finding |
|---|---:|---|
| 80GB interactive queue | about 90 s | No long wait in this sample |
| Checkpoint conversion | about 9-10 min | About 300 MB/s output; about 1.5 CPU cores |
| Distributed/NCCL initialization | 0.7-1.8 s | Fast after configuration fixes |
| Read 48 shards from Scratch | about 8 s | Not a bottleneck |
| MXFP4-to-INT8 MoE preparation | 770 s | Primary repeat-start bottleneck |
| CUDA graph capture | 268 s | Secondary startup bottleneck |
| First small HTTP request | 13.56 s | Includes first-inference effects |

Server readiness took roughly 17m45s. Weight memory was about 38.84 GiB per
GPU. About 10.8 GiB per GPU remained after KV-pool allocation and graph capture.
The configured token capacity was 1,346,560.

`time -v` recorded 1,875 seconds system CPU, 44.6 million major page faults, and
322 GB filesystem input over the successful 19m34s process lifetime. Together
with the eight-second shard read, this points to large-scale in-memory/on-demand
weight transformation during the 770-second preparation step.

Tiny file operations during this audit also occasionally took 1-3 minutes on
Scratch, indicating intermittent metadata/process-launch latency worth measuring
separately next time.

## Failures identified and fixed

1. NERSC exported `HOST=login31`; the launcher used it for rendezvous and timed
   out. Force `HOST=127.0.0.1`.
2. External `libnccl-net.so` linked CUDA 12 while the environment used CUDA 13
   and NCCL 2.28.9. `NCCL_NET=Socket` and `NCCL_SOCKET_IFNAME=hsn` worked.
3. Custom-all-reduce JIT serialized behind a shared cache lock. The tested launch
   disables it and places TVM-FFI, Triton, and FlashInfer caches on node-local tmp.
4. The model has no Hugging Face chat template. Raw `/generate` works; chat use
   requires an explicit compatible template.

## Next optimization session

1. Investigate serializing and reloading the port's packed INT8 expert weights;
   this directly targets the 770-second cost.
2. Test `--cuda-graph-max-bs 8` or `16`, and separately
   `--disable-cuda-graph`, for interactive use. Compare startup and throughput.
3. Measure warm-request TTFT and decode tokens/s separately from the first call.
4. Record timestamped `nvidia-smi dmon`, CPU, page-fault, and Scratch I/O data.
5. Check for a NERSC-supported CUDA-13-compatible NCCL network plugin.

## Reproduction

Request an allocation only with explicit approval:

```bash
salloc --nodes=1 --qos=interactive --time=02:00:00 \
  --constraint='gpu&hbm80g' --gpus=4 --account=m5258_g --immediate=600
```

On the compute node:

```bash
./run_deepseek_v4_a100.sh
```

The optional `fast` startup profile adds
`--disable-cuda-graph` and should remove the approximately 268-second graph
capture observed above. Select a different profile when steady-state throughput
matters more than startup latency:

```bash
# Skip graph capture explicitly.
DEEPSEEK_STARTUP_PROFILE=fast ./run_deepseek_v4_a100.sh

# Capture graphs only through batch size 8.
DEEPSEEK_STARTUP_PROFILE=balanced ./run_deepseek_v4_a100.sh

# This is the default: use SGLang's auto-sized graph capture behavior.
DEEPSEEK_STARTUP_PROFILE=throughput ./run_deepseek_v4_a100.sh
```

Additional SGLang arguments can be appended to the launcher command. The
profiles do not address the approximately 770-second MXFP4-to-INT8 preparation,
which is a GPU-side per-layer weight remap rather than checkpoint read time.
Eliminating that phase requires a separately converted checkpoint containing
the port's final packed tensors.

Smoke test from the same node:

```bash
curl http://127.0.0.1:30002/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"The capital of France is","sampling_params":{"temperature":0,"max_new_tokens":8}}'
```

## Fact-knowledge evaluation: 2026-08-31

### Objective and protocol

The follow-up analysis measured which integer-valued facts DeepSeek V4 Flash
could answer under the same protocol used for the other local models.
`filler/fact_eval/sglang.py` sends raw prompts to SGLang's `/generate` endpoint.
It uses DeepSeek's bundled `encoding/encoding_dsv4.py` encoder in non-thinking
`chat` mode because the checkpoint has no Hugging Face chat template.

- Seed 42 selects 99 facts from each of `age_facts.json`,
  `atomic_facts.json`, and `static_facts.json` (297 facts total).
- Each fact is asked through five unique paraphrases without history, retrieval
  context, or few-shot examples.
- Generation is greedy (`temperature=0`) with at most eight new tokens.
- Only a complete response matching `Answer: <integer>` is accepted.
- At least four correct responses out of five classify a fact as known.
- Every response is appended to a progress file, allowing exact resumption.

After starting the server, the one-fact smoke test was:

```bash
.venv-sglang/bin/python -m scripts.fact_eval.sglang \
  --max-facts 1 \
  --output-dir runs/deepseek-v4-flash/fact-smoke-1
```

All five Atatürk age-at-death paraphrases returned exactly `Answer: 57`. The
full warm-server evaluation was then run with:

```bash
.venv-sglang/bin/python -m scripts.fact_eval.sglang \
  --output-dir runs/deepseek-v4-flash/fact-knowledge
```

### Results

The full evaluation completed 1,485 trials in 341.58 seconds.

| Source | Known | Unknown | Mean trial accuracy |
|---|---:|---:|---:|
| Age | 78 | 21 | 81.2% |
| Atomic | 99 | 0 | 99.8% |
| Static | 85 | 14 | 88.1% |
| **Total** | **262** | **35** | — |

Sixteen responses failed the strict format parser. Integrity checks confirmed
1,485 unique `(fact_id, trial)` pairs, five trials for every fact, and mutually
exclusive classifications for all 297 facts. SGLang was then stopped and Slurm
allocation `57794970` was relinquished.

Durable outputs are in `runs/deepseek-v4-flash/fact-knowledge/`:

- `run_config.json` records the exact model, endpoint, prompt format, selection,
  decoding, and classification settings.
- `source_manifest.json` records source paths, counts, and SHA-256 hashes.
- `fact_eval_progress.jsonl` contains every raw trial response.
- `known_facts.json` and `unknown_facts.json` contain final classifications.
- `fact_eval_checkpoint.json` contains aggregate completion counts.
- `evaluator.log` is the chronological evaluation log.

## Startup launcher optimization: 2026-08-31

The launcher gained explicit `fast`, `balanced`, and `throughput` startup
profiles. Static validation used `bash -n` and mocked the environment and Python
entry point to verify each generated SGLang command without GPUs or a Slurm job.
The default was subsequently restored to `throughput` for workloads consisting
of many small prompts, where graph replay reduces repeated decode-launch
overhead. The expected 268-second saving for opt-in `fast` comes from the measured graph
capture and remains to be confirmed end to end on four A100s. The dominant
770-second weight-remap phase is unchanged.

## Packed-checkpoint design: 2026-08-31

Source inspection confirmed that the repeat-start remap converts each MoE
layer's native MXFP4 weights and UE8M0 block scales into six persistent tensors:
`w13_mxfp4`, `w13_shift2`, `w13_channel_scale`, `w2_mxfp4`,
`w2_shift2`, and `w2_channel_scale`. Three Triton kernels compute the
per-channel exponent, pack the two-bit shifts, and rewrite the packed FP4
layout. SGLang currently allocates native checkpoint parameters first and calls
this conversion serially from each layer's `process_weights_after_loading`.

The preferred acceleration is a one-time, four-GPU offline conversion into
rank-local safetensors, followed by a custom loader path that allocates and
loads the six final tensors directly. The checkpoint manifest must bind the
packed data to the source checkpoint hash, tensor-parallel size, A100 port and
SGLang revisions, headroom bits, and packing-format version. Startup validation
must compare tensor hashes or samples, server output, and warm throughput
against the runtime-remapped baseline before making the packed path default.

This should remove nearly all of the measured 770-second repeat-start phase,
replacing it with reads of similarly sized packed weights. A smaller first step
is to add per-layer timings and benchmark batched or concurrent remapping, but
kernel optimization cannot avoid paying the full transformation on every
launch. CPU-side offline conversion is not attractive: the existing fallback
materializes expanded codes and is likely slower and more memory-intensive than
the Triton path.

## One-fact addition setup: 2026-08-31

`filler/addition/one_fact.py` adds a factual-only baseline-versus-filler test.
It uses the already classified known-fact set, assigns each fact a deterministic
two-digit addend, and asks for the full integer sum (not modulo 10). There is no
numeric control. The baseline and filler records share a stable pair ID, fact,
addend, target, system instruction, and decoding settings; only the forced
assistant prefix differs. The defaults are no filler and 10, 20, 50, and 100
space-separated periods, each ending in `Answer: ` before greedy generation.

The model-free smoke construction completed with five prompts for the first
stably sorted known fact (Atatürk's age at death): addend 11 and target 68.
Unit tests cover full-sum targets, paired conditions, prompt prefix placement,
question wording, strict integer parsing, one-token validation, and score/rank
extraction. No server or Slurm job was started during this setup.

The evaluator subsequently gained first-token scoring using SGLang's pinned
`return_logprob`, `token_ids_logprob`, and `top_logprobs_num` interface. Before
each generation it compares `/tokenize` results for the prompt and
prompt-plus-target and fails unless the target is exactly one appended token.
Each result records the target probability and log-probability independently
of its rank, plus an exact top-20 rank or a `>=21` lower bound. The summary
computes paired changes from baseline for every filler length. Eight evaluator
unit tests and the four existing DeepSeek SGLang tests passed; prompt-only mode
still constructs the expected five prompts. Live scoring remains untested until
a DeepSeek server is started in an approved allocation.

To run the five-request smoke test against a warm server:

```bash
.venv-sglang/bin/python -m scripts.addition.one_fact \
  --output-dir runs/deepseek-v4-flash/one-fact-addition-smoke
```

## Two-fact addition setup: 2026-08-31

`filler/addition/two_fact.py` mirrors the one-fact protocol while requiring two
retrieved quantities. It orders known facts by a seed-keyed hash, forms
disjoint adjacent pairs, and asks for the full sum of the two numeric factual
answers. Each pair shares its facts, target, prompt wording, decoding settings,
and stable pair ID across the baseline and 10, 20, 50, and 100 forced-dot
conditions. The default smoke scope is one pair (five requests); `--max-pairs`
expands it and `--prompt-only` performs no server request or Slurm action.

The two-fact evaluator now also uses the one-fact scoring path: it requires the
full sum to be one continuation token, requests that token's exact log-probability
plus the top 20 next tokens, records probability and observed rank in each
result, and reports condition means and paired changes from baseline. Targets
that are absent from the returned top 20 retain an exact requested-token
probability and receive a `>=21` rank lower bound.

The current known-fact file has 262 records, which seed 42 assigns to 131
disjoint pairs with no leftover fact. The full five-condition analysis is 655
model generations; because each generation has two tokenizer preflight calls,
the client makes 1,965 HTTP requests in total. Generation progress is fsynced
after every result to `results_progress.jsonl`; the same command validates its
configuration and resumes compatible partial output.

The complete prompt-only manifest was constructed at
`runs/deepseek-v4-flash/two-fact-addition-full/`: it contains 655 unique prompt
IDs, 131 unique pair IDs, and exactly 131 prompts in each condition. All 15
focused one- and two-fact tests, compilation, and whitespace checks passed.
This preparation did not contact a server or start a Slurm job.

Validation was repeated on 2026-09-01 on the login node without GPUs: the
focused one- and two-fact suites passed all 14 tests, `py_compile` and
`git diff --check` passed, and a prompt-only smoke run constructed five prompts
for one pair with the scoring configuration recorded in `run_config.json`.
The run emitted expected local NVML/MUNGE warnings but exited successfully;
no generation result has yet been collected from a warm server.

```bash
.venv-sglang/bin/python -m scripts.addition.two_fact \
  --output-dir runs/deepseek-v4-flash/two-fact-addition-smoke
```
