# DeepSeek V4 Flash: four-A100 audit

Audit date: 2026-08-31

Startup attribution was corrected on 2026-09-09; see the final dated entry for
the opt-in kernel-cache implementation and pending GPU validation protocol.

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
| Enumerate 48 mmap-backed shards | about 8 s | Does not include all deferred reads or GPU transfers |
| Weight loading and preparation | 770 s | Dominant interval; compilation/read/remap attribution was not measured |
| CUDA graph capture | 268 s | Secondary startup bottleneck |
| First small HTTP request | 13.56 s | Includes first-inference effects |

Server readiness took roughly 17m45s. Weight memory was about 38.84 GiB per
GPU. About 10.8 GiB per GPU remained after KV-pool allocation and graph capture.
The configured token capacity was 1,346,560.

`time -v` recorded 1,875 seconds system CPU, 44.6 million major page faults, and
322 GB filesystem input over the successful 19m34s process lifetime. Together
with the eight-second shard enumeration, this is consistent with deferred reads
and weight preparation. It does not isolate their cost from JIT compilation.
The per-layer attribution previously made here is superseded by the
2026-09-09 startup investigation below.

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

1. Time deferred reads, assignment/transfers, expert remapping, and JIT separately;
   test reusable kernel bundles before considering prepared-weight checkpoints.
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
profiles do not address the approximately 770-second weight-loading interval.
Its dominant component is not yet measured. A prepared-weight checkpoint is
one possible optimization, deferred until timing identifies the bottleneck.

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
770-second weight-loading interval is unchanged.

## Packed-checkpoint design: 2026-08-31

Source inspection confirmed that the repeat-start remap converts each MoE
layer's native MXFP4 weights and UE8M0 block scales into six persistent tensors:
`w13_mxfp4`, `w13_shift2`, `w13_channel_scale`, `w2_mxfp4`,
`w2_shift2`, and `w2_channel_scale`. Three Triton kernels compute the
per-channel exponent, pack the two-bit shifts, and rewrite the packed FP4
layout. SGLang currently allocates native checkpoint parameters first and calls
this conversion serially from each layer's `process_weights_after_loading`.

The originally proposed acceleration was a one-time, four-GPU offline conversion into
rank-local safetensors, followed by a custom loader path that allocates and
loads the six final tensors directly. The checkpoint manifest must bind the
packed data to the source checkpoint hash, tensor-parallel size, A100 port and
SGLang revisions, headroom bits, and packing-format version. Startup validation
must compare tensor hashes or samples, server output, and warm throughput
against the runtime-remapped baseline before making the packed path default.

The earlier prediction that this would remove nearly all 770 seconds was not
supported by component timings and is withdrawn. This design remains deferred;
the current investigation targets JIT caching and startup instrumentation first.
CPU-side offline conversion is not attractive: the existing fallback
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

## Startup timing and reusable TVM-FFI kernels: 2026-09-09

Objective: reduce repeat startup without changing model weights, quantization,
kernel selection, hooks, or serving settings. Prepared-weight checkpoints remain
deferred. No Slurm job or model server was launched during implementation.

Implementation baseline: workspace revision `10763c9b6b31abe1e05a7700c9c4a092e47a00d7`
with pre-existing uncommitted work; the A100-port/SGLang revisions listed above
remain unchanged. Installed packages: torch 2.11.0, apache-tvm-ffi 0.1.9,
flashinfer-python 0.6.11.post1, triton 3.6.0, Ninja 1.13.0.

### Corrected evidence and attribution

The completed [job 58129149 startup log](runs/deepseek-v4-flash/one-fact-patching-discovery/runtimes/58129149-6a8c8f39011a/server.log)
records weight loading from 14:11:54 to 14:24:32 UTC: 757.72–758.07 seconds across
TP ranks. Layer 0 preparation finishes at 14:24:31, and every rank reaches
layer 42 by 14:24:32. Readiness follows at 14:24:35. This does not support the
previous assertion that repeated per-layer repacking explains nearly all of
startup. JIT prewarm on the first layer and deferred mmap reads/GPU transfers
are leading hypotheses, not measured conclusions. The eight-second progress
bar measures shard enumeration, not all I/O. CUDA graphs are already disabled
in this patching workflow, so disabling them again offers no saving there.

[TVM-FFI's official load_inline documentation](https://tvm.apache.org/ffi/reference/python/cpp/generated/tvm_ffi.cpp.load_inline.html)
documents persistent compiled modules through `TVM_FFI_CACHE_DIR`. The installed
implementation still invokes Ninja when loading a cached module; it writes
generated sources only when their contents change. Consequently, a reusable
bundle preserves objects, shared libraries, generated sources, recipes,
`.ninja_deps`, `.ninja_log`, and timestamps at the same absolute build path.

### Implementation and boundaries

- [run_deepseek_v4_startup.sh](run_deepseek_v4_startup.sh) is an experimental copy
  of the A100 launcher with cache preparation before the server `exec`. The
  original launcher is frozen by the campaign manifest and remains unchanged.
  A regression test compares all effective serving arguments across all three
  startup profiles, including caller-supplied activation hooks. The new launcher
  accepts `SGLANG_PORT` only as a convenience input, removes it from the server
  environment, and checks the effective HTTP port before loading the model.
- [filler/dsv4/startup_cache.py](filler/dsv4/startup_cache.py) implements
  `DSV4_KERNEL_CACHE_MODE=off|reuse` (default **off**) and
  `DSV4_KERNEL_CACHE_ROOT` (default `runs/deepseek-v4-flash/startup-cache/`).
  Active TVM builds use private `/tmp/dsv4-kernels-UID/FINGERPRINT/tvm` storage,
  independent of job-specific `TMPDIR`. Triton and FlashInfer settings are
  unchanged; this implementation transfers only the TVM cache.
- Fingerprints cover A100-port and SGLang JIT Python/CUDA/header contents,
  CUTLASS, TVM-FFI/DLPack and CUDA headers, the TVM runtime library, compiler
  and package/runtime versions, compiler flags/environment, source paths, and
  visible GPU architectures. Collection occurs on the compute node before
  loading weights. It performs no weight conversion or model-weight hashing.
- A complete bundle contains a versioned identity, absolute local path, file
  checksums/sizes, and validation provenance. Private permissions, symlink
  rejection, checksums, and completion markers reject unsuitable bundles.
  Interrupted/invalid caches are renamed and retained for diagnosis; they are
  not accepted as same-node hits. Restoration falls back to a local build.
  Publication uses an atomic directory rename while holding local, bundle,
  and TVM module locks. Concurrent conflicting publishers fail safely.
- `DSV4_STARTUP_TIMING=1` installs the
  [startup bootstrap](filler/dsv4/startup_bootstrap/sitecustomize.py) ahead of
  the original A100 port. It times enumeration, assignment/transfers, post-load
  processing, per-layer expert preparation, remap calls, JIT prewarm, individual
  TVM module builds/loads, and Ninja build edges. GPU boundaries synchronize
  only when timing is enabled. Per-process JSONL includes rank, timestamps,
  elapsed time, CPU percentage, process/child CPU, page faults, block I/O, and
  `/proc/self/io` counters. Set `DSV4_STARTUP_TIMING_DIR` to retain the files;
  otherwise timing goes to stderr. Nested spans are inclusive and must not be
  summed as independent costs. Compilation counts describe TVM/Ninja only.

Example opt-in launch, **inside a separately approved four-A100 allocation**:

```bash
DSV4_KERNEL_CACHE_MODE=reuse DSV4_STARTUP_TIMING=1 \
  bash run_deepseek_v4_startup.sh [existing serving and activation-hook arguments]
```

### Verification and GPU comparison procedure

[scripts/dsv4/benchmark_startup.py](scripts/dsv4/benchmark_startup.py) runs CPU
preflight before any load. `--preflight` does only those checks. Its GPU path
requires the approved four-hour interactive allocation, account `m5258_g`, and
four A100s with at least 79 GiB each. It records actual account/QOS, job ID, node,
GPU capacities, and UTC start/end times. It launches and stops only its own
server, never submits or cancels a Slurm job, and preserves campaign sources.

The protocol uses the existing campaign hooks and native final-layer validator:
exact returned residual and TP-rank agreement, identical argmax, and maximum
candidate/top-token logprob error <=0.15. Three fixed representative tokenized
prompts from the first frozen campaign panel are saved once and reused. All
greedy output IDs must match across repetitions and phases. One warm cycle is
discarded; the median throughput of five further cycles must remain at least
95% of the cold baseline. Throughput includes HTTP request time and excludes
artifact writes to Scratch. The workflow records current source checksums and
checkpoint shard metadata, rejecting changes between phases or during a run.
The controller records time through readiness,
the first response, and completion of its native validation, so CPU readout
validation cost is visible separately.

After validation, the owned server exits and the workflow publishes its bundle.
The `cold` phase starts with an empty TVM cache (retiring any prior local cache),
`same` requires a validated cache on the original node, and `fresh` requires
restoration on a different node with an identical fingerprint. Shared output
paths must be new for each comparison; completed phase outputs are never
overwritten. All benchmark bundles below are inside the requested startup-cache
root. The commands are prepared for review, **not yet submitted**:

```bash
# CPU only; safe on the login node.
.venv-sglang/bin/python -m scripts.dsv4.benchmark_startup \
  --root runs/deepseek-v4-flash/startup-cache/validation-2026-09-09 --preflight

# After explicit approval: empty cache + same-node restart in allocation A.
salloc --account=m5258_g --qos=interactive --nodes=1 --gpus=4 \
  --constraint='gpu&hbm80g' --time=04:00:00 \
  srun --ntasks=1 --cpus-per-task=128 --cpu-bind=none --gpus=4 \
  .venv-sglang/bin/python -m scripts.dsv4.benchmark_startup \
  --root runs/deepseek-v4-flash/startup-cache/validation-2026-09-09 --phase cold-same

# After explicit approval: replace NODE_A with cold/result.json's hostname.
salloc --account=m5258_g --qos=interactive --nodes=1 --gpus=4 \
  --constraint='gpu&hbm80g' --time=04:00:00 --exclude=NODE_A \
  srun --ntasks=1 --cpus-per-task=128 --cpu-bind=none --gpus=4 \
  .venv-sglang/bin/python -m scripts.dsv4.benchmark_startup \
  --root runs/deepseek-v4-flash/startup-cache/validation-2026-09-09 --phase fresh
```

Each phase retains `runtime.json`, `cache.json`, `server.log`, `timing/*.jsonl`,
`timing-summary.json`, `native/`, repeated raw `responses/`, and `result.json`.
`bundles/FINGERPRINT/COMPLETE.json` records the published cache inventory and
validation provenance. Timing includes fingerprint and restore costs. Fresh-node
default eligibility additionally requires zero TVM build edges and positive
time savings through both readiness and validated response. The script only
reports eligibility; it never changes the launcher default.

The full CPU preflight passed **70 tests**. Verification covers invalidation, incomplete/corrupt bundles, active builds,
concurrent publication, file timestamps, environment/argument preservation,
startup hook order, disabled profiling behavior, and the comparison gates. A
real CPU Ninja recipe restored at its original path reports `no work to do`.
Initial verification exposed the NERSC group-writable default umask; reuse mode
now uses umask 077 for active cache creation. A first launcher mock accidentally
allowed site initialization; it was interrupted and corrected to use Python
`-S`. The final preflight record is retained at
[validation-2026-09-09/preflight-tests.log](runs/deepseek-v4-flash/startup-cache/validation-2026-09-09/preflight-tests.log).
The existing CPU suites emit sandbox MUNGE diagnostics on interpreter exit;
their checks do not submit jobs or require a scheduler connection.

The [static check report](runs/deepseek-v4-flash/startup-cache/validation-2026-09-09/static-checks.json)
also compares the 26 source hashes in the earlier discovery manifest. Five
existing files differ from that historical manifest (`patching.py`,
`patching_analysis.py`, `patching_campaign.py`, `one_fact_patching.py`, and
`test_one_fact_patching.py`); this task did not edit any of those files or the
original launcher. The new comparison records current code identities rather
than asserting that the historical campaign manifest describes today's entire
working tree. It reuses only that manifest's saved prompt IDs.

Measured GPU startup savings, fresh-node recompilation avoidance, inference
equivalence on the live model, and warm-throughput results are **pending explicit
Slurm approval**. Reuse remains opt-in. The remaining bottleneck cannot yet be
identified from the available timestamps.

### Approved sequential run on one node: 2026-09-09/10

The user approved one node and sequential runs. The benchmark now supports
`--phase sequential`: cold TVM cache, validated same-node restart, then forced
restoration from the published bundle on that same node. The restoration test
retires (preserves) the active local cache before copying the bundle. This does
not test fresh-node portability and cannot enable default reuse. The updated
CPU preflight passed 71 tests, including forced same-node restoration.

Allocation `58132972` was granted on `nid008312` with account `m5258_g`, internal
QOS `gpu_interactive`, one node, 128 logical CPUs, and four NVIDIA
A100-SXM4-80GB GPUs reporting 81920 MiB each. UTC allocation start/end:
2026-09-09 23:55:03 / 2026-09-10 03:55:03. Actual constraints include
`gpu&a100&hbm80g`. The allocation is retained through validation/restarts using
an interactive shell in the single `srun` step:

```bash
salloc --account=m5258_g --qos=interactive --nodes=1 --gpus=4 \
  --constraint='gpu&hbm80g' --time=04:00:00 \
  srun --ntasks=1 --cpus-per-task=128 --cpu-bind=none --gpus=4 \
  bash --noprofile --norc

# Within that compute-node shell, after the instrumented import check:
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 \
  .venv-sglang/bin/python -m scripts.dsv4.benchmark_startup \
  --root runs/deepseek-v4-flash/startup-cache/validation-2026-09-09 \
  --phase sequential
```

The pre-load import check uses the actual startup overlay and A100 port with
visible GPUs; artifacts are `import-check.log` and `import-check/*.jsonl` under
the comparison root. Results and any fixes will be recorded below.

The node was left idle after its initial import check while a separate host-tool
interaction was pending. This was an operational mistake; the user explicitly
instructed that future allocations must not wait idle for agent action. That
standing requirement is now in `AGENTS.md`. Control and monitoring now use the
original allocation shell, the full phase sequence runs automatically, and a
successful sequence signals that shell to exit and release the allocation.

The first cold startup was stopped early to fix an identity-serialization bug:
`platform.libc_ver()` returns a tuple, but persisted JSON contains a list.
Canonicalizing identity values through JSON prevents valid restored bundles
from being rejected. The corrected run uses
`runs/deepseek-v4-flash/startup-cache/validation-58132972/`, with 72 CPU tests
passing on the compute node. The earlier interrupted artifacts remain under
`validation-2026-09-09/`. Automatic `telemetry.log` records GPU memory/activity,
power, and process CPU/RSS/wait states every 15 seconds throughout the corrected
sequence. No source changes are planned while those measurements run.

The corrected cold run passed validation: readiness 755.306 s, first HTTP
response 925.121 s, first validated response 926.087 s; 21 TVM compilations and
42 Ninja build edges. Median warm throughput was 6.581 generated tokens/s for
these short two-token answers. Returned residuals and all TP ranks agreed
exactly, argmax matched, and maximum checked logprob error was 0.125 (within
the existing 0.15 tolerance). Weight enumeration took 4.22–4.34 s, assignment
and transfers 577.09–577.27 s, and summed expert prewarm about 98–99 s per rank.
Summed expert remapping was under 3 s on every rank. The first response's
additional 169.8 s includes further lazy kernel initialization/compilation.

The automatic next launch encountered an HTTP-port probe rejection after the
old server had exited (no listening socket or GPU process remained). The
probe did not set `SO_REUSEADDR` and treated TCP's post-close state as a busy
port. To preserve identical measured source hashes, the remaining phases were
immediately started with `--port 30013` and `--port 30014`; the port-probe fix
will follow after the measurements. Both phases remain sequential in the same
allocation, with logs in `restarts.log`, and release the shell automatically on
success. The same-node phase accepted fingerprint
`7dd62425cd6f4a82d59170a6d6bfa9d61a93cfb2d4e563849a0742e652ca1629`.

The same-node restart reached readiness in 260.230 s and validated response in
273.151 s with zero TVM compilation, identical generated IDs, exact returned
residual/rank agreement, and the same maximum logprob error of 0.125. Its five
warm cycles measured a median 5.878 tokens/s, 10.7% below the cold 6.581 baseline,
so the performance gate failed. That failure is retained in `same/result.json`.

To investigate sustained throughput without changing inference, the controller
was extended to retain failed performance results while continuing independent
correctness-validated restoration tests, and to accept `--warm-cycles 60`.
`--allow-driver-update` permits and records changes only to the benchmark driver
and its CPU test file; model, hooks, instrumentation, cache implementation, and
checkpoint identities must still match the baseline. Its new guard test ensures
that a cache/inference source change cannot be waived. Compute preflight for
the follow-up runs is in `followup.log`.

Two follow-ups run automatically and sequentially in the existing allocation:
`restored --port 30014 --warm-cycles 60 --allow-driver-update`, followed by
`rebuild --port 30015 --warm-cycles 60 --allow-driver-update`. The latter retires
the local TVM cache and recompiles, providing a control with warmed filesystem
caches. A throughput failure still sets `passed=false` and prevents publication;
it no longer discards the build/timing summary or prevents a separate diagnostic
phase from running. The original 5% threshold is unchanged. GPU clock and
temperature samples were added to the external telemetry for these follow-ups.

At the user's request, `scancel 58132972` was issued from the existing compute
shell on 2026-09-10 at approximately 02:21:59 UTC. Slurm confirmed cancellation
at **02:22:00 UTC** and reported that the allocation was revoked; the `salloc`
session exited. No replacement allocation was requested. The restoration run
was interrupted before a completed measurement, and the rebuild control was
not started. Their performance results must not be inferred from the completed
cold/same-node pair.

The durable [summary.json](runs/deepseek-v4-flash/startup-cache/validation-58132972/summary.json)
records the stop reason and available results. Completed measurements show a
495.08-second reduction through readiness and 652.94 seconds through validated
response on the same-node restart, combining warm filesystem/cache effects with
avoided compilation. Cold TVM builds totaled 99.79 seconds before readiness and
146.51 seconds afterward across compiler processes. Repeated expert remapping
was a small cost; deferred weight assignment/transfers remain the main startup
target. Identical tokens/native equivalence passed, but short-prompt warm
throughput dropped 10.7%, so **reuse remains opt-in**. Fresh-node portability,
sustained-throughput follow-up, and the HTTP probe's `SO_REUSEADDR` fix remain
unfinished. The user's stop request takes precedence over those pending tasks.
