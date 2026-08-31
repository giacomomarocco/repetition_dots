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

Smoke test from the same node:

```bash
curl http://127.0.0.1:30002/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"The capital of France is","sampling_params":{"temperature":0,"max_new_tokens":8}}'
```
