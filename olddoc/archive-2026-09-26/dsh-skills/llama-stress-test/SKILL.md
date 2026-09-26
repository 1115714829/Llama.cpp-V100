---
name: llama-stress-test
description: Standard, parameterized llama.cpp stress test. Measures TTFT (prefill), decode (tg), and prefill wall time at a chosen context size for one or more builds (variants) on single/multi-GPU (1/2/4/6 cards). Use to benchmark llama.cpp builds (e.g., stock vs V100-optimized) for prefill/TTFT and decode speed, including large contexts that need multiple GPUs.
---

# llama-stress-test

Standard llama.cpp stress test. Measures **TTFT (time-to-first-token = prefill)**,
**decode speed (tg)**, and **prefill wall time** at a chosen context, for a given
llama.cpp build (variant), on 1..6 GPUs. Built to compare a stock build against
V100-optimized builds.

## When to use
- Compare two llama.cpp builds at a target context (e.g. stock `b11053` vs a
  CUDA-optimized build).
- Measure TTFT (prefill) + decode speed + prefill wall time.
- Large contexts (e.g. 256k) that need several GPUs to fit weights + KV.

For the project's *fixed-yardstick* single-stream decode acceptance
(`/root/p60-ab-harness.sh`, speculative decoding with AL + ms/round), prefer that
harness; this skill is the prefill/TTFT-oriented stress test. See
`1cat-vllm-v100-study/HANDOFF.md` and the workspace `AGENTS.md`.

## How to run
Runs on the GPU server (AC922: ppc64le + 6x V100). From DSH, use the shell tool with
`ssh`/`scp` directly (Windows OpenSSH; key already authorized):

1. `scp` `stress-test.sh` to the server (`/root/stress-test.sh`).
2. Run it with env-var parameters (one variant at a time):
   ```
   ssh -o BatchMode=yes root@192.168.50.235 "MODEL=... CONTEXT=262144 GEN=128 GPUS=2,5 REPS=2 BENCH_BIN=... VARIANT=c4 bash /root/stress-test.sh"
   ```
   Nested quoting through two shells is fragile: keep the remote command in a
   single `.sh`/`.bat` file (or a here-string) instead of long inline quotes.
3. Repeat per variant and compare the `SUMMARY` lines.

## Parameters (env vars)
| var | default | meaning |
|---|---|---|
| `MODEL` | (required) | gguf model path |
| `CONTEXT` | 262144 | context size / prompt length (256k = 262144) |
| `GEN` | 128 | generation (decode) length |
| `GPUS` | 2,5 | `CUDA_VISIBLE_DEVICES` list |
| `BENCH_BIN` | (required) | the `llama-bench` binary for this variant |
| `TENSOR_SPLIT` | `1/1` | **slash** syntax, one weight per visible GPU. `1/1` = even over 2 GPUs; `1/1/1/1/1/1` = even over 6 |
| `MAIN_GPU` | 0 | main gpu **index within the visible set** (pinned for the small tensors) |
| `UBATCH` | 512 | prefill micro-batch (`-ub`) |
| `REPS` | 1 | repeat count; **use >=2** — 256k prefill shows ~8% run-to-run drift |
| `FLASH_ATTN` | 1 | pass `-fa on` (required when KV is q8_0) |
| `CACHE_K` / `CACHE_V` | `q8_0` / `q8_0` | KV cache types (`f16` = higher quality, more VRAM) |
| `VARIANT` | run | label for the log file name + summary |

## Metrics
- **TTFT (prefill)**: `llama-bench -p CONTEXT` reports `pp` (tokens/s);
  **TTFT = CONTEXT / pp**.
- **Decode (tg)**: `llama-bench -n GEN` reports `tg` (tokens/s).
- **Prefill wall time**: wall time of the whole run, logged by the script.

## Measurement integrity (learned the hard way, 2026-09-20)
- **`llama-bench`/`llama-server` are ~210 KB launcher stubs; the real code is in
  `libggml-cuda.so` (~125 MB) + `libllama*.so`.** Their `DT_RUNPATH` is an
  ABSOLUTE path to the build dir, so **copying the executable alone does
  nothing** — it still loads the build dir's libs. Copy the **whole `build/bin/`
  directory** per variant and select it with `LD_LIBRARY_PATH` (`DT_RUNPATH` is
  searched *after* `LD_LIBRARY_PATH`, so this works; note `DT_RPATH` would NOT).
- **Validity self-check (mandatory):** `md5sum` the libraries that actually carry
  the change in both dirs — `libggml-cuda.so.0.24.0` **and**
  `libllama-common.so.0.4.1` (speculative-decoding code lives in the latter).
  If the md5s are EQUAL, the A/B is meaningless. An earlier attempt compared a
  stub against itself and produced a bogus "no effect" result.
- **Record `--version` of both sides and keep them on the SAME source commit.**
  A previous comparison silently put upstream `434ddbb` against `b11053`+patch.
- **Same prompt, same context, same KV type, same ubatch, same card count** on
  both sides. Everything but the code under test must match.
- **Repeat >=2x and report the spread**, not just one number. Long prefill on
  this box drifts ~8% between identical runs; `-r 3` plus interleaving is the floor.
- Free the GPUs first (`nvidia-smi` should be quiet) — other tenants shift results.

See `1cat-vllm-v100-study/same-source-ab.md` for the full write-up and results.

## Models on AC922 (see `~/.qwen/skills/ssh-server/HOSTS.md`)
- `Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf` — 9.14 GiB Q2_K. Fits 1 card;
  the fast-iteration bench model. Low quality is fine here (t/s only).
- `Qwen3.8-27B-TurboFCFusion-gguf/...Q4_K_M.gguf` — **17.23 GiB** (+
  `mmproj-F16.gguf`). The real production model; **does not fit one 16 GiB V100**
  → needs >=3 cards at 256k.
- `Qwen3.8-27B-Q8_0.gguf` — 29 GB; the fixed-yardstick acceptance model for the
  1cat comparison (needs 3 cards).
- Card-count guidance: 1 card = small models only; 3-4 cards for the 17 GiB
  model at 256k; 6 cards for maximum headroom. **Prefer GPU groups inside one
  NUMA node** (0/1/2 or 3/4/5) — cross-NUMA splits traverse `SYS` and are slower.
  Measured: **TP3 (0,1,2) is the throughput optimum**, TP4+ regresses.

## Sampling parameters (production model the user pointed at)
`bbkdevops/Qwen3.8-27B-TurboFCFusion-CyberSec` on HF ships
`generation_config.json` with `temperature 0.7, top_p 0.8, top_k 20,
repetition_penalty 1.05, do_sample true`. Use these for the production model
unless the user says otherwise. `llama-bench` does not sample (fixed seed), so
these only matter for the server route below.

## MTP / speculative decoding — use `llama-server`, not `llama-bench`
`llama-bench` cannot do speculative decoding. To measure MTP:
- Run `llama-server` with `--spec-type draft-mtp --spec-draft-n-max N`
  (no separate draft file: `draft-mtp` uses the target model's built-in NextN
  head). Disable it with `--spec-type none` (default).
- Send a request to `/v1/chat/completions` (raw `/completion` with plain text makes
  an instruct model emit EOS immediately) and read the timing lines:
  `prompt eval time` = TTFT, `eval time` = effective tg,
  `draft acceptance = X (A accepted / B generated), mean len = Z`.
- **Caveat**: with speculation, `eval time = ms / N tokens` is effective
  throughput, so it moves with the **acceptance rate** as well as kernel speed.
  To attribute a change to the kernel, either disable MTP, or fix the sampling
  seed so both sides generate the same text. Degenerate/repetitive output is
  very easy for n-gram speculation to accept and will inflate tg.
  **Always report AL (acceptance length) and ms/round (`AL / tg`), never t/s alone.**
- The MTP sampler runs on CPU under `--split-mode tensor` (backend sampling is not
  implemented for tensor split). Measured 2026-09-20: **`tensor` + CPU sampling
  (85.81 t/s) still beats `layer` + GPU sampling (77.06 t/s)** ⇒ keep
  `--split-mode tensor`; do not chase the CPU sampler (upper bound ~7%).
  An earlier note claiming "MTP makes tg slower" was wrong — MTP measured **+107%**
  on same-context prompts (41.33 → 85.81). See `mtp-sampler-cpu.md`.

## Notes
- 256k prefill on V100 is LONG: roughly 10-11 min even on 2 cards. Plan for it;
  poll the log instead of using one huge blocking timeout, and `nohup` anything
  over ~10 min.
- Correctness first: verify each build with `llama-cli` (sane short generation)
  before trusting t/s; for speculative changes compare AL and greedy sha256
  (`temperature=0`) or `llama-perplexity` (<=0.1% relative change).
- **DFlash2** originally came from 1cat-vLLM (n-gram draft + top-k/top-p sparse
  rejection + `_requires_sm70_tail` for CC 7.0). llama.cpp b11053 does support
  `--spec-type draft-dflash` (the draft GGUF is on the box), but measured MTP
  n=4 is faster overall; see `dflash-in-llamacpp.md`.
