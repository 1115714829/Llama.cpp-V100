# RESEARCH: 1cat-vLLM V100 native implementations - push-based allreduce and FP16 GEMV

Output of this file: F:/vllm+llama.cpp/1cat-vllm-v100-study/RESEARCH-allreduce-fp16gemv.md

Status: research only. Nothing under 1cat-vllm/, vllm/, vllm-forkpoint/ or llama.cpp/ was modified.
No git commit / push / PR action was performed.

## 0. Scope, method, evidence legend

Trees read (read-only): 1cat-vllm (v1.5.0-672), llama.cpp (HEAD 79504e72b, base tag b11053).
GPU topology re-verified live on the AC922 with nvidia-smi topo -m during this session:
GPU 0/1/2 are pairwise NV2, GPU 3/4/5 are pairwise NV2, and every cross-island pair is SYS.
All six are Tesla V100-SXM2-16GB at PCIe link gen 3. NV2 = 2 bonded NVLinks.
NVLink 2.0 is 25 GB/s per link per direction, so NV2 is 50 GB/s per direction per GPU.
Cross-island SYS = PCIe plus the POWER9 SMP interconnect (X-Bus).

Evidence tags used below:

| Tag | Meaning |
|---|---|
| [CODE] | read directly from source in this session |
| [MEASURED] | a concrete number recorded in a file in the repo, cited with file:line |
| [CLAIM] | a prose assertion in a design doc, not itself a raw measurement |
| [DERIVED] | arithmetic I performed on top of [MEASURED] values - my number, not the repo number |
| [UNVERIFIED] | could not be confirmed from the tree |

---

# Part A - TP4 / SM70 push-based allreduce

## A1. Code map

| Concern | File:line |
|---|---|
| Push kernel, plain allreduce | 1cat-vllm/csrc/custom_all_reduce.cuh:760-835 |
| Push kernel, fused two-input sum2 | 1cat-vllm/csrc/custom_all_reduce.cuh:837-915 |
| Per-lane FP32 ordered reduce | 1cat-vllm/csrc/custom_all_reduce.cuh:700-708 |
| NaN sentinel helpers | 1cat-vllm/csrc/custom_all_reduce.cuh:666-674 |
| volatile 16 B peer load/store | 1cat-vllm/csrc/custom_all_reduce.cuh:676-698 |
| Buffer geometry constants | 1cat-vllm/csrc/custom_all_reduce.cuh:83-141 |
| Grid-size admission policy | 1cat-vllm/csrc/custom_all_reduce.cuh:143-192 |
| Dispatch from allreduce() | 1cat-vllm/csrc/custom_all_reduce.cuh:1940-1955 |
| Dispatch from allreduce_sum2() | 1cat-vllm/csrc/custom_all_reduce.cuh:2195-2239 |
| Buffer registration + sentinel init | 1cat-vllm/csrc/custom_all_reduce.cuh:1777-1810 |
| Class members | 1cat-vllm/csrc/custom_all_reduce.cuh:1693-1694 |
| C++ entry points | 1cat-vllm/csrc/custom_all_reduce.cu:1139-1141 and 1157-1168 |
| Torch schema | 1cat-vllm/csrc/torch_bindings.cpp:1096-1105 |
| Python wrappers | 1cat-vllm/vllm/_custom_ops.py:3228-3251 |
| Python enable and buffer creation | 1cat-vllm/vllm/distributed/device_communicators/custom_all_reduce.py:332-368 |
| Env declarations | 1cat-vllm/vllm/envs.py:277-282 and 2403-2437 |
| Env publishing into os.environ | 1cat-vllm/vllm/envs.py:5037-5048 |
| GLM5 DFlash2 forced rollback | 1cat-vllm/vllm/config/vllm.py:365-399 |

Provenance comment, verbatim intent: the protocol is "Adapted from SGLang-V100 one-shot push
collective at haohervchb/sglang-V100 at commit 845b9fdf7a7e", with the empty-slot sentinel changed
from the SGLang positive zero to a reserved FP16 NaN payload
(1cat-vllm/csrc/custom_all_reduce.cuh:662-665). The design doc repeats the upstream pin and the
SGLang measurement (1cat-vllm/docs/design/sm70_dflash2_target_graph_20ms.md:357-359).

## A2. Algorithm - one-shot push, not allgather+reduce, not a ring

Per call, per rank, per 16-byte packed lane [CODE], custom_all_reduce.cuh:778-829:

1. Load the local input pack P (16 B = 8 x half).
2. Escape any payload whose bit pattern equals the sentinel (:782-784, :666-669).
3. Store that same pack into EVERY peer IPC buffer, including the own-rank slot (:786-794);
   that is 4 volatile 16 B stores per lane per rank at TP4.
4. Spin-poll all 4 source slots (own plus 3 peers) with ld.volatile.global.v4.b32 until no slot
   still holds the sentinel (:796-812).
5. Reduce the 4 packs with an FP32 accumulator in fixed rank order 0,1,2,3 and downcast
   (:814-815 and :700-708).
6. Write the sentinel back into all 4 slots so the next epoch sees them empty (:817-828).
7. After the per-thread loop, __syncthreads() and thread 0 of each block advances the per-block
   epoch word: local_epochs[blockIdx.x] = (epoch + 1) % 2 (:831-834).

Key properties:

- It is a broadcast-then-local-reduce ("push") collective, not an allgather plus tree, and not a
  ring. Every rank writes its full payload to every peer and every rank independently computes the
  identical FP32 sum in identical rank order, so the result is bitwise identical across ranks by
  construction. The same intent is stated for the pull kernel at :730-731 ("we do not reorder the
  address so the accumulation order is the same for all ranks, ensuring bitwise identical results").
- Wire cost is n*(n-1)*S bytes for TP4, i.e. 2x the bytes of a ring (2*(n-1)/n*n*S), traded for ONE
  network hop of latency instead of 2*(n-1) steps. [DERIVED] - standard result, not stated in repo.
- Double buffering is exactly 2 epochs (kSm70Tp4PushAllreduceEpochs = 2,
  custom_all_reduce.cuh:86). Two epochs are sufficient only because the collective is a barrier:
  a rank cannot reach the write phase of collective k+1 before every peer has finished reading
  epoch k, so no rank can be more than one epoch ahead. [DERIVED].
- Empty-slot detection uses the exact bit pattern 0x7f7f, an FP16 quiet NaN
  (kSm70Tp4PushAllreduceSentinel = 0x7f7f, :87). The whole data region is pre-filled with byte 0x7f
  at registration (:1790-1794, byte constant :88). A finite payload that collides is remapped to
  0x7e00 (:666-669), so every FINITE input is preserved, including both signed zeros. Two doc
  entries record why the naive positive-zero sentinel was rejected: it turned an all-positive-zero
  reduction into negative zero with 40,960 bit mismatches
  (docs/design/sm70_dflash2_target_graph_20ms.md:368-371); and a release/acquire CTA-ready
  alternative was tried and rejected because it regressed the 128-call chain to 1.884-1.901 ms
  (same doc, :373-378).

## A3. Transport - CUDA IPC over P2P-mapped device memory

[CODE] findings:

- The collective reads and writes RankData::ptrs[rank], obtained from
  cudaIpcOpenMemHandle(..., cudaIpcMemLazyEnablePeerAccess)
  (custom_all_reduce.cuh:1727-1729). The buffer itself is created with plain cudaMalloc plus
  cudaIpcGetMemHandle in allocate_shared_buffer_and_handle (custom_all_reduce.cu:1206-1237).
- It therefore requires genuine P2P. Registration hard-fails otherwise:
  register_sm70_tp4_push_buffer throws unless world_size_ == 4 AND fully_connected_ AND
  custom_all_reduce_current_device_is_sm70() (custom_all_reduce.cuh:1777-1782).
- fully_connected is decided in Python by current_platform.is_fully_connected(physical_device_ids);
  TP greater than 2 without it disables custom allreduce entirely
  (vllm/distributed/device_communicators/custom_all_reduce.py:216 and :219-233).
- The peer stores are ordinary volatile 16 B global stores issued by the SM. There is no multimem,
  no NVLink SHARP, and no copy-engine involvement.
- The buffer is cudaMalloc-allocated outside any graph-capture allocator window, so it is NOT part
  of the register_graph_buffers / get_graph_buffer_ipc_meta address bookkeeping that the ordinary
  pull path needs (custom_all_reduce.cuh:1735-1753 and :1863-1894).

## A4. Buffer geometry

Constants at custom_all_reduce.cuh:83-141. [DERIVED] arithmetic, integer division, sizeof(half)=2:

    kSm70Tp4PushAllreduceBlocks        = 80
    kSm70Tp4PushAllreduceThreads       = 128     (one 16 B pack per thread)
    kSm70Tp4PushAllreduceEpochs        = 2
    kSm70Tp4PushAllreduceWorldSize     = 4
    kSm70Tp4PushAllreduceM8Bytes       =  8 * 5120 * 2 =  81,920 B  (80 KiB)
    kSm70Tp4PushAllreduceM16Bytes      = 16 * 5120 * 2 = 163,840 B  (160 KiB)
    kSm70Tp4PushAllreduceM32Bytes      = 32 * 5120 * 2 = 327,680 B  (320 KiB)
    kSm70Tp4PushAllreduceMaxBytes      = M32 = 327,680 B             (slot stride basis)
    kSm70Tp4PushAllreduce8KiBBytes     = 4096 * 2 = 8,192 B
    kSm70Tp4PushAllreduceQwen4ExpBytes = 2560 * 2 = 5,120 B
    kSm70Tp4PushAllreduceSignalBytes   = ceil((80*4 + 127)/128)*128 = 512 B
    kSm70Tp4PushAllreduceGenericBufferBytes = 512 + 2 * 4 * 327,680 = 2,621,952 B
    kSm70Tp4PushAllreduceBufferBytes        = 2,686,208 B total (2.562 MiB, includes HC regions)

So: about 2.50 MiB of generic allreduce region per rank (512 B of epoch words plus 2 epochs x
4 ranks x 320 KiB payload stride), inside a 2.56 MiB total buffer per rank that also carves out
HC (head-channel) signal, down, gate and fused-up regions (:109-141).
The slot stride is always the M32 size, so an M8 collective wastes 3/4 of each slot.
The doc states the same thing as "expands its IPC slot"
(docs/design/sm70_quasar_nvfp4_dflash2_acceptance.md:489-490).

The M8 payload is a real verifier shape, not a generic GEMV: it is [8 tokens x 5120 hidden] FP16,
and the enable log calls it the "FP16 80-KiB verifier"
(vllm/distributed/device_communicators/custom_all_reduce.py:360-364).

## A5. Grid-size admission policy - the real content of the switch family

sm70_tp4_push_allreduce_blocks(bytes, allow_generic) at custom_all_reduce.cuh:143-192.
Line numbers below are inside that function.

| Condition | Result | Line |
|---|---|---|
| SMALL_MESSAGES=1, allow_generic, 0 < bytes <= M8Bytes, bytes % 16 == 0 | ceil(bytes / (128*16)) - minimal covering grid | :148-154 |
| bytes == M8Bytes, or CONCURRENCY=1 and (bytes == M16Bytes or bytes == M32Bytes) | 80 | :155-162 |
| bytes == 8 KiB | 4 | :163-165 |
| bytes == 5,120 (Qwen4Exp M1) | 3 | :166-168 |
| QWEN38_BATCH enabled and bytes == 20,480 (M4) or 40,960 (M8) | env override in [ceil(bytes/2048), 80], else 10 or 20 | :169-186 |
| MTP5=1 and bytes == 25,600 | 13 | :187-191 |
| anything else | 0, push is declined, caller falls through to the pull path | :191, :1949, :2230 |

Effective values [DERIVED]: M8 -> 40 blocks (not 80), 8 KiB -> 4, 5,120 B -> 3.
The comment at :179-180 is the correctness note: the admission must be conservative because the
kernel "handles one pack per thread, without a grid-stride loop. An undersized launch silently
leaves the output tail unwritten." Note that the kernel DOES contain a grid-stride loop at
:778-779. Both statements exist in the source; I did not run the kernel to decide which is
authoritative for every shape, so treat the comment as the conservative reading.

## A6. Switch family and defaults

Declared defaults, vllm/envs.py:277-282 with resolved lambdas at :2403-2437:

| Switch | Declared default | Effect |
|---|---|---|
| VLLM_SM70_TP4_PUSH_ALLREDUCE | 1 (:277, :2403-2406) | Master gate; decides whether the push buffer is allocated at all (custom_all_reduce.py:332-348) |
| VLLM_SM70_TP4_PUSH_ALLREDUCE_CONCURRENCY | 1 (:278) | Admits M16/M32 payloads into the push path at the full 80-block grid. Without it those sizes return 0 and use the pull path |
| VLLM_SM70_TP4_PUSH_ALLREDUCE_MTP5 | 0 (:279) | Opt-in admission of the 25,600 B payload at 13 blocks |
| VLLM_SM70_TP4_PUSH_ALLREDUCE_QWEN38_BATCH | 1 (:280) | Admits 20,480 B / 40,960 B payloads at 10 / 20 blocks (Qwen3.8 hidden = 2560) |
| VLLM_SM70_TP4_PUSH_ALLREDUCE_SUM2_M1 | 1 (:281) | Admits the fused two-input allreduce_sum2 push for the 5,120 B M1 payload |
| VLLM_SM70_TP4_PUSH_ALLREDUCE_SMALL_MESSAGES | 1 (:282) | Minimal-covering-grid admission for any payload <= 80 KiB and 16 B aligned |

Two mechanisms decide what actually reaches the C++:

1. SMALL_MESSAGES and CONCURRENCY are PUBLISHED into os.environ by vllm/envs.py:5037-5048,
   because the native code reads them with std::getenv at kernel-launch time. This was added after
   an audit found the declarations were dead (docs/design/sm70_qwen38_default_fastpath.md:117-124).
   QWEN38_BATCH, SUM2_M1 and MTP5 are not published, but their C++ tests already default to
   enabled when the variable is absent (:170, :2209-2210, :188-190), matching the declared
   defaults. Only MTP5 needs an explicit 1, and both sides default it off.
2. One model-specific forced rollback exists. vllm/config/vllm.py:365-399 auto-sets
   VLLM_SM70_TP4_PUSH_ALLREDUCE=0 when model_type is glm5_next or glm5_next_text, the speculation
   method is dflash, and TP is 4. An explicit non-zero value is preserved but emits a warning.
   The stated reason is that the push collective FAILED AN OUTPUT-QUALITY AUDIT for that route.

THE "all-fast graph + PUSH_ALLREDUCE=1 rejected" ITEM IN THE TASK BRIEF IS A QUALITY REJECTION,
NOT A CUDA-GRAPH CAPTURE REJECTION. Evidence: docs/design/sm70_v100_migration_control.md:44804-44828.
The matched graph with CUDA Graph and all DFlash fast paths enabled but push all-reduce disabled
scores 59/60 GSM8K. Enabling push all-reduce drops it to 58/60, with case 20 becoming a new
low-margin divergence that reaches the 1024-token cap (:44817-44823). The same section records
108.934 tok/s for the push-off ordinary-custom-AR graph and 103.530 tok/s for the conservative
route (:44820-44821), and concludes the material quality difference is isolated to push all-reduce,
not to CUDA Graph or the DFlash compute path (:44822-44823). The mirrored entry is
docs/design/sm70_glm53_flash_nvfp4.md:317-321.

## A7. Measured latency and bandwidth

### A7.1 What is actually measured

Payload [8,5120] FP16 = 80 KiB, 128-call CUDA Graph chain:

| Variant | 128-call chain | Per call | Source |
|---|---|---|---|
| vLLM pull path | 1.854 ms | about 14.5 us [DERIVED] | sm70_dflash2_target_graph_20ms.md:387-389 |
| SGLang-V100 pin 845b9fdf7a7e | 0.846-0.886 ms | - | sm70_dflash2_target_graph_20ms.md:357-359 |
| first vLLM port | 0.873 ms | - | sm70_dflash2_target_graph_20ms.md:359-361 |
| CTA-ready acquire variant | 1.884-1.901 ms - REJECTED | - | sm70_dflash2_target_graph_20ms.md:373-375 |
| accepted NaN-sentinel push | 0.850-0.856 ms | 6.64-6.68 us | sm70_dflash2_target_graph_20ms.md:380-389 |

The accepted artifact is named at sm70_dflash2_target_graph_20ms.md:382-383 as
results/vllm-push-ar-nan-sentinel-correctness-timing-m8-h5120-v1.json.
THE JSON ITSELF IS NOT IN THIS CHECKOUT, so the 0.850-0.856 ms figure is a doc-recorded
measurement that references an absent artifact. Treat it as MEASURED-in-doc, not as something
that can be re-verified from this tree.

Per-collective medians across 128 consecutive collectives per CUDA Graph replay and four input
patterns, every rank bitwise equal to the custom-order reference
(sm70_quasar_nvfp4_dflash2_acceptance.md:491-493):

| Payload | Pull | Push | Saving | Source |
|---|---|---|---|---|
| M16 / 160 KiB | 18.45 us | 11.03 us | 40.2% | sm70_quasar_nvfp4_dflash2_acceptance.md:495-497 and sm70_v100_migration_control.md:45555-45558 |
| M32 / 320 KiB | 26.78 us | 18.36 us | 31.5% | same |
| M64 | 30.79 us | 34.08 us | regression, removed from admission | sm70_quasar_nvfp4_dflash2_acceptance.md:500-501 |

Retained artifacts named there are .artifacts/runtime/tp4-push-concurrency-control-r1.json and
.artifacts/runtime/tp4-push-concurrency-final-r1.json (:502-503). They are also NOT present in
this checkout.

In-graph service bucket inside a real full model: the mixed NVFP4 DFlash2 target graph attributes
1.294 ms to push all-reduce, against 3.327 ms for QPN8 projections, 2.511 ms for QPN2 gate/up and
1.200 ms for recurrent GDN (sm70_dflash2_target_graph_20ms.md:758-761). A separate historical B4
trace attributed 5.31 ms and about 133 launches per round to TP all-reduce
(sm70_quasar_nvfp4_dflash2_acceptance.md:486-487).

Endpoint-level effects [MEASURED]:

- VLLM_SM70_TP4_PUSH_ALLREDUCE_CONCURRENCY: C=16 aggregate throughput +3.4%. Off measured
  807.5/812.7/813.8/820.6, on measured 830.8/848.3/849.2/855.4/857.6 - non-overlapping intervals
  (sm70_qwen38_default_fastpath.md:106 and :112-115).
- VLLM_SM70_TP4_PUSH_ALLREDUCE_SMALL_MESSAGES: C=16 +1.0% (sm70_qwen38_default_fastpath.md:107).
- Both together against both off: C=1 +7.8% (flagged, ranges overlap), C=4 0%, C=8 -0.9% (flagged),
  C=16 +4.3% (sm70_qwen38_default_fastpath.md:108 and :112).
- Combined QPN8 plus push endpoint: B2 258.04 tok/s (68.7%), B4 317.11 tok/s (42.2%)
  (sm70_v100_migration_control.md:45560-45565).

### A7.2 Bandwidth - [DERIVED], my arithmetic, not in the repo

Per rank the push writes (n-1)*S bytes and reads back (n-1)*S bytes. The algorithm bandwidth of an
allreduce is 2*(n-1)/n * S / t.

| t | S | per-rank egress 3S/t | algorithm bandwidth | aggregate on wire n*(n-1)*S/t |
|---|---|---|---|---|
| 6.66 us | 81,920 B | 36.9 GB/s | 18.5 GB/s | 73.8 GB/s |
| 11.03 us | 163,840 B | 44.6 GB/s | 22.3 GB/s | 89.1 GB/s |
| 18.36 us | 327,680 B | 53.5 GB/s | 26.8 GB/s | 107.1 GB/s |

Two derived observations that matter for porting:

1. The marginal cost from M8 to M16 is +4.37 us for +245,760 B of per-rank egress, about 56 GB/s
   marginal - consistent with saturating a 2-link NVLink (NV2 = 50 GB/s per direction on this
   AC922). Below that the collective is latency-bound, not link-bound.
2. Extrapolating that marginal rate back to zero payload leaves a fixed cost of about 2.3 us per
   collective (launch plus poll plus reduce). That fixed cost is the thing a port must attack; the
   payload-dependent part is already near link speed.

Caveat: TP4 on this machine necessarily includes a SYS (cross-island) pair unless it uses two
cards from one island plus two from the other, and the docs do not state which GPU set produced
6.66 us / 11.03 us / 18.36 us. The NV2 vs SYS split of those measurements is [UNVERIFIED].

## A8. CUDA Graph interactions and conflict points

### A8.1 The push path is graph-only by construction [CODE]

Both dispatch sites require an active stream capture:

    custom_all_reduce.cuh:1944-1947
      if (sm70_tp4_push_buffers_registered_ &&
          status == cudaStreamCaptureStatusActive &&
          world_size_ == kSm70Tp4PushAllreduceWorldSize && fully_connected_ &&
          custom_all_reduce_current_device_is_sm70()) { ... }

    custom_all_reduce.cuh:2224-2228   (allreduce_sum2, same guard)

Rationale given in the comment at :1941-1943: "The push protocol amortizes peer polling across
captured collective chains. A lone eager call stays on the ordinary registered-buffer pull path."
The build artifact is literally named for this: "graph-only push"
(docs/design/sm70_dflash2_target_graph_20ms.md:649). The correctness gate exercises "eager,
one-call Graph, and 128-call Graph-chain modes" (same doc, :386).

CONSEQUENCE FOR llama.cpp: llama.cpp does not use CUDA Graphs. Its per-token graph is enqueued and
run eagerly through ggml_backend_graph_compute_async. The 1cat push path as written would NEVER
FIRE in llama.cpp. A port must drop the capture guard. See Part C.

### A8.2 Epoch state is persistent device memory, not per-capture state [CODE]

local_epochs[blockIdx.x] is a uint32_t in the registered IPC buffer
(custom_all_reduce.cuh:772-773 and :833) and is advanced on every launch. During stream capture the
kernel does not execute, so capture itself does not perturb the counters, and every replay advances
them by one. This is correct as long as (a) the number of blocks per launch never exceeds the 80
epoch words and (b) each rank issues the SAME sequence of collectives. The dedicated mixed-size
gate exists precisely because changing block counts and interleaving plain allreduce with
allreduce_sum2 on the same storage is the fragile case:
"Odd count, changing CTA counts, followed by another graph: per-block epochs must stay valid across
all these transitions" (benchmarks/kernels/benchmark_sm70_tp4_small_message_push.py:78-86).
That script sets the flag with os.environ[FLAG] and calls envs.disable_envs_cache() BETWEEN
captures (:68-69) - a direct statement that the launch policy is re-read at capture time and must
stay consistent with the epoch geometry of the buffer.

### A8.3 The push buffer deliberately avoids the graph-buffer IPC machinery [CODE]

The ordinary pull path must record every new input address during capture and exchange IPC handles
afterwards (custom_all_reduce.cuh:1922-1936, :1735-1753, :1863-1894 and
vllm/distributed/device_communicators/custom_all_reduce.py:385-418). That path needs
cudaIpcGetMemHandle to work, which forces the allocator to disable expandable_segments for the whole
capture window (custom_all_reduce.py:51-76). The push buffer is allocated once at init
(custom_all_reduce.py:341-348) outside any capture, so it is immune to that constraint.
Side effect worth knowing: allreduce() still pushes input into graph_unreg_buffers_ BEFORE the
push-path branch (custom_all_reduce.cuh:1922-1936 runs ahead of :1940), so the push path still burns
a RankData slot and still makes the peer IPC handles be gathered for the input address. Not a
correctness bug, only avoidable capture-time host work; a clean port should reorder it.

### A8.4 Exact rank-synchrony requirement

Because the protocol is a barrier and only double buffered, every rank must issue the same
collectives in the same order and must not skip one. A rank whose slice is zero-sized must STILL
participate, otherwise the peers spin forever. In llama.cpp the zeroing of inactive shards is done
by a separate scale kernel (llama.cpp/ggml/src/ggml-backend-meta.cpp:2321-2343) and the allreduce
still runs for that rank (:2453), so the contract is satisfiable - but a port must not let the
kernel early-exit before it signals.

### A8.5 Host-side determinism

The launch policy reads std::getenv at kernel-launch time (custom_all_reduce.cuh:148-191). Inside a
capture that host code runs at capture time, so the captured graph freezes whatever the environment
said then; changing the environment between capture and replay silently changes only the eager path.
That is exactly why the mixed-size benchmark must call envs.disable_envs_cache() between arms.

## A9. Non-conflicts worth stating explicitly

- There is NO conflict between push allreduce and CUDA-Graph capture legality. The kernel is
  launched on the capture stream with fixed arguments (persistent RankData passed by value as a
  kernel parameter, custom_all_reduce.cuh:1950-1952), which is what capture requires. The famous
  rejection is a numerical-quality rejection, see A6.
- VLLM_SM70_USE_BREAKABLE_CUDAGRAPH is a separate, independently rejected knob (-12.7% to -29.2%,
  sm70_qwen38_default_fastpath.md:109) and is NOT part of the push allreduce family.

## A10. Not verified in this session

- The results/*.json and .artifacts/runtime/*.json files referenced by the numbers in A7.1 are
  ABSENT from this checkout. Those numbers are reproduced from design-doc prose, not from raw
  artifacts.
- No GPU run was performed. No NVLink utilization counter was read.
- The exact GPU set behind 6.66 / 11.03 / 18.36 us is not stated in the docs.
- Whether SMALL_MESSAGES really reduces the M8 grid to 40 blocks in a live deployment depends on the
  envs.py:5037-5048 publishing running before the first capture. I read the code, I did not run it.

---

# Part B - SM70 FP16 GEMV kernels

There are TWO DISTINCT FP16 GEMV efforts in this tree. Do not conflate them.

| | GLM-5.3 exact KDA GEMV | DeepSeek-V4 FP16 GEMV |
|---|---|---|
| Implementation | CUDA C++, csrc/sm70_turbomind/ops/glm53_fp16_gemv_sm70.cu, 527 lines, 4 variants | Triton, vllm/models/deepseek_v4/sm70/gemv.py:26-42, _BLOCK_K = 1024, _NUM_WARPS = 4 |
| Shapes | [M,4096] x [6416,4096] (glm53_fp16_gemv_sm70.cu:16-17 and :396-405) | five shapes: router 256x4096, indexer 64x4096, C4 indexer 512x4096, C4 main 2048x4096, C128 1024x4096 (docs/design/sm70_deepseek_v4_fp16_gemv.md:8-14) |
| Design doc | none dedicated. Prose and numbers live in docs/design/sm70_v100_migration_control.md:44612-44654 and :46042-46049 | docs/design/sm70_deepseek_v4_fp16_gemv.md |

The task brief paired the CUDA file with the DeepSeek-V4 doc. They are different kernels; the doc
does not describe the four CUDA variants at all. This is a framing correction, not a detail.

## B1. Shared problem geometry [CODE]

glm53_fp16_gemv_sm70.cu:16-21:

    kGlm53K       = 4096        K (reduction length)
    kGlm53N       = 6416        N (output rows)
    kLanesPerRow  = 16          lanes cooperating on one output row
    kChunkK       = 512         K slice owned by one (row, chunk)
    kChunkCount   = 4096/512 = 8
    kThreads      = 16 * 8 = 128   (base kernel only)

Every variant accumulates FP16 products into an FP32 accumulator with __fmaf_rn and every variant
deliberately preserves a specific reduction order. The base kernel states it at :32-33:
"Each (chunk, lane) pair preserves the cuBLAS 32-element ascending FMA chain. Shared memory then
joins chunk 0..7 followed by lane 0..15." The arithmetic contract is spelled out in
docs/design/sm70_v100_migration_control.md:44620-44626: 16 lanes per output, eight 512-element
chunks, 32 ascending FP32 FMAs per (chunk,lane), then chunk 0..7 and lane 0..15 FP32 additions.
This is a bitwise-reproducibility constraint, not a performance one, and it is the reason there are
four variants instead of one.

No variant declares __launch_bounds__, __maxnreg__, maxrregcount or cudaFuncSetAttribute; every
launch passes dynamic smem 0 (third launch argument) at :427, :436, :445, :454, :463, :494, and all
shared memory is static __shared__. PER-THREAD REGISTER ALLOCATION IS [UNVERIFIED]. The only
register statement anywhere in the tree is the design doc note that a forced 32-register
launch-bound variant regressed and was removed (sm70_v100_migration_control.md:46046-46047).

## B2. The four variants

### V0 base = glm53_fp16_gemv_sm70_kernel<kBatch>, :23-80

| Property | Value | Line |
|---|---|---|
| Grid / block | <<<kGlm53N = 6416, kThreads = 128>>> - one CTA per output row | :30, :494 |
| Thread mapping | chunk = tid/16, lane = tid & 15 | :28-29 |
| Loads | scalar half via __ldg for both weight and input | :38, :41 |
| Inner loop | kChunkK/kLanesPerRow = 32 iterations, #pragma unroll 4 | :35-36 |
| Accumulators | float chunk_sum[kBatch] | :34 |
| Shared memory | float chunk_partials[kBatch][128] = 512 B at M=1, 4096 B at M=8 | :47 |
| Reduce | BOTH: smem cross-chunk serial join over 8 chunks, then half-warp __shfl_sync(0x0000ffff, ..., 16); only chunk == 0 participates | :50-79 |

This is the exactness reference and the only variant covering batch 1..8.

### V1 half2 = glm53_fp16_gemv_half2_sm70_kernel<kBatch, kRowsPerBlock>, :88-188

| Property | Value | Line |
|---|---|---|
| Threads | kThreadsPerRow = kChunkCount*kLanePairs = 64, kThreads = kRowsPerBlock*64; static_assert kThreads <= 1024 and kGlm53N % kRowsPerBlock == 0 | :92-96 |
| Possible blocks | rows in {1,2,4,8,16} -> 64/128/256/512/1024 threads | :461-482 |
| Grid | <<<kGlm53N/rows, rows*64>>> | :461-466 |
| Accumulators | float2 chunk_sum[kBatch] = 2*kBatch (16 at M=8) | :104 |
| Weight load | one 32-bit ld.global.cg.u32 via inline-PTX helper glm53_load_weight_half2 | :82-86, :108-109 |
| Input load | per-thread __ldg of a half2, so every row group loads x itself | :113-114 |
| Inner loop | 32 iterations, #pragma unroll 1 | :105-106 |
| Shared memory | tail_partials[rows][batch][64] plus lane_partials[rows][batch][16] = 10,240 B at rows=4, M=8 | :123-125 |
| Reduce | BOTH plus a serial tail: chunks 0..3 joined by full-warp __shfl_sync, chunks 4..7 staged to smem, chunk 0 accumulates them, then a SERIAL 16-term smem add by row_thread == 0 | :127-187 |

The head/tail split exists because one output row spans 64 threads = 2 warps, so a warp-only
reduction cannot cover all 8 chunks.

### V2 half2_broadcast = glm53_fp16_gemv_half2_broadcast_sm70_kernel<kBatch, kRowsPerBlock>, :190-276

Fixed at kRowsPerBlock == 4 and kThreads == 256 by two static_asserts (:196-197).
New mapping: ONE WARP OWNS ONE K CHUNK FOR FOUR OUTPUT ROWS - chunk = tid/32 (:202),
row_in_block = warp_lane/8, lane_pair = warp_lane%8 (:203-204). Launch
<<<kGlm53N/4 = 1604, 256>>> (:453-454).

Shared memory: chunk_partials[4][kBatch][8][16] = 16,384 B at M=8 (:231-232).
Reduction: smem plus __syncthreads (:235-240), chunk 0 sums the 8 chunk partials from smem into
even/odd per lane_pair (:242-255), then a width-8 __shfl_sync over the 8 source pairs (:259-269).

### V3 half2_broadcast_staged = glm53_fp16_gemv_half2_broadcast_staged_sm70_kernel<kBatch, kRowsPerBlock, kSwizzled, kBroadcastInput>, :287-382

Same geometry and asserts as V2 (:292-295); same launches (:427, :436, :445).
Same register layout as V2: float2 chunk_sum[kBatch] (:303) and a 32-iteration unroll-1 K loop
(:304-330).

Two real changes versus V2:

(a) STAGED = the batch loop moves OUTSIDE the reduction and a __syncthreads() separates batches
    (:334-338 write partials, :339 sync, :378-380 sync before the next batch). That lets the shared
    array drop the kBatch dimension: float chunk_partials[4 * 8 * 16] = 512 floats = 2,048 B,
    CONSTANT in kBatch (:332), versus 16,384 B at M=8 for V2. An 8x static-smem cut at M=8, paid
    for with 2*kBatch-1 __syncthreads instead of 2. What is staged is neither weights nor
    activations - it is the PER-THREAD PER-BATCH PARTIAL SUMS.
(b) SWIZZLE: glm53_staged_partial_index (:278-285) selects at compile time between
        kSwizzled true : ((chunk * kRowsPerBlock + row) * kLanesPerRow + lane)   // chunk-major
        kSwizzled false: ((row * kChunkCount + chunk) * kLanesPerRow + lane)     // row-major
    and kBroadcastInput is a template switch (:313-322): false makes every lane __ldg its own x;
    true applies the V2 broadcast.

The "what" of the swizzle is [CODE]. The "why" (bank conflicts on the reducing warp reads) is
[DERIVED] from the index arithmetic and the access pattern at :341-358; the source states no
rationale for the layout.

## B3. What "broadcast" adds, and why the production default turns it OFF

Broadcast is over the ACTIVATION VECTOR x - not the weight, not the block output. The shuffled
value is input_packed loaded from input (:216-222); weights are still loaded per-thread per-row
(:211-213); outputs are written per (row,batch) by lane_pair == 0 (:270-273). Code rationale at
:199-200: "One warp owns one K chunk for four output rows. The first eight lanes load each input
half2 once, then broadcast it to the other three rows."

Versus V1: in V1 every thread of every row group issues its own __ldg of the same x half2
(:113-114), so the x fetch is replicated kRowsPerBlock times per block. Broadcast issues one __ldg
per warp plus one shuffle, cutting x load requests to 1/4. x is 4096 halves = 8 KiB per token and is
reused by all 6416 output rows, so redundant per-block x fetches waste L1 and LSU issue capacity
while DRAM is dominated by the weights (6416 x 4096 x 2 B = 52.56 MB per call).

BUT THE PROJECT REMOVED IT. The recorded rationale is the -5 change:
"The retained exact KDA candidate lets every row lane directly load the cached input half2 and
removes the row-0 warp shuffle while preserving the FP32 FMA and reduction order."
(docs/design/sm70_v100_migration_control.md:46042-46044), measured 126.222 -> 114.924 us,
+9.8% effective bandwidth (:46044-46045). So the FINAL POSITION IS THAT THE BROADCAST SHUFFLE WAS
NOT WORTH KEEPING in the staged kernel. An isolated measured broadcast-on versus broadcast-off A/B
at fixed geometry is not recorded anywhere: [UNVERIFIED] as an isolated effect.

## B4. Dispatch, batch shapes and the effective default

[CODE] sm70_glm53_fp16_gemv_out, :413-511:

- Hard SM70 gate: properties->major == 7 and properties->minor == 0 (:418-419).
- Shape gate in the validator (:396-405): input [M,4096] with 1 <= M <= 8; weight [6416,4096];
  output [M,6416]; all FP16, contiguous, same device (:387-395, :406-408).
- Variant read at :422-423: VLLM_SM70_GLM53_EXACT_KDA_HALF2_ROWS, and it is consulted ONLY when
  input.size(0) == 8 and variant != 0 (:424).

| variant | Kernel | Grid x block | Line |
|---|---|---|---|
| -5 (C++ fallback default) | staged<8, 4, kSwizzled=true, kBroadcastInput=false> | 1604 x 256 | :425-433 |
| -3 | staged<8, 4, true, true> (broadcast ON) | 1604 x 256 | :434-442 |
| -2 | staged<8, 4, false> (row-major, broadcast ON) | 1604 x 256 | :443-451 |
| -4 | half2_broadcast<8, 4> (V2) | 1604 x 256 | :452-460 |
| 1,2,4,8,16 | half2<8, rows> (V1) | <<<6416/rows, rows*64>>> | :461-482 |
| 0, or any M in 1..7 | base<M> (V0) | <<<6416, 128>>> | :424, :492-508 |
| invalid value at M == 8 | TORCH_CHECK failure | - | :483-487 |

So V1, V2 and V3 are reachable ONLY at M == 8. This is an M=8 verifier-width specialization.

Two upstream gates decide whether the op is reached at all [CODE]:
vllm/models/glm5next/nvidia/kda.py:372-376 requires the fused fg/b decode path and
hidden_size == 4096; kda.py:520-528 additionally requires 1 <= num_tokens <= 8, FP16 hidden states
and weight, contiguity, and weight.shape == (6416, 4096). A missing op raises RuntimeError rather
than falling back silently (kda.py:530-534). The env gate is VLLM_SM70_GLM53_EXACT_KDA_GEMV, default
1 (kda.py:185-186).

LATENT DEFAULT MISMATCH WORTH FLAGGING. vllm/envs.py:270 declares
VLLM_SM70_GLM53_EXACT_KDA_HALF2_ROWS: int = -3, and envs.py:2364-2365 documents the -3 variant as
"a four-row CTA with swizzled shared partials". But (i) the C++ fallback when the variable is absent
is -5 (glm53_fp16_gemv_sm70.cu:423), and (ii) unlike the two allreduce switches this variable is
NOT published into os.environ - the publishing loop at envs.py:5037-5048 covers only
..._SMALL_MESSAGES and ..._CONCURRENCY, and the writer helper get_env_or_set_default at
envs.py:1092-1110 is not used for this key. The only writers in the tree are
benchmarks/kernels/benchmark_glm53_fp16_gemv.py:331 and the tests at
tests/models/glm5next/test_sm70_kda.py:247, 249, 258, 260.
THEREFORE THE EFFECTIVE PRODUCTION DEFAULT IS -5 (staged, swizzled, broadcast OFF), NOT THE -3
DECLARED IN envs.py. This is [CODE]-inferred; I did not run a deployment to confirm it.

## B5. Performance numbers

### B5.1 The four CUDA variants - recorded measurements exist, but in prose, and artifacts are absent

[MEASURED-in-doc] docs/design/sm70_v100_migration_control.md:44627-44634, for the BASE kernel at
[M=1, N=6416, K=4096], five post-cleanup runs:

- base kernel 65.485-65.509 us; paired cuBLAS 77.532-77.657 us; 15.55%-15.68% latency reduction;
  802.6-802.9 GB/s effective traffic bandwidth.
- bitwise equal to cuBLAS with zero maximum error across those runs (:44629-44630).
- Evidence files cuda_final_seed0..4.json and checkpoint_layer0_tp4_exact_audit.json under
  /data/minimax-h3/task-cache/glm53-nvfp4-sm70-20260827/cublas_match_kda_20260828/ (:44641-44645).
  Those paths are OUTSIDE this checkout.
- Nsight Compute counters were unavailable (ERR_NVGPUCTRPERM), so bytes divided by synchronized
  CUDA Graph time is an explicit bandwidth LOWER BOUND (:44645-44647).

Rejected in the same section (:44648-44650): one-pass row-grouped CUDA variants reached only about
78.7 us, and the global split-partial workspace route was about 130 us. The accepted kernel
"theoretically removes about 0.403 ms/token over 34 KDA calls; this is a projection, not an
end-to-end result" (:44650-44651).

[MEASURED-in-doc] The retained -5 variant, docs/design/sm70_v100_migration_control.md:46042-46049:

- bitwise equal to the -3 form, and moves 126.222 to 114.924 us (+9.8% effective bandwidth).
- rows-8 CTA and forced 32-register launch-bound variants regress to 132.407 and 131.568 us and were
  removed.
- the full-model output remains exact, but endpoint movement is below run-to-run noise, so only the
  operator result is claimed.
- the SHAPE for those microsecond values is not stated in that bullet: [UNVERIFIED].
- NCU and clock control were unavailable there too; the V100 stays at 877/1290 instead of 877/1530
  (:46066-46070).

Correctness measurements: focused native suite 6 cases; 5 random seeds require eager and CUDA Graph
output to be bitwise equal to torch.nn.functional.linear; a checkpoint audit fuses real layer-0 BF16
weights across all four TP ranks and finds 68 comparisons all bitwise equal with zero max error
(:44635-44640). In-tree equivalents: tests/models/glm5next/test_sm70_kda.py:189-226 (M in
{1,2,4,8}, 2 seeds, graph replay) and :237-263 (variant -2 versus -3 bitwise equal).

Whole-model endpoint with the exact KDA route enabled, GLM-5.3 TP4/PP2, B1, no MTP, 1024-in/256-out:
53.013085 / 53.018516 / 53.017527 tok/s, mean 53.016376 tok/s, mean TPOT 18.862097 ms, range
0.01024%, identical output hash (:44673-44679). That is an endpoint number, not a kernel number.

NO NUMBER ANYWHERE records a speedup of these four variants versus TurboMind or torch.matmul for
the 6416x4096 shape. The only recorded comparator is cuBLAS gemv2T at 77.5 us (:44630-44631).
And no stored benchmark output exists: benchmark_glm53_fp16_gemv.py computes latency_us and
effective_gbps itself (:221-229, :337-345) and only prints or writes JSON when --out is given
(:708-721); its CUDA variant sweep is (-5,-4,-3,-2,4,0,1,2,8,16) at :633. Globs for *gemv*.json,
*kda*.json and *cublas_match* under 1cat-vllm return nothing.

### B5.2 The DeepSeek-V4 Triton kernel - do not attribute these to the CUDA variants

[MEASURED] docs/design/sm70_deepseek_v4_fp16_gemv.md:37-46. Real checkpoint weights, CUDA Graph,
500 replays, five repeats; BLOCK_K=1024 and num_warps=4 selected for every accepted shape:

| Role | Shape N x K | cuBLAS median | Candidate median | Speedup | Projected saving |
|---|---|---:|---:|---:|---:|
| MoE router | 256 x 4096 | 10.047 us | 5.624 us | 1.79x | 0.190 ms/token |
| Indexer weights | 64 x 4096 | 6.834 us | 4.418 us | 1.55x | 0.051 ms/token |
| C4 indexer compressor | 512 x 4096 | 8.772 us | 5.415 us | 1.62x | 0.070 ms/token |
| C4 main compressor | 2048 x 4096 | 26.792 us | 22.737 us | 1.18x | 0.085 ms/token |
| C128 main compressor | 1024 x 4096 | 15.901 us | 12.597 us | 1.26x | 0.066 ms/token |

Summed operator projection 0.463 ms/token; the three C4 ops in a joined multi-stream graph improve
from 42.004 to 31.908 us, projecting 0.212 ms/token across 21 C4 layers before stream overlap
(:48-51). Sixteen real-weight router seeds preserve top-6 IDs, max normalized routed-weight error
3.61e-5, compressor max absolute error at most 2.39e-6 (:53-55).
The route is default-off behind VLLM_SM70_DSV4_FP16_GEMV=1 (:57-59; declaration
vllm/envs.py:228 with default 0 at :2143-2144), and stays default-off after a full-graph transfer run
measured 20.276 ms/token low-overhead and 22.708 ms/token node-traced, with 2.392 ms/token
attributed to 125.9 fixed-shape GEMV launches, because that is composition evidence rather than a
route-on/route-off A/B (:68-83). A later fused-router prototype measured 0.5201 / 0.4551 / 0.3861 ms
over 40 layers, best saving 0.66% of the 20.276 ms baseline, below the 0.2 ms/token admission
threshold, and was not integrated (:103-112).

The original trace motivating the whole effort: 3.235 ms/token for dense GEMV plus split-K reducers
plus compressor work at 1024-input TP8 decode, with per-shape traced main-kernel means of 9.73 us
(router, 43 calls), 20.07 us (indexer, 21), 16.02 us (C4 indexer, 21), 38.90 us (C4 main, 21) and
22.15 us (C128, 20) (:5-14).

## B6. Honest summary of why the variants were needed

- half2 exists to halve load-issue count; V0 issues one scalar __ldg per operand per element. V0 is
  kept because it is the batch 1..7 path and the arithmetic reference.
- broadcast exists because at M=8 four output rows share one activation row, so the x load is 4x
  redundant inside a warp. Trading one __shfl_sync for three fewer __ldg only wins when the load is
  the bottleneck - and at kBatch = 8 the activation row is already L1-hot, so the project measured
  the shuffle as a loss and turned it OFF in the production default (-5).
- staged exists to cut static shared memory from 16 KiB to 2 KiB per CTA (8x at kBatch=8) by moving
  the batch loop outside the reduction, and to allow a chunk-major swizzle so the reducing warp
  reads consecutive addresses. NO occupancy / L2-reuse / register-pressure rationale is recorded;
  the only statement is the envs.py:2364-2365 comment naming the four-row CTA and the swizzle.
- Net structural idea: one warp owns one K chunk for several output rows, so the weight stream is
  the only high-volume traffic, activation traffic is amortized, and the cross-chunk reduction is a
  short warp shuffle plus a small smem join. That is a weight-bandwidth-bound design that never uses
  tensor cores - the right call on V100 for M <= 8, where a tensor-core tile would be dominated by
  the K-loop load anyway.

## B7. Mapping onto llama.cpp

The equivalent hotspot in llama.cpp is mul_mat_vec_q in llama.cpp/ggml/src/ggml-cuda/mmvq.cu, the
quantized mat-vec used for every MUL_MAT with a small batch. Structure [CODE]:

| Aspect | llama.cpp mul_mat_vec_q | 1cat GLM53 gemv |
|---|---|---|
| Block shape | dim3(warp_size, nwarps, 1) (mmvq.cu:1023) | 1D, 128 or 256 threads |
| Rows per CTA | calc_rows_per_block (mmvq.cu:596-614): 1 for ncols_dst 1, 2 for 2..8 | 1 (V0) or 4 (V2/V3) |
| Batch dimension | ncols_dst, a template parameter switched 1..8 (mmvq.cu:1192-1283); equals src1 ne[1] (mmvq.cu:1538, ids ? ne2 : ne1); MMVQ_MAX_BATCH_SIZE 8 (mmvq.cuh:3) | kBatch template, M in 1..8, but the half2 family only at M == 8 |
| ne12 role | channel dim -> nchannels_y -> blockIdx.y (mmvq.cu:1539, :1022) | not applicable |
| Warp-to-work mapping | each warp handles one token independently (mmvq.cu:866) - the OPPOSITE assignment | each warp owns one K chunk and serves 4 rows |
| Per-thread accumulators | float tmp[ncols_dst][rows_per_cuda_block] (mmvq.cu:726-727); the same weight block vx is reused for every column j (mmvq.cu:758-771) | float2 chunk_sum[kBatch] |
| K handling | kbx = tid/(qi/vdr), strided by blocks_per_iter = vdr*nwarps*warp_size/qi (mmvq.cu:643, :732) | fixed 32-iteration per-thread serial chain |
| Reduction | warps 1..nwarps-1 write tmp_shared (mmvq.cu:774-790), __syncthreads (:791), warp 0 adds and does warp_reduce_sum (:804-812) | smem staging plus 16-lane or 8-pair __shfl_sync |
| Volta table | MMVQ_PARAMETERS_VOLTA, mmvq.cu:98 and :468-484: ncols_dst 1 -> nwarps 2, 2..4 -> 4, 5..8 -> 2; comment records nwarps 8/4/2/1 = 32.05/36.23/37.52/37.24 tok/s on AC922 (:469) | none, fixed 256 threads |
| Small-K handling | should_use_small_k raises rows_per_block to nwarps when blocks_per_row_x is small (mmvq.cu:1123-1127) | not present |
| Grid | 3D: (ceil(nrows_x/rpb), nchannels_dst, nsamples) (mmvq.cu:1016-1025) | 1D: 6416 or 1604 blocks |

THE TRANSFERABLE IDEA is the row-ownership axis. llama.cpp gives each warp a WHOLE TOKEN
(mmvq.cu:866), so at ncols_dst = 8 the same weight row is walked 8 times by 8 different warps with
no reuse inside a warp. 1cat inverts it: one warp walks one K chunk for 4 rows, so the weight stream
is read once for 4 rows within the warp and the activation is what gets duplicated or broadcast. On
a weight-bandwidth-bound V100 decode that is the right axis, and it is the same axis this fork
already tunes via MMVQ_PARAMETERS_VOLTA nwarps.

What does NOT transfer: the bitwise-cuBLAS-order contract (glm53_fp16_gemv_sm70.cu:32-33, enforced
by tests/models/glm5next/test_sm70_kda.py:208-210) is a 1cat acceptance requirement, not a llama.cpp
one; llama.cpp vec_dot order is free. And the FP16 weight path itself is not applicable to a Q8_0 or
Q4_K_M production route, where weights are block-quantized and dequantized inside the dot product.

The fourth structural difference: llama.cpp weights are block-quantized and dequantized inside the
vec_dot (int8 dp4a with per-block scales, ggml/src/ggml-cuda/vecdotq.cuh:246-258 and :851-866, with
VDR_Q8_0_Q8_1_MMVQ 2 at vecdotq.cuh:243, selected at mmvq.cu:77) while 1cat reads raw FP16 and does
FP32 FMA. The dp4a path proper lives in MMQ: MMQ_DP4A_MAX_BATCH_SIZE 64 (mmq.cuh:8), eligibility at
mmq.cu:323-335 (for NVIDIA: not fp16_mma_hardware_available(cc) or ne11 < 64), configs at
mmq.cuh:255-256 and :281-282 pointing at mmq-config-pascal-dp4a.cuh, dp4a vec dots in
mmq-vec-dot.cuh, tile tables at mmq.cuh:387-421, and the instrinsic wrapper ggml_cuda_dp4a at
common.cuh:720-758 (__dp4a at :750).

Production gating this fork already encodes and any GEMV work must respect:
llama.cpp/ggml/src/ggml-cuda/mmvq.cuh:5-7 sets MMVQ_VOLTA_MAX_BATCH_SIZE_K = 4 with the comment
"Measured on V100: MMQ beats MMVQ at ne11=8 (+2.6%) and ne11=6 (+1.1%), MMVQ wins at ne11=4
(-1.2%)", and mmvq.cu:374-383 routes Q2_K/Q3_K/Q4_K to MMVQ only for ne11 <= 4.

---

# Part C - Minimal push-based multi-GPU allreduce for llama.cpp

## C1. What llama.cpp has today [CODE]

### Communication stack, llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu

- ggml_backend_cuda_comm_context holds one try_allreduce function pointer chosen at init
  (:968-995). Three implementations exist: ..._nccl (:1000-1071), ..._internal (:1076-1121), and
  ..._butterfly, a stub returning false (:1141-1144).
- Init chain: comm_init_nccl -> on failure comm_init_internal -> on failure comm_init_none
  (:1157-1205). Env selector GGML_CUDA_ALLREDUCE = nccl | internal | none; the platform default on
  Linux is NCCL, elsewhere internal (:1225-1245).
- comm_init_nccl refuses NCCL under CUDA virtual devices (:1177-1184) and otherwise calls
  ncclCommInitAll (:1188).
- NCCL heuristics: FP32 when ne < 32768 for 2 backends, < 131072 for 3, < 262144 for 4 or more;
  otherwise an FP32 to BF16 to FP32 round trip (:1017-1068).
- The internal pipeline is HARD-LIMITED TO 2 DEVICES: ggml_cuda_ar_pipeline_init returns nullptr for
  n_devices != 2 with the message at :402-406, and ggml_backend_cuda_comm_allreduce_internal asserts
  n_backends == 2 (:1081). It also requires cc >= VOLTA (:410-418).
- Critically, the internal path is a HOST-MEMORY PCIe DESIGN, not NVLink. The file header at
  allreduce.cu:13-38 says it "targets setups without NVLink/xGMI, where data is exchanged between
  the GPUs by staging it through pinned host memory over PCIe".
- Its resources: a 2-deep ring of 1 MiB pinned staging per device (allreduce.cu:234, :238,
  :504-505), 32 MiB copy-engine staging plus 32 MiB device scratch per device (:242, :519-533), a
  cache-line-padded arrival-token ring (:72, :484-499), and 8 x 256-thread blocks for the chunked
  kernel (:77, :925).
- P2P is enabled only when GGML_CUDA_P2P is set, or implicitly by an NCCL build
  (ggml-cuda.cu:391-402 and :605-609).

### Insertion point, llama.cpp/ggml/src/ggml-backend-meta.cpp

- The meta backend splits the graph at GGML_BACKEND_SPLIT_AXIS_PARTIAL tensors (:2192-2221) into
  n_subgraphs, and runs one allreduce after each subgraph except the last (:2434-2462). On failure
  it falls back to a host-side butterfly of ggml_backend_tensor_copy_async plus GGML_OP_ADD
  (:2318-2431), sized by backend_ctx->max_tmp_size, the largest subgraph-boundary tensor (:2190,
  :2229-2234).
- comm_allreduce is resolved from the backend registry at :1834-1837 and dispatched at :2453.
- Consequence: about one allreduce PER LAYER BOUNDARY, on the layer output hidden state, in the
  tensor own dtype (F32 in this fork graph) - not a small verifier payload. [DERIVED] from the split
  rule; the exact count for the 65-block model was not measured here.

## C2. Why the 1cat design does not drop in

Three blockers, all [CODE]-verified:

1. GRAPH-ONLY GUARD. custom_all_reduce.cuh:1944-1947 and :2224-2228 require an active stream
   capture. llama.cpp never captures; it enqueues eagerly per token.
2. PAYLOAD SHAPE AND COUNT. 1cat pushes an 80 KiB FP16 verifier payload on a fixed schedule inside
   one captured graph. The llama.cpp boundary tensor is [n_tokens, 5120] F32 = 160 KiB at n_tokens=8
   and would be reduced tens of times per token across the 65-layer graph.
3. TRANSPORT. 1cat relies on fully_connected (all pairs NVLink). On this AC922 a TP4 group that
   includes a cross-island pair is NOT fully connected (nvidia-smi topo -m: NV2 within {0,1,2} and
   {3,4,5}, SYS across). A direct push to a SYS peer would work but at PCIe and X-Bus bandwidth and
   latency. That does not kill the idea, it means the topology must be chosen deliberately.

## C3. Minimal design - "P0": an N-device push fast path inside the internal pipeline

The smallest change that captures the 1cat win without importing its machinery.

SCOPE: extend the existing internal allreduce from 2 devices to N devices with a push protocol,
keeping current 2-device behaviour bit-identical.

Why this shape:

- The internal pipeline already owns the right resources (persistent per-device buffers, a
  non-blocking stream per device, an event pool, a token-based arrival ring) and already has the
  correct semantics: an in-place sum over matching tensors, with inactive shards contributing zeros
  (allreduce.cu:791-804 and :917-922). Reuse it; do not build a parallel subsystem.
- The 2-device chunked kernel is ALREADY A PUSH: it writes to pinned host memory, fences, then polls
  the peer arrival token (allreduce.cu:88-108 and :62-67). The change is to generalize one peer into
  N-1 peers and give each peer its own staging slot instead of a single host_other.
- Keeping GGML_CUDA_ALLREDUCE=internal as the opt-in means zero behaviour change for anyone who does
  not set it.

Concrete steps, in order:

1. LIFT THE 2-DEVICE GATE. allreduce.cu:402-406 -> accept n_devices <= GGML_CUDA_MAX_DEVICES. Keep
   the cc >= VOLTA check (:410-418).
2. GIVE EACH RANK N STAGING SLOTS. Replace host_buf[i] and host_large[i] with host_buf[i][dst]
   indexed by destination rank. Sizing: POOL_SIZE(2) * n_devices * GGML_CUDA_AR_MAX_BYTES(1 MiB)
   per device for the chunked path; the current allocation at allreduce.cu:504-505 is only
   POOL_SIZE * 1 MiB. At TP4 that is 8 MiB of pinned host memory per device instead of 2 MiB, which
   is trivial. Keep the 32 MiB copy-engine path as a 2-DEVICE-ONLY special case: it hard-asserts
   n_devices == 2 at allreduce.cu:607. Do not generalize it in P0.
3. GENERALIZE THE ARRIVAL RING. ggml_cuda_ar_arrival_ptr already indexes
   (slot * n_devices + rank) * KERNEL_BLOCKS * ARRIVAL_STRIDE (allreduce.cu:342-346), so the ring is
   already N-device shaped. What is missing is the per-destination dimension for a push, and the
   cheapest correct answer is P0a: every rank writes THE SAME payload to all peers and then waits
   for all N arrival tokens. This is exactly the 1cat algorithm with the transport swapped from
   device IPC to pinned host memory. Each rank then reads back N-1 peer copies from host and sums
   them locally in rank order, which is bitwise deterministic across ranks - the same property 1cat
   relies on (custom_all_reduce.cuh:730-731). There is no cheaper option without a ring, and a ring
   reintroduces 2*(n-1) latency steps. CHOOSE P0a.
4. MOVE THE REDUCE TO FP32 IN REGISTER. Follow sm70_push_reduce
   (custom_all_reduce.cuh:700-708): accumulate peers in FP32 in a FIXED rank order and downcast
   once. This preserves the guarantee already promised for 2 devices at allreduce.cu:96-100 ("both
   GPUs truncate identically - this guarantees bit-equivalent results across the two devices").
5. DOUBLE-BUFFER WITH THE EXISTING 2-DEEP SLOT RING. GGML_CUDA_AR_POOL_SIZE = 2 already exists
   (allreduce.cu:234); the current code uses one slot at a time and waits for pool wraparound with
   cudaEventSynchronize (:364-377). Do not change this in P0.
6. NO CUDA-GRAPH GUARD. llama.cpp is eager, so the entire cudaStreamCaptureStatusActive condition
   from 1cat is simply deleted. The 1cat epoch-counter trick is also unnecessary here: with the
   arrival-token protocol (allreduce.cu:43-50) the token IS the epoch, it increases monotonically per
   call, and it never needs resetting. That is strictly simpler than the 2-epoch modular counter at
   custom_all_reduce.cuh:833.
7. KEEP THE BARRIER SEMANTICS. The collective must remain a barrier for all N ranks, including
   ranks whose slice is zero-sized. Today the zeroing is done by a separate scale kernel for
   inactive shards (ggml-backend-meta.cpp:2321-2343) and the allreduce still runs for them (:2453),
   so this is already satisfied - but the kernel must not early-exit before it signals.

EXPLICITLY OUT OF SCOPE FOR P0: device-memory IPC transport, CUDA-Graph capture, replacing the NCCL
path, fused reduce-plus-activation kernels, and any change to the meta backend subgraph splitting.

## C4. Landing points

| Change | File | Anchor |
|---|---|---|
| Accept N devices | llama.cpp/ggml/src/ggml-cuda/allreduce.cu | :402-406 |
| Per-destination staging slots | same | :314-315, :504-513 |
| N-way arrival wait | same | :94-95, :342-346 |
| FP32 ordered N-way reduce | same | :110-210 and ggml_cuda_ar_add_kernel at :159-210 |
| Remove the n_backends == 2 assert | llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu | :1081 |
| Meta-backend dispatch already correct | llama.cpp/ggml/src/ggml-backend-meta.cpp | :2443-2461 |
| Update the file header comment | llama.cpp/ggml/src/ggml-cuda/allreduce.cu | :13-38 |

Total surface: one init function, one kernel, one assertion. That satisfies the workspace rule
"simple beats complete" and the llama.cpp rule "prefer reusing existing infrastructure".

## C5. Risks and open questions

1. BANDWIDTH CEILING. The 1cat numbers imply about 2.3 us of unavoidable fixed cost per collective
   plus about 56 GB/s of marginal per-rank egress [DERIVED]. A host-memory P0 design will be far
   slower than device IPC. Whether P0 can beat NCCL at the layer-boundary payload (160 KiB F32 at
   n_tokens = 8) is NOT KNOWN and must be measured with --tensor-split 1/1/1 on CARDS=0,1,2 and
   GGML_CUDA_ALLREDUCE=internal against the current best (L=/root/libdir-nccl P2P=1).
2. TOPOLOGY. A device-IPC variant (P1) requires all-pairs connectivity, which TP4 must violate on
   this AC922 unless the cards are chosen to fit one island. The honest P1 target is either a
   fully-NVLink subset or a hierarchical two-level push, exactly like the 1cat TP8 variant at
   custom_all_reduce.cuh:917-938, which is the same idea at two levels.
3. PAYLOAD COUNT PER TOKEN. llama.cpp issues one allreduce per subgraph boundary. If that is about
   65 per token, an 11 us per-collective cost is roughly 0.7 ms/token, material but not dominant in
   the 58.9 ms/round budget. THE COUNT MUST BE MEASURED BEFORE OPTIMIZING THIS; it is cheap to get
   from existing logging or a counter.
4. NCCL IS CURRENTLY THE BEST MEASURED CONFIGURATION (workspace record: +18.4% from compiling NCCL
   in). Any P0 work must be justified as a REPLACEMENT for a known-good baseline, not as an
   addition. If P0 cannot beat NCCL at the real payload size, the correct outcome is to delete it.
5. ZERO-SIZED SHARDS. Confirmed handled by the meta backend today, but the N-way kernel must not
   deadlock when one rank has nothing to send. Use the existing "inactive contributes zeros"
   contract (allreduce.cu:917-922).

## C6. Suggested measurement plan, no code change required to start

1. Count the allreduces per decode graph and log the payload bytes. The internal path already has
   GGML_LOG_DEBUG statements (allreduce.cu:1088, :1107, :1113); the NCCL path prints nothing.
2. Get the NCCL per-call cost at the real payload by adding a temporary counter around
   ggml_backend_cuda_comm_allreduce_tensor (ggml-cuda.cu:1252-1258), or by driving the same shape
   through test-backend-ops.
3. Only then decide between P0 (host staging, N devices) and a device-IPC push (P1).

---

# Bottom line

1. The 1cat push allreduce is a ONE-SHOT PUSH-BROADCAST WITH LOCAL FP32 RANK-ORDERED REDUCE, over
   CUDA IPC on P2P-mapped device memory, with a 2-epoch double buffer of about 2.50 MiB per rank and
   an FP16 NaN (0x7f7f) empty-slot sentinel. It is not an allgather+reduce and not a ring.
2. Measured: about 6.64-6.68 us per 80 KiB collective versus about 14.5 us for the pull path, and
   11.03 versus 18.45 us at 160 KiB - a 31-40% cut. It is the accepted production transport for
   Qwen3.8 and the DFlash2 target graph. It is REJECTED ON QUALITY GROUNDS (GSM8K 59/60 -> 58/60) for
   SM70 GLM5 DFlash2 TP4 and is auto-disabled there by vllm/config/vllm.py:365-399.
3. The only CUDA-Graph issue is that the push path is GRAPH-ONLY BY CONSTRUCTION: both dispatch
   sites require cudaStreamCaptureStatusActive. The famous "all-fast graph + PUSH_ALLREDUCE=1
   rejected" entry is an accuracy rejection, NOT a capture failure. Since llama.cpp never captures, a
   port must DELETE that guard, which is easy.
4. The FP16 GEMV variants exist to serve M=8 verifier width on a weight-bandwidth-bound V100: half2
   halves load issue, broadcast removes a 4x redundant activation load, staged cuts static smem from
   16 KiB to 2 KiB with a batch-major loop plus a chunk-major swizzle. The production default is
   variant -5: STAGED WITH BROADCAST OFF.
5. Measured for the CUDA gemv: base kernel 65.485-65.509 us versus cuBLAS 77.532-77.657 us (15.55 to
   15.68% faster, 802.6-802.9 GB/s lower bound) at [M=1,N=6416,K=4096]; the retained -5 moves
   126.222 to 114.924 us. The DSV4 numbers (1.18x-1.79x over cuBLAS) belong to the separate TRITON
   kernel, not to these four variants.
6. The minimal llama.cpp landing is to GENERALIZE ggml_cuda_ar_pipeline_init (allreduce.cu:402) from
   2 to N devices and make ggml_cuda_ar_kernel write to N-1 peer staging slots and wait on N-1
   arrival tokens, reusing the existing pinned staging, token ring, streams and events. Roughly one
   init function, one kernel and one assertion (ggml-cuda.cu:1081). Do NOT port the device-IPC
   transport, the CUDA-Graph guard, or the epoch machinery. MEASURE AGAINST THE CURRENT NCCL BEST
   BEFORE WRITING ANY OF IT.
