# M5 optimization acceptance

## P1: prepared weight ABI

Evidence: `build/m5-p1-bxhl8mbv/report.json`. Reproduce with:

```bash
python tests/compiler/verify_ffn_layout.py \
  --fixtures build/m3-acceptance-results/acceptance-fe6qikxj/ffn-import/ffn-import-r4t1hl9f \
  --real-fixtures build/m3-model-ffn-results/model-ffn-1k4vkadq \
  --aot build/m4-aot-results/aot-cvoknmn_ \
  --matmuls build/m2-artifacts/model-aot-c2u8iuxg \
  --pointwise build/m3-pointwise-artifacts/pointwise-xc76cmwd
```

Three compiled profiles passed GGML CPU / official StableHLO interpreter /
Coral comparisons: `(2,3,4)`, `(1,64,172)` and `(5,64,172)`.
Official reference outputs before and after P1 were bit-identical. The largest
Coral-versus-CPU absolute error in these cases was `6.55651e-7`; all comparisons
used `atol=rtol=1e-4`. Each layout ran 10 calls in each of two device lifetimes.

Evidence includes actual before/after IR, a role-specific weight preparation
plan, compiler logs, executable sources, placement, VMFBs and numerical JSON.
An identity F32 conversion was removed; altered mathematics, a narrowing/
widening pair, incorrect shape, already-transformed input and an ABI mismatch
were rejected. Mutable and immutable scheduler fixtures passed, as did explicit
invalidation followed by **same-address weight mutation** and graph reuse.

The real stories260K model completed 84 member matmuls plus 20 complete FFNs
with the new AOT ABI, five weight preparations and 15 weight cache hits.
Maximum logit absolute difference from CPU was `1.14441e-5`. There were no
host executable loads, failed dispatches or observer errors.

| Profile | Layout | Dispatches/call | MPACT instructions/call | Warm call median ms | Weight preparation median ms |
|---|---|---:|---:|---:|---:|
| M=1,D=64,F=172 | Native | 6 | 39,737 | 2.994 | 0.061 |
| M=1,D=64,F=172 | Transposed | 3 | 6,188 | 1.140 | 0.135 |
| M=5,D=64,F=172 | Native | 3 | 132,296 | 9.735 | 0.053 |
| M=5,D=64,F=172 | Transposed | 3 | 142,346 | 10.763 | 0.125 |

P1 helps this decode profile but **regresses this prefill profile**. It stays
opt-in. The source already had zero transpose nodes; the benefit is the changed
downstream kernel/layout selection. Runtime logical transfer bytes did not
decrease. These are local MPACT simulator measurements, not hardware latency
or a universal speedup claim; preparation, compilation and loading are separate
costs. Timing is descriptive and has no pass/fail threshold.

The same transposed ABI passed the full JIT matrix in
`build/m5-jit-4r77a4ap/report.json`: two cold compiler invocations, two persistent
hits in a new process with zero child executions (strace), activation changes
without recompilation, corrupt artifact recovery, index repair, stale environment
rejection, worker failure/timeout including descendant termination, and all
scheduler boundary regressions. Run `tests/ggml/verify_ffn_jit.py` with the
existing catalog arguments plus `--weight-layout transposed` to reproduce.

## P2: token-row bucket specialization

Independent evidence: `build/m5-p2-gbeyani2/report.json`. The 20 cases cover
both weight layouts, M=1/2/3/4/5/7/8/9 with policy `[1,4,8]` at D=3,F=4,
and captured real-model M=1/5 at D=64,F=172 with B=8. All 20 Pad/Core/Crop
witnesses produced bit-identical official-reference results to their unpadded
programs. Coral-versus-CPU maximum absolute error was `8.34465e-7`.

Twenty requests produced ten canonical artifacts; the other ten reused them.
Same-capacity cores were byte-identical across logical M. A single live module
and weight snapshot also passed growing/shrinking rows `8,1,7,2,5,1` and a
subsequent changed-weight preparation, in both layouts. The suite includes
26 negative checks: invalid policies, truncation, a **well-typed row-mixing dot**
and runtime binding/layout/context violations. No attention or feature-axis
padding is admitted.

Real-model JIT evidence: `build/m5-jit-45qox56d/report.json`. With P1 and a
deliberately coarse `[8]` policy, 20 FFN calls share one compiled capacity:
one cold worker/compiler invocation, one fresh-process disk hit, zero warm
child executions, 19 loaded-module hits, five weight preparations and 15 weight
hits. Logit error versus CPU is `9.53674e-6`. Cache corruption/index repair,
stale environment, worker failure/timeout and scheduler guards all pass.

The trace has **four M=5 FFNs and sixteen M=1 FFNs**: llama.cpp prunes the
last prefill layer to the last token. Thus B=8 pads/crops 124 rows over these
20 calls. It uploads/downloads 31,744 additional logical bytes in each
direction versus exact execution. This policy reduces compilation diversity
but increases device work; it is a reuse stress case, not the default.

Reproduce the independent and full JIT gates with:

```bash
python tests/compiler/verify_ffn_buckets.py \
  --real-fixtures build/m3-model-ffn-results/model-ffn-1k4vkadq
python tests/ggml/verify_ffn_jit.py \
  --matmuls build/m2-artifacts/model-aot-c2u8iuxg \
  --pointwise build/m3-pointwise-artifacts/pointwise-xc76cmwd \
  --aot build/m4-aot-results/aot-cvoknmn_ \
  --weight-layout transposed --m-buckets 8
```

The versioned AOT catalog also passed `tests/ggml/verify_ffn_bucket_backend.py`:
`build/m5-p2-backend-6g14nlkn/report.json`. It reuses one B=8 module without
reading the deliberately nonexistent JIT configuration. Mutable/immutable/
generation changes, external consumers/views and evaluation callbacks pass.
Seven malformed catalogs are rejected. A different, corrupt package listed
for M=1 is still validated after four healthy M=5 FFNs; an existing same-capacity
module cannot conceal the error. No final inference report is published.

The public AOT CLI with `--m-buckets 1 8` and standalone runner with
`--capacity-m 8` also passed three invocations, retaining
`build/m5-p2-cli-artifact` and `build/m5-p2-cli-run.json`.

## Final regression and compiler-free deployment

The final implementation passed all default-path regressions in
`build/m5-final-regression-ugtnl5aj/checks.json`:

- Rebuilt all runtime/test targets; CTest **5/5**.
- Shared service, persistent cache invalidation/repair and failure retries.
- AOT admission: 14 normalized fixtures, 10 rejected inputs, six real compiled profiles.
- Runtime: six profiles (including M=16/32), 86 numerical calls and nine negative gates.
- GGML: 84 scheduled graphs executed three times each, 11 catalog rejections.
- Five paired real-model CPU/member/full-region runs.

The strict artifact boundary additionally passed **54 negative cases** in
`build/m5-final-contract-results/contract-ox_id1nz`.
The complete default native/exact JIT matrix passed in
`build/m5-jit-7pkfekk1/report.json`, including zero warm child executions.
The upstream llama.cpp checkout remains clean at the locked revision.

`build/m5-final-deployment-results/deployment-p0g9ne2d/report.json` records
**41 isolated processes** with no compiler/frontend/Python/shell in their
runtime root. All 16 executable absence probes returned ENOENT. Programs ran
as UID 65534, with zero capabilities and no active network interface.
Nine original FFN profiles, six optimized FFN cases, original model controls,
two optimized bucket-model processes and standalone matmul passed.
Source compilers and every pre-existing build/cache directory were preserved.
The optimized model reproduced its prior outputs exactly without a JIT config.

Reproduce the extended deployment gate by preparing as the workspace owner,
then verifying the newly printed deployment directory:

```bash
python tests/deployment/verify_aot_deployment.py prepare \
  --model-evidence build/m5-final-regression-ugtnl5aj/model/model-7pr9c5yi \
  --ffn-evidence build/m4-runner-results/three-way-p3m7_t6e \
  --aot-evidence build/m4-aot-results/aot-cvoknmn_ \
  --optimized-model-evidence build/m5-p2-backend-6g14nlkn \
  --optimized-ffn-evidence build/m5-p2-gbeyani2 \
  --output-root build/m5-final-deployment-results
sudo /usr/bin/python3 tests/deployment/verify_aot_deployment.py verify \
  --deployment PATH_PRINTED_BY_PREPARE
```

Root is used only to create the namespace/chroot and drop privileges. The
deployment gate does not remove host compilers or existing build outputs.

## Paired simulator performance

Evidence: `build/m5-performance-gnqqkxvb/report.json`.
Five rounds rotate the order of five policies, with one process running at a
time and fixed replay tokens. All use already-compiled AOT artifacts, the same
model and member catalog, 84 member matmuls and 20 complete FFNs per run.
Counters are deterministic across the five runs; timings are medians.
The report includes the SHA-256 of the final runner binary.
Process wall time includes startup, model loading and logging; VM and FFN
timings isolate narrower execution scopes. Pipe-based process waiting avoids
the 50 ms polling quantization seen in an earlier retained measurement.

| Policy | Process ms | All VM invoke ms | FFN invoke ms | Dispatches | Instructions | Padded rows |
|---|---:|---:|---:|---:|---:|---:|
| Native / exact | 165.42 | 142.16 | 94.50 | 256 | 1,504,628 | 0 |
| P1 / exact | 141.90 | 118.18 | 69.52 | 208 | 1,008,044 | 0 |
| Native / B=8 | 350.95 | 327.79 | 280.02 | 208 | 4,058,472 | 124 |
| P1 / B=8 | 309.49 | 287.00 | 239.00 | 208 | 3,147,732 | 124 |
| P1 / buckets [1,8] | 142.43 | 119.09 | 71.09 | 208 | 1,000,276 | 12 |

P1/exact reduces total FFN invocation time by **26.4%** on this replay.
Its M=1 kernels improve, while its M=5 kernels regress. A single B=8
specialization reduces the code/profile count but wastes decode work.
Keeping decode exact with `[1,8]` avoids that penalty; in this workload it
still takes slightly longer than P1/exact. Fewer instructions do not guarantee
less simulator time, and fewer frontend nodes are not an acceleration result.

Logical host/HAL upload/download bytes per replay are 1,088,512 / 48,128 for
exact policies, 1,120,256 / 79,872 for B=8, and 1,091,584 / 51,200 for `[1,8]`.
These counters do **not** measure MPACT DDR/TCM traffic, DMA overlap or physical
hardware bandwidth. Preparation is measured separately in the P1 table above.
All optimizations remain opt-in; no automatic profitability selector is added.

Reproduce this measurement with:

```bash
python tests/ggml/benchmark_ffn_optimizations.py \
  --native-catalog build/m4-region-model-results/model-69qio4fj/artifacts \
  --layout-catalog build/m5-p1-bxhl8mbv/catalog \
  --bucket-evidence build/m5-p2-gbeyani2 --repeats 5
```

## Cold compilation versus persistent hits

These strace runs are separate diagnostics, not a randomized compiler benchmark.
Initialization hashes the actual 1.3 GiB compiler library and tool closure once
per process; it still costs about nine seconds even when no compilation occurs.
This cost remains visible and is not included in the millisecond lookup column.

| Policy | Cold compiler invocations | Cold JIT resolution s | Warm lookup ms | Warm child executions |
|---|---:|---:|---:|---:|
| Native / exact | 2 | 28.578 | 10.493 | 0 |
| P1 / exact | 2 | 31.393 | 9.849 | 0 |
| P1 / B=8 | 1 | 15.031 | 5.107 | 0 |

M5 steps 4–6 are accepted within the documented F32, single-threaded,
static-FFN scope. Shape/dtype/target/policy/compiler changes, source semantics,
physical ABI and bucket capacities participate in validation or code identity.
Concurrent compilation deduplication, device staging reuse, quantization and
SoC integration remain later milestones.

The final API audit also rejects invalid logical shapes **before** selecting a
positive bucket, preventing M=0 or a negative M from being hidden by B=8.
Five checks passed in `build/m5-final-logical-shape-test.log`. A subsequent real
model cache hit (`build/m5-final-boundary-warm`) and the final 41-process
deployment run use the rebuilt runtime containing this guard.
