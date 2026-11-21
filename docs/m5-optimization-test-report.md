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
