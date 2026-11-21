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
