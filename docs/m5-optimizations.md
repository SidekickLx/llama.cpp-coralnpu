# M5 FFN optimization contracts

P1 and P2 transform **standard StableHLO** after the imported SSA graph has
passed the FFN semantic admission check. There is no project computational
dialect or numerical interpreter. The official StableHLO interpreter remains
the semantic oracle, with bounded inputs and operation counts.

## P1: prepared weight layout

The default ABI remains `coral.ffn.f32.v1`:
`X[M,D], gate[F,D], up[F,D], down[D,F] -> Y[M,D]`.

The opt-in ABI `coral.ffn.f32.weights-transposed.v1` accepts
`X[M,D], gateT[D,F], upT[D,F], downT[F,D] -> Y[M,D]`.
Both use dense row-major F32. The runtime preparation API **always** accepts
the original GGML weight layout; it transposes each role once and uploads an
owning immutable snapshot. Prepared weights are compatible across M values
only when D, F, physical ABI and device ownership match.

The original importer has **no transpose operations**: it expresses native
weights using dot contracting dimensions `[1] x [1]`. P1 changes the three
argument types and RHS contracting dimensions to `[1] x [0]`. It preserves
the actual imported operations, operand roles, reduction order and numeric
policy. MLIR verification and full FFN admission run again after the rewrite.
An identity F32-to-F32 `stablehlo.convert` may be removed; a narrowing/widening
pair, unsupported operation or different mathematical graph is rejected.
Fewer frontend nodes alone are not evidence of faster device execution.

`optimize_region(...)` retains `before.mlir`, `after.mlir`, logs and `plan.json`
in its evidence directory. The plan records each role, permutation and source/
prepared shape. The artifact's checked contract and VM reflection bind the
physical ABI; an old runtime rejects an unknown ABI before execution. The
outer artifact envelope stays v1 because its fields have not changed.

Select P1 for AOT with:

```bash
python compiler/compile_ffn.py --input imported.mlir --m 5 --d 64 --f 172 \
  --weight-layout transposed --output build/ffn-p1
build/m1/coral-ffn-runner --artifact build/ffn-p1 \
  --weight-layout transposed --inputs inputs.json --output build/p1-run.json
```

Input tensor JSON remains in the original GGML layout. For a GGML AOT catalog,
append `transposed` to each relevant `ffn.tsv` row; omitted layout means native:

```text
coral-m4-ffn-v1
5 64 172 ffn-prefill transposed
```

For JIT, add `--weight-layout transposed` to `jit_worker.py configure`.
The option is frozen in the session configuration and request identity. The
canonical artifact key includes the physical ABI and rewritten IR, so native
and transposed artifacts cannot collide. Regenerate configuration after changing
compiler/frontend/service binaries or source files.

### Weight changes and lifetime

Default mutable inputs prepare a fresh snapshot on each call. Enabling
`immutable_model_weights` promises stable contents and storage lifetime.
Same-shaped layers have separate role-ordered tensor/buffer/data identities;
layout is also part of the prepared-weight key.

Call `ggml_backend::InvalidateModelWeights()` between synchronous graph calls
**before** modifying or replacing immutable weights. On their next execution,
all affected backend contexts rebuild their runtime caches and device. This
also discards MPACT's lower-level read-only binding cache, including stale
entries whose host addresses could otherwise be reused. Recompilation is not
needed; validated code remains reusable from disk. The backend remains
single-threaded and a failed graph cannot be revived by invalidation.
