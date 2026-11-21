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

## P2: token-row capacities

A strictly increasing list of at most 16 positive `--m-buckets` chooses the
smallest capacity B >= actual M. An empty list keeps exact specialization.
M above the largest configured bucket falls back to exact compilation; it is
never truncated. The normal signed-32-bit tensor-byte bounds still apply to B.
For example, `[1,4,8]` maps M=3 to B=4, M=5/7 to B=8 and M=9 to B=9.

The pass first admits the entire actual FFN graph, proving that dot reductions
use feature dimensions and all other operations are scalar broadcasts or
elementwise operations. It changes only X and row-carrying SSA result types to
B rows, then runs MLIR verification and full FFN admission again. Weight shapes,
D, F, operation order and numeric policy stay unchanged. A well-typed graph
that mixes token rows is rejected. This proof does not apply to attention or
to padding reduction/feature dimensions.

The execution boundary is:

```text
X[M,D] -- zero-pad token rows --> X[B,D]
    -- compiled FFN[B,D,F] --> Y[B,D] -- crop first M rows --> Y[M,D]
```

`specialize_rows(...)` retains:
- `before.mlir`: admitted logical-M core, after any P1 conversion.
- `core.mlir`: canonical B-row program actually compiled and cached.
- `boundary.mlir`: one standard StableHLO function containing Pad/Core/Slice,
  evaluated by the official interpreter as an equivalence witness.
- `plan.json`: explicit row counts, pad value, crop bounds and proof scope.

`FFNModule::RunRows` implements these boundary operations on the host.
`Run` retains its original exact-shape checks. Each padded call initializes
a fresh zero tail, uploads the full B-row input and downloads the full B-row
output before cropping. This is not a device-transfer reduction. Boundary
time and padded/cropped rows are counted separately from VM invocation time.

The artifact ABI describes the **compiled capacity**, not the current M.
No new computational ABI is necessary: an exact B-row artifact is the same
mathematical program as the bucket core. Canonical keys include B, D, F,
physical weight ABI and core IR. Request indexes also include the complete
bucket policy and logical GGML snapshot. Different policies selecting the same
core can share its canonical package; different M values sharing B reuse
one loaded JIT module. Explicit AOT package paths are separate module keys,
so a different listed package cannot bypass its own validation.

Enable both passes for JIT:

```bash
python compiler/jit_worker.py configure --cache build/ffn-cache \
  --output build/jit-p1-p2.json --weight-layout transposed --m-buckets 1 4 8
```

For AOT, pass the same `--m-buckets` option to `compile_ffn.py`; use the selected
capacity with the standalone runner's `--capacity-m B`. Its tensor JSON and
returned output still have the logical M. A bucket-aware GGML catalog is
explicitly versioned; each row is `logical-M D F directory layout capacity-M`:

```text
coral-m5-ffn-v2
1 64 172 shared-capacity-8 transposed 8
5 64 172 shared-capacity-8 transposed 8
```

This coarse example tests sharing, not a recommended decode policy. Padding
M=1 to B=8 can increase execution work substantially. Exact specialization is
the default. `[1,4,8]` keeps decode exact while allowing prefill sharing.
GGML's existing member capability, scheduler and fusion guards still apply;
a bucket policy does not add support for arbitrary graph operations or shapes.
