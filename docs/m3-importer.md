# M3: direct GGML-to-StableHLO import

The compiler imports borrowed, live `ggml_tensor` metadata directly. It does
not parse the diagnostic graph JSON or introduce a project computational IR.
The caller keeps the GGML snapshot alive and unchanged throughout import.
No tensor payload, backend, compiler subprocess, or device runtime is accessed
by the importer.

## Interfaces and ownership

`compiler/ir/ggml_importer.h` provides three layers:

- `ImportGGMLTensorType`: validates static F32, unbatched canonical dense layout
  and maps GGML `[columns, rows, 1, 1]` into StableHLO `[rows, columns]`.
- `ImportGGMLNode`: validates one operator, its GGML sources, parameters, result
  shape, and supplied SSA operand types; emits standard typed StableHLO ops.
  Operands arrive in GGML `src[]` order. A rejected node emits no operations.
- `ImportGGMLFunction`: imports an explicit topological operation selection,
  with ordered boundary inputs and results. Dependencies must resolve to a
  boundary value or an earlier selected operation. Failure returns no module.

`GGMLImportRequest` holds references and identity metadata, not owned graph
nodes. Input/operation tensor IDs must be unique, nonzero and representable as
positive signed 64-bit MLIR integers. Outputs must reference the same source
identity. The caller supplies graph instance/revision. Function arguments and
results carry tensor IDs; every generated operation carries its source tensor
ID, operator spelling and complete integer parameter words.

The generic function has one argument per distinct boundary tensor. A producer
outside the selection is not recursively imported. Duplicate source edges
share the same SSA value. This function is not declared to implement the fixed
four-position FFN ABI: a future FFN binder must preserve semantic roles, including
identical/shared weights, when supplying operands to the per-node importer.

The importer rejects explicit views, noncanonical strides and backend-specific
extra storage. It does not prove absence of overlapping physical allocations,
external consumers, intermediate observation, or cross-backend splits.
Those are future planner checks. Successful import does not authorize fusion
or establish Coral support.

## Initial MUL_MAT rule

GGML `src[0] = weight[N,K]` and `src[1] = activation[M,K]` become
`dot_general(activation, weight) -> [M,N]` with contracting dimensions `1 x 1`.
The importer validates all source slots and all 16 parameter words. Only
default/F32 accumulator (word 0) and source-1 precision (word 3) are admitted;
unknown fields, reduced precision and hints fail closed.

The standalone builders and importer share the standard dot/SiLU constructors
in `compiler/ir/stablehlo_ops.{h,cc}`. No alternate mathematical implementation
is maintained.

## Build and test

After the README setup, run from the project root:

~~~bash
source env.sh
bash scripts/build_m3.sh
cmake --build build/m1 --target coral-ggml-import-test coral-model-import-test -j4
python tests/compiler/verify_importer.py --runner build/m1/coral-ggml-import-test --check-library build/m3/libcoral-importer-check.so --output-root build/m3-importer-results
~~~

Tests build graphs with the pinned GGML constructors and execute them using
`ggml_graph_compute_with_ctx` on one CPU thread. The CMake test executable
loads a test-only Bazel shared library to pass the live tensors to the importer
and official StableHLO reference interpreter. This avoids linking a second
GGML implementation or rebuilding LLVM with CMake. The test bridge is not a
production backend/plugin API.

Negative metadata is mutated only after a valid GGML CPU execution; malformed
graphs never reach GGML kernels. Each case runs in a fresh process and retains
MLIR, input values, GGML outputs, reference outputs, stdout/stderr and a report.
Rejections retain diagnostics but publish no imported module. Tolerance is
`abs(actual-reference) <= 1e-4 + 1e-4*abs(reference)`. Reports include maximum
absolute/relative error and every failing index.

## MUL_MAT acceptance

The initial milestone passed 5 numerical cases and 37 rejection cases:
nonsquare matrices, decode/prefill dimensions, explicit F32 precision and shared
input identity; dtype/layout/batching/shape errors, missing/extra sources,
unresolved dependencies, invalid IDs, and every unknown parameter word.

Evidence is retained at `build/m3-importer-results/import-rqdyakuh`.
This first commit covered MUL_MAT; the following milestone adds the remaining
operators.

## Pointwise and split SwiGLU rules

| GGML source | Standard StableHLO lowering | Additional admission |
| --- | --- | --- |
| UNARY(SILU) | negate, exponential, scalar-one broadcast, add, divide | One source; exact SILU enum; all remaining parameter words zero |
| MUL | multiply | Two sources and output with identical shapes; all parameters zero |
| GLU(SWIGLU) | SiLU(src[0]) multiplied by src[1] | Two split inputs; swap=0; all remaining parameter words zero |

Inputs, results and SSA values must have the same validated static F32 shapes.
Broadcasting, packed one-input GLU, alternate activations, clamped/OAI variants,
swapped GLU parameters, in-place/view operations and unknown parameters are
rejected. A MUL may put the SiLU result on either side; GLU gate/up order is
semantically significant and never inferred from matching dimensions.

The second milestone passed 12 numerical cases and 99 rejection cases at
`build/m3-importer-results/import-nzyj3tx3`. SiLU covers negative and positive
saturation (-80 to +80), near-zero values and zero. Asymmetric GLU fixtures
distinguish reversed gate/up edges. Composed GGML FFNs exercise all three
matmuls, split GLU, and SILU+MUL with both MUL operand orders; named intermediates
and the final output are compared with actual GGML CPU execution.
All parameter words are tested for every admitted operator. Reordered and
self-referential dependencies are rejected.

These composed test graphs are explicitly selected fixtures. They do not
constitute an automatic FFN matcher, fusion plan, or Coral execution path.


## Real-model node acceptance

The model harness loads the locked `stories260K-f32.gguf` through llama.cpp
with one CPU thread, F32 KV storage and offloading disabled. It imports the first
MUL_MAT and first split SwiGLU node during both five-token prefill and one-token
decode. These are actual runtime nodes; the selected matmul is not claimed to
be an FFN projection, and the test does not discover or fuse an FFN region.

The evaluation callback requests a synchronization boundary at every node.
It copies inputs before execution and the output after execution into an
owned test snapshot, preserving the original types, dimensions, strides,
operator parameters and source-edge identities. This is necessary because
GGML can overwrite input storage or reuse buffers. The live graph is unchanged.
Snapshot-local diagnostic IDs are not IDs from the whole graph exporter.

Run from the project root after building the two test targets above:

~~~bash
python tests/compiler/verify_model_import.py \
    --runner build/m1/coral-model-import-test \
    --check-library build/m3/libcoral-importer-check.so \
    --model models/stories260K-f32.gguf \
    --output-root build/m3-model-import-results
~~~

All four real-model cases passed. Evidence is retained at
`build/m3-model-import-results/model-import-w28dn1nn`. Each capture includes
`imported.mlir`, inputs, GGML output, StableHLO reference output and a JSON report;
the enclosing directory retains runner stdout/stderr and an aggregate report.

| Captured node | Output shape | Maximum absolute error | Import + verify | Reference evaluation |
| --- | --- | --- | --- | --- |
| Prefill MUL_MAT | [5, 64] | 1.90735e-06 | 3.250 ms | 791.201 ms |
| Prefill split SwiGLU | [5, 172] | 9.53674e-07 | 3.444 ms | 128.062 ms |
| Decode MUL_MAT | [1, 64] | 2.86102e-06 | 2.244 ms | 164.159 ms |
| Decode split SwiGLU | [1, 172] | 1.19209e-07 | 3.253 ms | 26.907 ms |

Every element passes the same `atol=rtol=1e-4` rule, with GGML CPU output as
the comparison baseline. The maximum absolute error is `2.861023e-6` across
these real-model cases; the composed/operator fixture suite has a maximum
absolute error of `2.980232e-8`.

Times are single-run host measurements from the development build, without
warmup or distribution statistics. Import timing includes dialect setup,
metadata validation, SSA construction and verification. Reference timing
includes validation, cloning and the official interpreter. Both exclude tensor
capture, model execution, process startup and JSON/MLIR serialization.
They are diagnostic timings, not an inference throughput benchmark or Coral
accelerator performance measurement.

The earlier builder/verifier and reference suites also passed regression:
`build/m3-verifier-results/verify-nxaw5ba0` and
`build/m3-reference-results/reference-ue9hjflf`.

## Completion boundary

M3 step 4 is complete for the admitted static F32 subset. Unsupported cases
remain explicit errors. The importer provides semantic conversion and verified
standard SSA; it does not infer fusion legality or change runtime placement.
[FFN matching and legality checks](m3-fusion.md) are now implemented using
unmodified upstream interfaces. Complete member support and real same-split
placement are next. Full-region Coral compilation,
dispatch evidence and CPU/StableHLO/Coral three-way comparison belong to M4.
