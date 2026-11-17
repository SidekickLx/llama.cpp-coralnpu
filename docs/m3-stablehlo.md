# M3: standard StableHLO infrastructure

The project builds against the exact StableHLO, MLIR and LLVM revisions used by
the pinned Coral compiler. No independent pip/JAX MLIR runtime is introduced.
StableHLO's revision is also recorded in `dependencies.lock.json`.

## Build and use

Complete the README setup first, then run from the project root:

~~~bash
source env.sh
bash scripts/build_m3.sh
build/m3/coral-stablehlo-tool emit-ffn 2 3 4 build/m3/ffn-example.mlir
build/m3/coral-stablehlo-tool verify build/m3/ffn-example.mlir
python tests/compiler/verify_stablehlo.py --tool build/m3/coral-stablehlo-tool --output-root build/m3-verifier-results
~~~

The build script injects the project as a Bazel local repository into the
existing Coral compiler workspace. This uses the upstream dependency mapping
and cache without modifying its MODULE.bazel or adding a second LLVM build.
Set `M3_BUILD_JOBS` to adjust build concurrency (default 4). Build outputs remain
cached. Emission refuses to overwrite an existing module.

## Code and responsibilities

- `compiler/ir/stablehlo_program.{h,cc}`: reusable typed construction, loading,
  writing and verification of standard MLIR modules.
- `compiler/ir/stablehlo_tool.cc`: a small command-line integration tool.
- `BuildMatmul`: `X[M,K] * W[N,K]^T -> Y[M,N]`, entry point `matmul`.
- `BuildFFN`: the contract's four F32 inputs and canonical split SwiGLU math,
  entry point `ffn`. It checks the metadata contract version, numeric policy,
  positive static dimensions and element-count overflow before construction.

All mathematical nodes are registered upstream StableHLO operations built with
MLIR OpBuilder and generated typed operation classes. Dot contracting dimensions
are `1 x 1`. SiLU uses negate, exponential, scalar-one broadcast, add, divide and
multiply. The `coral.ffn_abi` and `coral.numeric_policy` attributes are descriptive
metadata on the function; they do not define custom computation or automatically
set a downstream compiler's fast-math options.

`VerifyModule` calls the official `mlir::verify`: MLIR validates SSA and function
structure, while StableHLO validates operation types, shapes, dimensions and
broadcast rules. Valid generic StableHLO is not proof that a graph meets this
project's FFN admission contract. GGML provenance, aliases, external consumers,
backend ownership and runtime buffer lifetime remain separate planner checks.
The importer and matcher are not implemented by these standalone builders.

The verifier acceptance suite contains five valid generated modules, seven
invalid MLIR/StableHLO modules, three invalid shape specifications, and an
overwrite rejection check. It keeps the actual upstream diagnostics as evidence.

## Builder/verifier acceptance evidence

All cases passed. Evidence is retained locally at
`build/m3-verifier-results/verify-ni3on7iy`, including generated MLIR,
per-case stdout/stderr and `report.json`. A repeat build completed with no C++
compilation actions, confirming reuse of the existing compiler cache.

The pinned StableHLO verifier permits some mixed-type dot configurations.
Accordingly, generic verifier success must never replace the project's F32-only
FFN admission checks. The type rejection test uses an invalid mixed-type add;
the builder itself explicitly constructs every FFN input/intermediate as F32.

## Reference interpreter

`compiler/ir/reference.{h,cc}` wraps the official
`mlir::stablehlo::evalModule` API. It verifies the module, checks the selected
function and typed inputs, clones the module before interpreter preprocessing,
and returns typed dense results. No project numerical evaluator or fallback
kernel is registered. The scalar Python arithmetic in the tests is only an
independent oracle.

The current diagnostic admission set is: constant, dot_general,
broadcast_in_dim, negate, exponential, add, divide, multiply and func.return.
Functions must be single-block with static F32 tensors. Dot uses the contract's
unbatched `1 x 1` contraction, with no algorithm override. This is our current
tested subset, not the full list of operations supported by upstream StableHLO.
Unknown operations are rejected before interpreter dispatch. The wrapper caps
total input/intermediate elements at 1,048,576, operation count at 1,024, and dot
multiply-accumulate terms at 10,000,000. These diagnostic resource limits are
separate from device capability and the mathematical FFN ABI.

Run the full numerical/rejection suite:

~~~bash
python tests/compiler/verify_reference.py --tool build/m3/coral-stablehlo-tool --reference build/m3/coral-stablehlo-reference --output-root build/m3-reference-results
~~~

To evaluate a module yourself:

~~~bash
build/m3/coral-stablehlo-reference MODULE.mlir ffn INPUTS.json NEW_OUTPUT.json
~~~

The input file uses the four positional arguments from the contract, in order:
X, gate weight, up weight, down weight. Each tensor has `dtype: "f32"`,
`shape: [rows, columns]`, and a flat row-major `values` array:

~~~json
{"inputs": [
  {"dtype": "f32", "shape": [1, 2], "values": [1.0, -1.0]},
  {"dtype": "f32", "shape": [2, 2], "values": [1.0, 0.0, 0.0, 1.0]},
  {"dtype": "f32", "shape": [2, 2], "values": [0.5, 0.0, 0.0, 0.5]},
  {"dtype": "f32", "shape": [2, 2], "values": [1.0, 0.0, 0.0, 1.0]}
]}
~~~

Generate the matching module with `emit-ffn 1 2 2 NEW_MODULE.mlir`.
The result file contains typed output tensors and `evaluation_ms`. Wrong entry
names, input counts/types/shapes, non-finite inputs/outputs, unsupported ops, and
resource overruns are errors. Existing output files are never overwritten.

## Numerical acceptance and timing

Evidence: `build/m3-reference-results/reference-s1lv60se`. Generated input JSON,
output JSON, production/debug MLIR, stdout/stderr and `report.json` are retained
locally and ignored by Git.

| Program / shape | Maximum absolute error | Reference evaluation time |
| --- | --- | --- |
| Matmul M=2, K=3, N=4 | 0 | 1.191 ms |
| FFN M=1, D=3, F=4 | 0 | 2.603 ms |
| FFN M=2, D=3, F=4 | 0 | 3.919 ms |
| FFN M=1, D=64, F=172 | 0 | 865.493 ms |
| FFN M=5, D=64, F=172 | 0 | 4212.880 ms |

All five named FFN tensors (gate, up, SiLU, hidden, output) matched the scalar
F32 oracle for these fixed fixtures; the acceptance rule remains
`abs(actual-reference) <= 1e-4 + 1e-4*abs(reference)`. This observed equality is
not a general bit-exactness promise. Ten rejection cases passed, including a
mixed-type dot which generic StableHLO accepts but this F32 reference path rejects.

Times are single-run host measurements of validation, cloning and official
reference evaluation, excluding JSON parsing/serialization and process startup.
The build uses the upstream development configuration. These numbers describe
a correctness interpreter, not Coral inference performance or accelerator speed.

The standalone builder remains a fixture generator. M3 step 4 now provides a
separate [per-operation GGML importer](m3-importer.md), validated against actual
GGML CPU execution, including captured real-model nodes.
[FFN matching and legality checks](m3-fusion.md) now use upstream GGML interfaces.
Complete member support, real same-split placement and full real-region
acceptance remain subsequent steps; full FFN Coral VMFB execution belongs to M4.
