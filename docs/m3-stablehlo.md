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
