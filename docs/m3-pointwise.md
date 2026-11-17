# M3: standalone FFN members

The typed StableHLO builder exports SiLU, equal-shape multiply and split SwiGLU.
The split SwiGLU argument order is gate, then up. All arguments and results are
static dense row-major F32 matrices. There is no broadcast, packed GLU or
reduced-precision support. The versioned function reflection includes operation
and exact dimensions; it is validated before VM context initialization.

`PointwiseModule` shares VMFB loading, VM/HAL ownership, output validation and
Coral execution checks with the M1 matmul runtime. Host numerical executable
loads remain forbidden. Every successful call requires nonzero simulated
kernel instructions and complete successful Coral dispatch observations.
Arguments are copied each call; staging buffers are reused. Invalid bindings
are rejected before execution, and execution failures poison the context.

## Build and verification

Reuse the existing compiler, simulator and CMake build described in the README.
All preparation happens offline; inference never invokes a compiler.

~~~bash
source env.sh
bash scripts/build_m3.sh
cmake --build build/m1 --target coral-pointwise-test -j4
python scripts/prepare_pointwise.py \
  --tool build/m3/coral-stablehlo-tool \
  --compiler repos/coralnpu-compiler/bazel-bin/external/iree_core+/tools/iree-compile
# Use the artifact directory printed by the preparation script.
python tests/runtime/verify_pointwise.py \
  --runner build/m1/coral-pointwise-test \
  --artifacts build/m3-pointwise-artifacts/pointwise-EXAMPLE
~~~

The default shapes are 1x172 and 5x172, matching decode and prefill in the
cached tiny Llama model. An explicit `--shapes` list supports other static
profiles. Compilation disables Coral fast math. `pointwise.tsv` is a separate
member catalog, so existing M2 `profiles.tsv` files keep their original ABI.

## Standalone acceptance evidence

Evidence: `build/m3-pointwise-results/pointwise-t64s9x1w/report.json`.
Artifacts and compiler placement logs:
`build/m3-pointwise-artifacts/pointwise-xc76cmwd`.

- Six independently compiled operation/shape profiles; two contexts per profile.
- 48 successful calls and 48 Coral dispatches; 106,224 simulated kernel instructions.
- Maximum absolute error against actual GGML CPU: `9.53674316406e-7`.
- Comparison tolerance: `2e-6 + 1e-5 * abs(reference)`; finite signed inputs.
- 48 invalid-binding checks, eight failing-process checks, and changed inputs
  across repeated calls; buffer reuse and upload volume verified.
- Zero host executable loads. Runtime subprocesses run without a compiler in PATH.
- Aggregate VM invoke time: 27.299 ms across 48 calls, including first calls.
  This is a single simulator diagnostic run, not a hardware latency or speedup claim.
- Existing CTest suite: 5/5 passed after rebuilding the shared runtime and adapter.
  Log: `build/m3-pointwise-ctest.log`.

This milestone proves standalone member execution. It does not yet prove
backend placement of a complete FFN or a fused FFN executable.
