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

The standalone milestone was committed as `fbd95ae`. The backend integration
and placement acceptance below build on those independently verified programs.

## Backend integration and real placement

The backend loads the optional `pointwise.tsv` catalog during registration.
The supported operation, exact dimensions, source slots, parameter words,
dtype and all strides must match. Views, broadcasts, packed GLU and unknown
parameters are rejected. Queries consult only the in-memory catalog. A missing
profile stays on CPU; a selected but invalid VMFB fails the invocation.
All accepted members remain independently executable, even when fusion is
rejected. No graph node is skipped.

The default runner uses `--observation nodes`. Its public llama evaluation
callback observes CPU and Coral nodes and intentionally prevents fusion
admission. `--observation backend` installs no evaluation callback. It records
completed Coral nodes directly, including an ID for the actual `graph_compute`
range, plus preparation-plan metadata from our upstream `graph_optimize` hook.
CPU observation counts are `null` in that mode, rather than fabricated zeros.
The removed original-graph observer and scheduler patch are not used.

A plan log alone is not execution evidence: reservation can produce plans, and
graph reuse can execute without another planning callback. The model test
matches each executed FFN activation to a complete admitted pattern and checks
that every member actually executed on Coral in the same compute range. This
is placement evidence, not authorization to invoke a fused executable.

~~~bash
cmake --build build/m1 --target coral-member-adapter-test coral-llama-runner -j4
# Set these to the retained Quick start and pointwise preparation directories.
MATMULS=build/m2-artifacts/model-aot-c2u8iuxg
POINTWISE=build/m3-pointwise-artifacts/pointwise-xc76cmwd
python tests/ggml/verify_member_adapter.py \
  --runner build/m1/coral-member-adapter-test \
  --matmuls "$MATMULS" --pointwise "$POINTWISE"
python tests/ggml/verify_ffn_members.py \
  --runner build/m1/coral-llama-runner \
  --model models/stories260K-f32.gguf \
  --matmuls "$MATMULS" --pointwise "$POINTWISE"
~~~

The model verifier combines both catalogs into its own retained artifact
directory and runs fixed-token replay. It checks the locked, clean llama.cpp
revision and the CMake source directory. Runtime subprocesses have no compiler
in PATH.

## Backend acceptance report

Final evidence:

- `build/m3-member-adapter-results/adapter-hbfgdqyw/report.json`: 138 checks,
  including 111 capability cases, 18 numerical calls and a preflight failure
  preserving output; ten invalid catalog cases. Capability queries succeed
  after artifact files have been renamed, proving they do no deferred I/O.
- `build/m3-member-results/members-5blaf1n9/report.json`: real model results,
  binary/model hashes and the 20 executed FFN member groups.
- `build/m3-member-results/members-5blaf1n9/backend/trace.jsonl`: complete plans
  and individual node execution evidence with compute-range IDs.
- `build/m3-member-results/members-5blaf1n9/backend/steps.jsonl`: per-step counts
  and measured timing.
- `build/m3-unpatched-results/unpatched-njxf0b_v/report.json`: original M2,
  synthetic importer, real-model importer and 33 planner/lifetime cases pass
  after member integration. `build/m3-step6-ctest.log`: 5/5 existing CTests pass.

The five-layer tiny Llama model runs a five-token prefill and three one-token
decode steps. Each step contains five complete FFN member groups: 20 total.
Across the full run, 144 matmuls and 20 split SwiGLU calls produce 276 successful
Coral dispatches and 1,496,248 simulated kernel instructions. Every dispatch
has an instruction sample; host executable loads and runtime failures are zero.

Maximum CPU/Coral logits error is `8.58306884765625e-6` in both observation modes,
within `1e-4 + 1e-4 * abs(reference)`. The backend mode records 30 preparation
plans including reservations; this count is not the 20 actually executed groups.
The node-observation mode rejects all 30 candidates under the callback policy
while executing the same legal individual members.

Removing the SwiGLU profiles restores its CPU placement and removes complete
FFN admission without breaking logits. Corrupting a selected member VMFB causes
failure without a successful completion marker. The new backend tests exercise
all three pointwise operations; the real model uses split SwiGLU.

## Measured simulator timing

One final run, with no statistical sampling or speedup claim:

| Phase / step | Input tokens | Wall ms | VM invoke ms | Dispatches | Module loads |
| --- | --- | --- | --- | --- | --- |
| prefill 0 | 5 | 63.936 | 61.174 | 45 | 9 |
| decode 1 | 1 | 29.068 | 28.102 | 77 | 2 |
| decode 2 | 1 | 26.651 | 25.996 | 77 | 0 |
| decode 3 | 1 | 26.553 | 25.941 | 77 | 0 |

Wall time includes llama/GGML scheduling, CPU operations, simulated Coral work,
copies and diagnostic output. VM invoke time covers simulator execution and
HAL orchestration, not hardware latency. Warm decode reuses all modules and
model weights, with zero additional staging-buffer allocations. All build
outputs and caches are retained.

M3 step 6 is complete. The next step is full real-region GGML-to-StableHLO
import, verifier and official CPU reference acceptance. This implementation
does not compile or execute a fused full-FFN VMFB; that remains M4.
