# M4: complete FFN artifacts and runtime

## Step 1: artifact contract

Status: contract and loader validation implemented. A valid manifest is not
proof that bytes are executable; step 3 separately checks VM reflection and execution.

The wire format is [ffn-manifest.schema.json](../artifacts/ffn-manifest.schema.json).
The C++ validator additionally enforces role order, exact shape relations,
signed 32-bit tensor byte bounds, reflection identity, the supported target
and build compatibility. JSON Schema alone cannot establish those relations.

A package contains:

| File | Purpose |
| --- | --- |
| `manifest.json` | Version, completion state, FFN ABI, program identity, toolchain, target, integrity and placement summary |
| `region.stablehlo.mlir` | Verified, normalized standard computation; weights remain arguments |
| `region.vmfb` | Compiled IREE module, with `coral.m4.ffn.abi` function reflection |
| `region.metadata.json` | Source correspondence and diagnostic provenance, outside semantic identity |
| `placement.json` | Hashed unique executable exports and scheduling placement summary |

Exactly four F32 dense row-major inputs are required, in activation/gate/up/down
order, followed by one output. The schema, program pipeline and numeric policy
are independently versioned. No M1/M2 module is implicitly an FFN artifact.

The reflection value binds ordered tensor roles, dimensions, numerical policy,
runtime compatibility digest, semantic program digest and pipeline version.
It excludes weights, activation values, graph IDs and machine paths. The
compiler embeds this value; the runtime must compare it before VM context
initialization. Artifact digests establish consistency, not publisher authenticity.

Compatibility is embedded at CMake configure time from the dependency lock,
target profile and pinned CRT/linker/runtime source hashes. Version 1 supports
Linux x86_64, MPACT, the locked RV32 ILP32 feature set and default 8 KiB ITCM /
32 KiB DTCM with 4 KiB stack and 4 KiB reserve. A different profile requires
an explicitly supported configuration, not a manifest edit. Compiler and actual
linker/CRT binary digests are also recorded for provenance.

The loader rejects unknown fields/versions, duplicate JSON keys, incorrect
numeric wire types, mismatched roles/shapes/target/options, incomplete packages,
inconsistent placement and bad SHA-256 values. Fixed filenames prevent manifest
path traversal. Member symlinks and nonregular files are rejected; file sizes
and JSON nesting are bounded. Validated VMFB bytes are retained as an owned
snapshot so execution never reopens an unchecked file.

`nlohmann/json` and the public-domain SHA-256 implementation come from the
locked llama.cpp vendor directory. They are compiled without linking GGML,
MLIR or the compiler. Their upstream licenses remain applicable.

~~~bash
source env.sh
cmake --build build/m1 --target coral-ffn-artifact-check -j4
python tests/artifacts/verify_ffn_artifact.py \
  --checker build/m1/coral-ffn-artifact-check
# Inspect the exact supported contract and target:
build/m1/coral-ffn-artifact-check --describe 1 64 172
~~~

Evidence: `build/m4-contract-results/contract-m52fub4u/report.json`.
One valid contract-only fixture and 54 negative cases pass. These fixture VMFB
bytes intentionally are not executable: this milestone tests integrity and
compatibility before the IREE loader, not device execution.

## Step 2: complete FFN AOT compilation

Status: implemented and verified for six static profiles. This compiles an
entire imported FFN into one VMFB; the compiler may generate several device
dispatches. It does not promise one machine kernel.

The input is the actual standard StableHLO emitted by the M3 GGML importer.
The pinned StableHLO verifier checks IR validity; the AOT admission check then
matches the full SSA computation against the positional FFN contract, including
argument roles, dot properties, constants, types and numeric policy. Valid IR
with incorrect FFN semantics is rejected. No custom dialect or numerical
interpreter is introduced.

After admission, operation diagnostic IDs are written to source metadata;
the original input (including graph and argument metadata) is retained as
source.mlir. Diagnostic metadata is removed from semantic IR.
Independent operation order and accepted commutative spellings are normalized.
The actual imported operations are compiled, not replaced by a shape template.
The semantic digest excludes runtime values, source identities and locations.
The reflection binds this digest to the runtime compatibility and ABI.

~~~bash
source env.sh
bash scripts/build_m3.sh
cmake --build build/m1 --target coral-ffn-artifact-check -j4
python compiler/compile_ffn.py \
  --input PATH_TO_M3_CAPTURE/imported.mlir \
  --m 1 --d 64 --f 172 \
  --output build/my-ffn-decode
~~~

The output must be a new directory. Compilation occurs in a sibling staging
directory; failed builds retain their logs. Only a completely validated package
is published, atomically and without replacing an existing directory. The
runtime never needs the compiler. Automatic cache lookup and JIT are not
implemented in this milestone.

The compiler records the actual compiler/linker/CRT binary hashes and checks
the source lock and runtime compatibility. Numerical work must be assigned to
Coral. The affinity report counts scheduled dispatch sites; the executable
dumps enumerate unique exports. Deduplication can reuse an export at several
sites (for example, the two weight transposes in decode). These counts are
recorded separately. Every exported executable must target Coral.

Reproduce the compiler gate using retained M3 importer evidence:

~~~bash
python tests/compiler/verify_ffn_aot.py \
  --fixtures PATH_TO_SYNTHETIC_FFN_IMPORT_EVIDENCE \
  --real-fixtures PATH_TO_MODEL_FFN_EVIDENCE
~~~

Evidence: `build/m4-aot-results/aot-cvoknmn_/report.json`. Fourteen
imported programs pass semantic normalization, ten invalid semantic variants
are rejected, and seven equivalent source spellings produce the same semantic
identity. Round-trip normalization, overwrite rejection and failed-publication
rejection also pass.

| Profile | M / D / F | Scheduled Coral dispatches | Unique exports |
| --- | --- | --- | --- |
| tiny | 2 / 3 / 4 | 3 | 3 |
| real-decode | 1 / 64 / 172 | 6 | 5 |
| real-prefill | 5 / 64 / 172 | 3 | 3 |
| profile-m1 | 1 / 128 / 256 | 6 | 5 |
| profile-m16 | 16 / 128 / 256 | 3 | 3 |
| profile-m32 | 32 / 128 / 256 | 3 | 3 |

All six profiles contain zero host numerical dispatches. These are compilation
results; runtime correctness and actual MPACT execution are a separate gate.
The updated artifact validator also passes the 54 contract negative cases;
evidence: `build/m4-contract-results/contract-9m6spbnx/report.json`.

## Step 3: independent FFN runtime

Status: implemented. The public API is
[`src/runtime/ffn_runtime.h`](../src/runtime/ffn_runtime.h).
`FFNModule` and `PreparedFFNWeights` reuse the existing VM/HAL ownership,
Coral MPACT device, rejecting host executable loader, dispatch observer and
output checks. The runtime links neither GGML nor MLIR/compiler libraries;
GGML is linked only into the numerical test executable.

The module validates the package before loading its owned bytecode snapshot.
Actual VM function reflection and the synchronous four-reference calling
convention must match before context initialization. A call binds
`X[M,D], W_gate[F,D], W_up[F,D], W_down[D,F]` and returns an owned `Y[M,D]`.

Example, assuming the vectors already contain dense F32 tensors:

~~~cpp
#include "runtime/ffn_runtime.h"

coral::contracts::FFNCompileSpec spec;
spec.shape = {1, 64, 172};

coral::runtime::DeviceContext device;
coral::runtime::FFNModule module(device, artifact_directory, spec);
coral::runtime::PreparedFFNWeights weights(
    device,
    spec,
    coral::runtime::TensorBinding::Dense(gate_values, 172, 64),
    coral::runtime::TensorBinding::Dense(up_values, 172, 64),
    coral::runtime::TensorBinding::Dense(down_values, 64, 172));

auto output = module.Run(
    coral::runtime::TensorBinding::Dense(input_values, 1, 64),
    weights);
~~~

All three weight bindings are validated before any upload. Preparation creates
immutable snapshots, with no cache keyed only by shape. The original vectors
can change or die after preparation. Snapshots can serve different M profiles
with matching D/F on the same device; modules still require an exact M match.
Activation buffers are reused but their contents are uploaded on every call.

Modules and weights retain the device after the `DeviceContext` wrapper dies.
Moved-from objects, wrong shapes, wrong strides, missing values and foreign
device weights are rejected before invocation. Calls are synchronous and
objects are confined to one host thread.

A result is returned only after successful VM execution, complete Coral
instruction evidence, output shape/type/size checks and readback. Execution
failure returns no partial result and makes the shared context unusable;
there is no implicit CPU replay. Artifact/ABI rejection before VM context
initialization leaves the device available for loading a valid module.

After the README's M2 build setup:

~~~bash
source env.sh
cmake --build build/m1 --target coral-ffn-runtime-test -j4
python tests/runtime/verify_ffn.py \
  --artifacts build/m4-aot-results/aot-cvoknmn_

# Inspect one already-compiled real-model shape directly:
build/m1/coral-ffn-runtime-test run \
  build/m4-aot-results/aot-cvoknmn_/real-decode 1 64 172
~~~

Use the evidence directory produced by your own AOT verification run on another
machine. `verify_ffn.py` preserves artifacts/logs and uses generated finite test
values with GGML CPU as its numerical reference. Its fault-injection cases
intentionally bypass production semantic admission to test VM/output rejection.

See the [steps 1–3 test report](m4-step1-3-report.md) for all results and timing
definitions. A general standalone artifact runner with captured real-model
inputs and a CPU/StableHLO/Coral comparison, GGML complete-region routing, and
the final compiler-unavailable acceptance remain subsequent M4 work.
