# M4 independent FFN runner

The `coral-ffn-runner` executable loads a complete AOT package and runs supplied
tensors on Coral MPACT. `coral-ffn-cpu-reference` evaluates the same positional
FFN through upstream GGML CPU in a separate process. The official
`coral-stablehlo-reference` remains the StableHLO numerical oracle.

Acceptance: all 34 synthetic/real-model cases pass across nine profiles.
See the [test report](m4-runner-test-report.md) for execution evidence and timing.

## Build and run

After the README build setup:

~~~bash
source env.sh
cmake --build build/m1 --target coral-ffn-runner coral-ffn-cpu-reference -j4

build/m1/coral-ffn-runner \
  --artifact PATH_TO_ARTIFACT \
  --inputs PATH_TO_M3_CAPTURE/inputs.json \
  --repeat 3 --recreate 2 \
  --output build/ffn-coral-result.json

build/m1/coral-ffn-cpu-reference \
  --inputs PATH_TO_M3_CAPTURE/inputs.json \
  --repeat 3 --output build/ffn-cpu-result.json

build/m3/coral-stablehlo-reference \
  PATH_TO_ARTIFACT/region.stablehlo.mlir ffn \
  PATH_TO_M3_CAPTURE/inputs.json build/ffn-stablehlo-result.json
~~~

All output files must be new. Both standalone runners publish complete JSON
only after successful execution of every requested call, without overwriting an
existing file. Compiler execution is not part of either runtime runner.

The Coral runner is available in the runtime build. The CPU oracle requires
`CORAL_BUILD_M2=ON` to supply upstream GGML. The Coral runner does not link
GGML, MLIR or compiler libraries. The shared tensor I/O helper links only the
artifact utility and pinned JSON/SHA implementation.

## Input and execution contract

Input is the existing M3 positional tensor JSON, with exactly one `inputs`
array containing four objects. Each object has exactly `dtype`, `shape` and
flat `values` fields:

| Position | Role | Dense row-major shape |
| --- | --- | --- |
| 0 | Activation | M × D |
| 1 | Gate weight | F × D |
| 2 | Up weight | F × D |
| 3 | Down weight | D × F |

Only finite F32 values are admitted. Dimensions must be positive JSON integers;
boolean values and fractional dimensions are rejected. Unknown fields, duplicate
keys, incorrect lengths and incompatible role shapes are also rejected. The
artifact validator checks that the inferred exact profile matches the package.

Diagnostic resource bounds are 16 MiB per JSON file, 1,048,576 total input
elements per case, the same bound per M×F intermediate, 64 cases, 4,194,304
input elements across cases, and 1,024 invocations per process. Repeat is
limited to 100 and context recreation to 16. These are runner limits, not
device capability claims.

Repeat `--inputs` to run several cases with one module and exact profile:

~~~bash
build/m1/coral-ffn-runner --artifact PATH_TO_ARTIFACT \
  --inputs FIRST_CAPTURE/inputs.json --inputs SECOND_CAPTURE/inputs.json \
  --repeat 3 --recreate 2 --output build/ffn-batch-result.json
~~~

The runner loads one module per context and prepares independent immutable
weight snapshots for each case. It reuses activation storage while uploading
each activation anew. Across repeats and recreated contexts, each case must
produce identical finite F32 output. The input SHA-256 hashes identify the
exact bytes parsed, and `program_sha256` identifies the semantic program
actually loaded. Invalid inputs are checked before device creation.

## Output and timing

A `coral.ffn.runner.v1` result contains the profile, program identity, one
output tensor per case, input digests, per-call timing/dispatch/instruction
evidence, and complete counters for every context lifetime. `status=passed`
means the runtime's execution/ABI/repeat checks passed; numerical acceptance
against independent references is a separate gate.

`wall_ms` brackets the synchronous runtime call including explicit upload and
readback. `invoke_ms` brackets the VM invocation; it can include first-use
executable loading and internal transfers. The report also separates input
parsing, shared-device creation, module construction (including artifact
validation), and weight preparation. `first_call_in_context` distinguishes
cold calls from subsequent calls. These are simulator host times.

The GGML CPU report separates graph/tensor setup from execution, uses one CPU
thread and does not provide a runtime fallback. StableHLO timing includes
reference interpretation. None of these measures is NPU hardware throughput.

## CLI validation

~~~bash
python tests/runtime/verify_ffn_runner_cli.py \
  --artifact PATH_TO_TINY_ARTIFACT \
  --fixture PATH_TO_M3_FFN_REGION_SPLIT
~~~

The first runner milestone passes two batch inputs across two Coral context
lifetimes, three numerical comparisons and 54 negative processes. It covers
changed activation/weights, repeated calls, malformed tensors, CLI limits,
existing-output preservation and input/output aliasing.
Evidence: `build/m4-runner-cli-results/cli-h2wa2d9z/report.json`.

## Three-way validation

The acceptance tool consumes passing M3 synthetic and model capture directories
and the M4 AOT evidence directory. It reuses existing packages and prepares only
missing exact profiles in a new evidence directory. Compiler use is confined
to this preparation phase.

~~~bash
python tests/runtime/verify_ffn_runner.py \
  --fixtures PATH_TO_M3_SYNTHETIC_FFN_EVIDENCE \
  --real-fixtures PATH_TO_M3_MODEL_FFN_EVIDENCE \
  --artifacts PATH_TO_M4_AOT_EVIDENCE
~~~

For each captured case, the verifier:

1. Snapshots the four tensor inputs, captured GGML output and imported module.
2. Normalizes the captured computation and requires its semantic hash to match
   the artifact. Numerical interpretation uses the artifact's actual StableHLO.
3. Recomputes GGML CPU output, executes Coral in batches with independent
   weights, and reloads the package in another process.
4. Runs the official StableHLO interpreter on the same input bytes.
5. Compares all three engine pairs, and fresh GGML output against its original
   captured result, with fixed `atol=1e-4, rtol=1e-4`.
6. Swaps same-shaped real gate/up weights as a negative control: the engines
   must agree on the changed inputs, while the original captured expectation
   must reveal the binding mistake.

Per-case input/program hashes, all three outputs, maximum errors, failure
indices, timing and raw logs are retained. A numerical mismatch produces a
failed aggregate report and nonzero exit. No tolerance is adjusted by the tool.
The main Coral/CPU timing phase finishes before the slower reference phase.
Large reference profiles can take several minutes.

Coverage is all fourteen positive synthetic imports and all twenty captured
model regions (five layers across prefill and three decode steps). The real
capture includes four M=5 prefill regions and one M=1 last-layer prefill region,
plus fifteen M=1 decode regions.
