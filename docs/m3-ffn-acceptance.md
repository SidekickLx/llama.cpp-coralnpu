# M3: complete FFN semantic acceptance

**Accepted on 2026-11-17** for the documented static dense F32 subset.
All nine aggregate suites pass on the locked, unmodified llama.cpp checkout.
Complete FFN device artifacts and three-way CPU/StableHLO/Coral validation
remain M4. This is a development milestone, not a versioned release.

## Fixed role importer

`ImportGGMLFFNFunction` takes exactly four ordered boundary roles:
X, gate weight, up weight and down weight, plus the complete topological
member selection and one output. It exports `ffn` with the declared F32
contract and role attributes. It verifies the requested M/D/F dimensions,
the exact edge-to-role correspondence, member completeness and identities.
The existing per-operation importer handles parameter, type, layout and SSA
validation, followed by MLIR/StableHLO verification.

The matcher is shared with the planner in `src/planner/ggml_ffn_pattern.*`.
It reads GGML metadata and makes no scheduling or execution decision.
No additional computational IR, custom operation or numerical evaluator is
introduced.

Repeated source weight identities keep separate positional function arguments.
Matmul operands are assigned by their validated consumer role. This preserves
both original tensor identity metadata and the general four-argument FFN
semantics. The generic `ImportGGMLFunction` API retains its unique-input rule.

## Acceptance method

The semantic baseline is actual GGML CPU execution. The evaluator is the pinned
official StableHLO interpreter, using only standard operations and no custom
FFN callbacks. Both split SwiGLU and expanded SiLU/multiply spellings are covered.

The synthetic suite covers tail dimensions, prefill/decode, distinct and shared
weights, both multiply operand orders, gate/up reversal and independent
projection order. It includes the contract profiles M=1/16/32, D=128, F=256.
Malformed contracts, roles, parameters, dependency order/cycles, member sets,
outputs and shared identities must be rejected without publishing a module.

Two numerical counterexamples test semantics that types alone cannot prove:

- Swap the same-shaped gate/up payloads after GGML CPU execution. Evaluation
  must fail the numerical comparison.
- Evaluate a shared-source-weight FFN with distinct positional gate/up arguments.
  Compare with the distinct-weight GGML case. Neither argument may have been
  specialized away or collapsed by source-identity deduplication.

The real-model harness uses public llama.cpp evaluation callbacks on a CPU-only
run. It snapshots each input **before its consumer executes** and each result
**after its producer completes**. Owned copies survive allocator reuse.
It reconstructs each complete FFN selection from those metadata snapshots and
passes only X/W_gate/W_up/W_down to the reference interpreter. Captured CPU
intermediates are never supplied as intermediate reference results.

Coverage is all five layers of the locked stories260K F32 model: a five-token
prefill and three one-token decode steps, for 20 FFN regions. The final prefill
layer may consume only the selected last-token row. Layer identities must agree
across steps, and repeated decode captures must contain fresh activation values.
Each emitted module is reparsed and verified independently. A separate replay
with swapped real gate/up weights must disagree with GGML.

The full gate also reruns the original importer, M1/M2, all planner safety cases,
and standalone/real-model Coral member execution. This covers external consumers
including another split, intermediate outputs, views/aliases, callback
subdivision, graph reuse, stale plans and boundary input lifetime. See
[planner test details](m3-fusion.md) for the real upstream allocator experiments
and their CPU-backed cross-device fixture limitation.

## Reproduce

Start from the README setup, with the existing CMake runtime/simulator build,
clean locked llama.cpp checkout, model and configured M1 CTest artifacts.
Build incrementally; do not remove caches or generated artifacts.

~~~bash
source env.sh
bash scripts/build_m3.sh
cmake --build build/m1 --target \
  coral-matmul-runner coral-llama-runner coral-adapter-test \
  coral-ffn-planner-test coral-ggml-import-test coral-model-import-test \
  coral-pointwise-test coral-member-adapter-test -j4

# Reuse the printed directories from the M2 and M3 preparation guides.
MATMULS=build/m2-artifacts/model-aot-c2u8iuxg
POINTWISE=build/m3-pointwise-artifacts/pointwise-xc76cmwd
python tests/verify_m3.py --matmuls "$MATMULS" --pointwise "$POINTWISE"
~~~

The example directory names identify this machine's retained artifacts; use
your own generated directories after a fresh setup. Matmul preparation is in
the [README Quick start](../README.md#quick-start); pointwise preparation is in
the [member guide](m3-pointwise.md). The gate requires all five CTests to be
configured and produces a fresh evidence directory. It does not compile VMFBs.

For only the new full-region checks:

~~~bash
python tests/compiler/verify_ffn_import.py \
  --runner build/m1/coral-ggml-import-test \
  --check-library build/m3/libcoral-importer-check.so \
  --reference-runner build/m3/coral-stablehlo-reference \
  --output-root build/m3-ffn-import-results
python tests/compiler/verify_model_ffn.py \
  --runner build/m1/coral-model-import-test \
  --check-library build/m3/libcoral-importer-check.so \
  --model models/stories260K-f32.gguf \
  --tool build/m3/coral-stablehlo-tool \
  --reference-runner build/m3/coral-stablehlo-reference \
  --output-root build/m3-model-ffn-results
~~~

A retained region contains `capture.json` (real-model provenance),
`imported.mlir`, `inputs.json`, `ggml-output.json`, `reference-output.json`,
`report.json` with errors/failing indices/timings, and verifier logs.
The aggregate report references every child report and records the tested
binary and model hashes, source revision and whether the project had
uncommitted changes during testing. The locked llama.cpp source is checked
before and after acceptance. Runtime child processes retain the existing
no-compiler-in-PATH checks.

Initial fixed-role importer milestone: `18dfaa7`. Its original 11-case evidence
is `build/m3-ffn-import-results/ffn-import-cktynamd/report.json`; the final gate
below adds the three larger contract profiles and real-model region acceptance.

## Final results

Aggregate evidence: `build/m3-acceptance-results/acceptance-fe6qikxj/report.json`.
Build log: `build/m3-acceptance-build.log`; final fixture rebuild:
`build/m3-contract-profile-build.log`. The final gate took
267.271 seconds with retained builds and artifacts.

| Gate | Observed result |
| --- | --- |
| Existing CTest | 5/5 pass: three M1 shapes, M2 adapter boundaries and M3 planner |
| Typed StableHLO verifier | 5 valid modules, 7 malformed modules, 3 invalid specs; overwrite protection |
| Official reference integration | 5 numerical cases and 10 negative cases |
| Unpatched integration | M2 inference and exact CPU replay, 6 expected M2 failures, original importer 12 numerical / 99 rejection cases, 4 real model nodes |
| Fusion and lifetime | All 33 planner cases pass, including actual scheduler/allocator experiments |
| Complete synthetic FFNs | 14 numerical cases, 12 rejection cases, payload-swap failure and shared-argument independence |
| Complete real FFNs | 20 regions, 2,304 compared output values, 20 serialized-module verifier checks; swapped real weights detected |
| Standalone Coral pointwise | 6 profiles, 48 calls/dispatches, 48 invalid bindings and 8 failing processes |
| Member adapter | 138 checks and 10 malformed catalogs |
| Real Coral member placement | 20 complete FFN member groups; missing-member CPU routing and corrupt-member failure pass |

Rows describe coverage and overlap; they must not be summed as independent
test counts. The nine-suite gate includes planner and M2 checks in its
`unpatched` suite as well as CTest.

### Numerical evidence

The frozen FFN tolerance is elementwise
`abs(actual-reference) <= 1e-4 + 1e-4 * abs(reference)`. It was not widened.

| Comparison | Maximum absolute error | Result |
| --- | ---: | --- |
| Complete synthetic FFNs, including M=1/16/32 and D=128/F=256 | 0.000152587890625 | Pass |
| Complete real-model FFNs, official reference vs GGML CPU | 1.43051147461e-06 | Pass |
| Standalone Coral pointwise vs GGML CPU | 9.53674316406e-07 | Pass at the existing `atol=2e-6, rtol=1e-5` |
| Real-model logits with Coral member execution vs CPU | 8.58306884766e-06 | Pass |
| Original M2 matmul-only logits vs CPU | 7.152557373046875e-6 | Pass |

The larger synthetic fixtures have larger output magnitudes: their maximum
absolute error can exceed the absolute term alone while satisfying the
combined absolute/relative policy. The maximum relative error for nonzero
synthetic reference values is 1.91515977389e-05. Every compared value passes;
per-case reports retain maximum errors and failing indices.

Swapping the real gate/up payloads causes 320
mismatched outputs in the first prefill region. The ordinary run has no
failing indices. Fresh decode activation checks and layer coverage checks pass.

### Device execution evidence

The backend-only observation run executes 144 matmuls and
20 split SwiGLU calls across prefill and three decode
steps. These produce 276 successful Coral dispatches,
1,496,248 simulated kernel instructions and the same
number of instruction samples as dispatches. Host numerical executable loads,
failed dispatches and observer errors are all zero.

All 20 FFN member groups occupy complete admitted Coral compute ranges. The
30 preparation-plan count includes graph reservation and is not a count of
executed FFNs. Each member still invokes its own program; the final FFN
StableHLO module is not compiled or invoked as a complete device artifact yet.

### Diagnostic timing

| Measurement | This run |
| --- | ---: |
| Import/verify 20 real FFN regions, aggregate | 80.327 ms |
| Official reference evaluation of those regions, aggregate | 47.242 s |
| CPU capture plus import/reference, excluding later verifier/swap replay | 48.865 s |
| All synthetic FFN tests, including large contract profiles | 186.821 s |
| Standalone pointwise VM invocation, 48 calls | 27.194 ms |
| Real-model selected Coral member VM invocation, 164 calls | 145.080 ms |
| Real-model module loading, 11 modules | 0.7907 ms |

These are single-run host diagnostic measurements. The official interpreter
prioritizes reference semantics, and its elapsed time is not a CPU inference
performance baseline. MPACT VM invocation time is simulator wall time, not
device latency or a hardware speedup. The model VM total includes all selected
operations, not just the FFN members.

### Evidence map

The paths below are relative to the aggregate evidence directory. It also
contains `ctest.xml` and per-suite stdout/stderr. Raw generated files remain
local under ignored `build/` directories; the report and test code are tracked.

| Suite | Summary |
| --- | --- |
| ctest | `ctest.xml` |
| stablehlo | `stablehlo/verify-scxrkls_/report.json` |
| reference | `reference/reference-0_ohie3c/report.json` |
| unpatched | `unpatched/unpatched-hclvht6c/report.json` |
| ffn-import | `ffn-import/ffn-import-r4t1hl9f/report.json` |
| model-ffn | `model-ffn/model-ffn-0wac56lp/report.json` |
| pointwise | `pointwise/pointwise-29enhv_o/report.json` |
| member-adapter | `member-adapter/adapter-6ruisl64/report.json` |
| ffn-members | `ffn-members/members-t53y4fut/report.json` |

Within the real-FFN child directory, inspect
`capture/step-0-ffn-0/imported.mlir` and its input/CPU/reference JSON files.
Within the member-execution child directory, `backend/trace.jsonl` links
executed node IDs and compute-range IDs to preparation plans;
`backend/steps.jsonl` and `backend/summary.json` record execution and timing.

## Completion boundary and next milestone

M3 now provides the validated semantic bridge, standard typed computation,
verification/reference path, pattern recognition, fusion legality/lifetime
guards, and independently executable member coverage on Coral. The locked
upstream llama.cpp source remains clean, with no scheduler patch or private
`llama_context::get_sched()` dependency.

M4 starts from these verified complete FFN modules. It must compile a versioned
artifact, validate reflection/shape/roles/device compatibility at loading,
execute it with an independent runner, and compare GGML CPU, StableHLO reference
and Coral outputs before allowing GGML to replace member execution. Existing
runtime and planner checks alone do not authorize that replacement.
