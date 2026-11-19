# M4 steps 1–3: validation report

Date: 2026-11-18. Scope: artifact contract, complete FFN AOT compiler entry,
and independent FFN runtime. These three implementation milestones pass.
This is not acceptance of all M4 work. Subsequent independent runner results
are recorded in the [step 4 report](m4-runner-test-report.md).

## Implementation and evidence

- Step 1, `8793a89`: versioned manifest/schema, build compatibility,
  fixed input roles, hashes, reflection contract and bounded package loading.
- Step 2, `b596b5e`: semantic admission of actual imported StableHLO,
  normalized identity, full FFN compilation and atomic artifact publication.
- Step 3, the runtime commit containing this report: `FFNModule` and
  `PreparedFFNWeights`, reusing the existing strict VM/HAL execution layer.

The compiler inputs include actual M3 importer outputs for synthetic graphs
and real tiny-model decode/prefill shapes. Runtime numerical data in this gate
are deterministic generated F32 tensors; this is not captured model-weight
three-way acceptance or a new end-to-end inference benchmark.

| Gate | Result | Retained evidence |
| --- | --- | --- |
| Artifact contract | 1 positive + 54 negative cases pass | `build/m4-contract-results/contract-9m6spbnx/report.json` |
| AOT semantics/publication | 14 normalized imports, 10 semantic rejections, 6 compiled profiles; equivalent-source identity, round trip, overwrite and failed-publication checks pass | `build/m4-aot-results/aot-cvoknmn_/report.json` |
| Complete FFN runtime | 86 numerical comparisons; 9 artifact/VM/output negative cases pass | `build/m4-runtime-results/runtime-9b6tvve2/report.json` |
| M1/M2/planner CTest | 5/5 pass | `build/m4-ctest.log` |
| M3 pointwise | 48 calls, 48 invalid binding checks, 8 negative processes pass | `build/m4-pointwise-regression/pointwise-u57grydq/report.json` |
| Unpatched upstream | 33 fusion cases, M2 inference, importer and real-model importer pass | `build/m4-unpatched-regression/unpatched-j5yax2n1/report.json` |
| Real FFN members | 20 complete member groups, missing-member fallback and corrupt-member failure pass | `build/m4-members-regression/members-1lizkfml/report.json` |

All paths are relative to the repository and are ignored build outputs.
Builds, packages and logs were retained.

## Numerical and device results

The complete FFN numerical threshold is
`abs(actual - expected) <= 2e-5 + 2e-4 * abs(expected)`.
Every result must also be finite. GGML CPU evaluates the same positional
gate/up/SwiGLU/down computation.

Each profile runs fourteen calls across two device lifetimes: twelve with
exported counters and two additional calls after destroying the owning context
wrapper. Two further comparisons reuse one weight set across real-model
decode/prefill profiles. One harness-owned package is moved after loading to
verify that invocation uses owned VMFB bytes.

| Profile | M / D / F | Max absolute error | Observed dispatches (12 counted calls) | Kernel instructions |
| --- | --- | --- | --- | --- |
| tiny | 2 / 3 / 4 | 2.98023e-08 | 36 | 6,084 |
| real-decode | 1 / 64 / 172 | 1.01328e-06 | 72 | 476,844 |
| real-prefill | 5 / 64 / 172 | 1.43051e-06 | 36 | 1,587,552 |
| profile-m1 | 1 / 128 / 256 | 3.57628e-06 | 72 | 1,236,516 |
| profile-m16 | 16 / 128 / 256 | 5.24521e-06 | 36 | 398,532 |
| profile-m32 | 32 / 128 / 256 | 5.24521e-06 | 36 | 742,212 |

The 72 counted calls produced **288 successful Coral dispatches and 4,447,740
kernel instructions**, with a matching instruction sample for every dispatch,
zero host executable loads, zero failed dispatches and zero observer errors.
The extra lifetime/cross-profile calls also pass the runtime's per-call
execution checks. A complete FFN VM invocation does not imply one HAL dispatch.

Binding tests exercise all three malformed weight roles before allocation,
null/size/stride/shape errors, a foreign context, incompatible weights,
moved-from objects, independent same-shape weight sets, shared gate/up data,
host-vector mutation/destruction and repeated activation updates. Reusing a
module adds no new activation allocation after the first call.

Artifact tests cover missing/incomplete packages, changed bytes, target and
role mismatches, corrupt VMFB bytes with recomputed hashes, and mismatched
actual VM reflection. Test-only counterfeit modules exercise wrong argument
arity and output shape after package validation. Production semantic admission
cannot produce these counterfeit packages.

The wrong-output execution records one attempted call, zero completed calls,
one Coral dispatch and zero host loads. A second call is rejected before
another attempt:

~~~text
output violates the F32 [M,N] tensor ABI
context unusable after execution failure; create a new DeviceContext
~~~

Evidence: `build/m4-runtime-results/runtime-9b6tvve2/wrong-output.stdout.txt`
and `wrong-output.stderr.txt` in the same directory.

## Diagnostic performance

These are software MPACT wall times on the current WSL host, not hardware NPU
latency, throughput or speedup claims. The sample is a functional test with
twelve recorded invocations per profile, not a controlled benchmark.

- AOT time includes preparation, compilation and artifact validation.
- Constructor time includes package reads/hashes/validation and VM setup, but
  excludes creating the shared device context and preparing weight snapshots.
- Invoke time measures `iree_vm_invoke`, including its internal work, but excludes
  explicit activation upload and final output readback. The first call may
  include executable loading. The reported mean includes those first calls.
- Upload/download counters are logical host/HAL copies, not total simulator
  DDR/TCM traffic. CPU reference timings include test setup and are not a
  comparable optimized CPU benchmark.

| Profile | AOT build (s) | Mean module constructor (ms, 2 samples) | Mean VM invoke (ms, 12 samples) |
| --- | --- | --- | --- |
| tiny | 8.587 | 0.872 | 0.411 |
| real-decode | 8.954 | 0.969 | 3.448 |
| real-prefill | 9.538 | 0.900 | 10.108 |
| profile-m1 | 8.384 | 0.979 | 11.036 |
| profile-m16 | 8.534 | 0.943 | 12.946 |
| profile-m32 | 8.234 | 0.881 | 23.247 |

## Reproduction and remaining scope

Follow the [artifact/runtime guide](m4-artifacts.md). To inspect an already-built
profile in the retained workspace:

~~~bash
source env.sh
build/m1/coral-ffn-runtime-test run \
  build/m4-aot-results/aot-cvoknmn_/real-decode 1 64 172
~~~

The dependency lock remains unchanged. The llama.cpp checkout is clean at
`9fba51d2c9a1b5cb5f0c47986c031334e2d7663a`; no scheduler observer patch,
private scheduler accessor or original-graph observer was reintroduced.

Pending M4 work: a general runner with captured real inputs and
CPU/StableHLO/Coral comparison; connecting complete regions to GGML plans;
per-operation versus complete-region comparison; and final acceptance in an
environment where compiler entry points are unavailable. The existing GGML
backend still invokes individual supported members.

Runner SHA-256: `1e2dd3ba1af992435e3a568f5235d5643f65441742bfcd2b235df7df43f634dd`.
AOT report SHA-256: `0bbfec5d03be0e9ae5970eb20ce179693a77be3fcb13814ce4b86095e994f7fe`.
Runtime report SHA-256: `7bc4d06fb999f8a5e221eec7f123c592b49ea321f6768e7ef94533725095c9f2`.
