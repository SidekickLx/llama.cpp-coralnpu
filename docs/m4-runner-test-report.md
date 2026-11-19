# M4 step 4: independent FFN runner acceptance

Date: 2026-11-18. **Passed.** The standalone runner executes complete FFN AOT
packages with supplied tensor data. The acceptance suite compares fresh GGML
CPU, the official StableHLO interpreter and actual Coral MPACT execution.

Runner implementation commit: `bdf5475`. This report and the reproducible
three-way verifier are included in the subsequent acceptance commit.

This accepts the independent runner milestone. The GGML backend still invokes
individual supported members; complete-region routing is subsequent work.

## Coverage and numerical result

The gate reuses accepted M3 captures: fourteen synthetic FFNs and twenty real
model regions, representing all five layers over prefill and three decode
steps. Each case snapshots actual input bytes, its captured CPU result and
imported StableHLO. The normalized source digest must match the artifact before
running it. The real model is the locked F32 stories260K GGUF.

All 34 cases pass all three pairwise comparisons and replay against captured
GGML output. The threshold was fixed before execution:

`abs(actual - expected) <= 1e-4 + 1e-4 * abs(expected)`

| Comparison | Max absolute error, all cases | Max absolute error, real-model cases |
| --- | --- | --- |
| Coral / GGML CPU | 0.000152587891 | 1.66893005e-06 |
| StableHLO / GGML CPU | 0.000152587891 | 1.43051147e-06 |
| Coral / StableHLO | 6.10351562e-05 | 1.07288361e-06 |
| Fresh GGML / captured GGML | 0 | 0 |

The larger absolute synthetic error occurs on larger-magnitude outputs. Every
element passes the combined absolute/relative bound; the worst error consumes
less than 7.83% of its allowed tolerance. No tolerance was relaxed.

Coverage includes expanded/reversed/up-first FFN spellings, shared gate/up
identity, distinct same-shaped weights, tail dimensions, M=1/16/32 profiles,
and actual changing prefill/decode activations. Real prefill contains four
M=5 regions and one final-layer M=1 region; all fifteen decode regions use M=1.

Nine exact profiles were exercised. Six existing packages were reused; only
three additional profiles, (1,3,4), (5,3,4) and (3,7,11), were compiled. Build
caches and all packages were retained.

## Execution and independence evidence

The main gate performs six Coral calls per case across two context lifetimes,
plus one call after loading the package in a fresh process. That gives
**238 calls, 1,092 successful Coral dispatches and 9,604,119 kernel instructions**.
Every dispatch has an instruction sample. Host executable load attempts,
failed dispatches and observer errors are all zero.

Within each context, cases of the same profile reuse one loaded VM module.
Each case receives separately prepared immutable weights. Every repeat and
fresh-process reload produces identical finite output. Input hashes match
between the CPU and Coral runners, and the loaded program hash matches the
normalized captured computation.

The negative control swaps real gate/up weight payloads of identical shape.
The three engines still agree on those altered inputs, while comparison against
the original captured result detects **320/320 mismatched elements**. This
demonstrates why shape validation alone cannot establish correct role binding.

Additional command tracing ran the Coral runner with `PATH=/nonexistent`.
The trace contains one `execve`, for the runner itself, and no compiler,
interpreter or other child executable. Binary symbol inspection finds no
GGML or MLIR symbols in the Coral runner; the CPU oracle has GGML symbols
and no IREE VM invocation symbol. The isolated run made two additional calls,
and the role-swap control made one; these are excluded from the 238-call total.

Compiler files were still present on disk for this command-tracing check.
The staged deployment test with compiler entry points physically unavailable
remains a separate final M4 gate.

## Failure checks and regressions

| Suite | Result | Evidence |
| --- | --- | --- |
| Runner CLI and publication | 2 batch cases, 3 numerical comparisons, 54 negative processes pass | `build/m4-runner-cli-results/cli-h2wa2d9z/report.json` |
| Three-way replay | 14 synthetic + 20 model cases, 9 profiles, fresh-process reload and role-swap control pass | `build/m4-runner-results/three-way-p3m7_t6e/report.json` |
| Existing FFN runtime | 86 comparisons across 6 profiles and 9 artifact/ABI/output negatives pass | `build/m4-runner-runtime-regression/runtime-pp9sfki6/report.json` |
| M1 runtime, M2 adapter, M3 planner | CTest 5/5 pass | `build/m4-runner-ctest.log` |
| External command isolation | Only the Coral runner is executed | `build/m4-runner-results/three-way-p3m7_t6e/runner-exec.trace` |

CLI negatives cover malformed/duplicate JSON, arity, dtype, shape/rank, value
count, non-finite/overflow data, incorrect weight shapes, invalid options,
existing-output preservation and input/output aliasing. Rejected input or
failed execution publishes no result. Outputs are atomically published and
existing files are never overwritten.

The llama.cpp checkout remains clean at
`9fba51d2c9a1b5cb5f0c47986c031334e2d7663a`. No scheduler patch, private
scheduler accessor or original-graph observer is used.

## Diagnostic performance

The main Coral and CPU runs completed before the slower StableHLO reference
phase. Each case supplies six Coral timing samples across two fresh contexts.
The table shows medians from the main runs; fresh-process reload and the traced
isolation/control runs are excluded.

| M / D / F | Cases | Module constructor (ms) | Warm VM invocation (ms) | Full runtime call (ms) | GGML CPU evaluation (ms) |
| --- | --- | --- | --- | --- | --- |
| 1 / 3 / 4 | 1 | 1.0410 | 0.1884 | 0.2046 | 0.0037 |
| 2 / 3 / 4 | 7 | 0.9163 | 0.1156 | 0.1169 | 0.0011 |
| 5 / 3 / 4 | 1 | 0.9606 | 0.1432 | 0.1491 | 0.0039 |
| 1 / 128 / 256 | 1 | 1.2977 | 10.2078 | 10.2272 | 0.0093 |
| 16 / 128 / 256 | 1 | 1.2128 | 12.0563 | 12.0585 | 0.0439 |
| 32 / 128 / 256 | 1 | 1.2165 | 22.7912 | 22.8593 | 0.0840 |
| 3 / 7 / 11 | 2 | 0.9441 | 0.2197 | 0.2241 | 0.0024 |
| 5 / 64 / 172 | 4 | 1.2477 | 9.6799 | 9.6905 | 0.0077 |
| 1 / 64 / 172 | 16 | 1.4076 | 3.0647 | 3.0666 | 0.0046 |

These are diagnostic host wall times, not physical NPU performance or speedup
claims. The workload is a functional test, not a controlled benchmark.

- Module construction includes artifact reads/hash checks and VM setup, but
  excludes shared-device creation and weight preparation (recorded separately).
- Warm invocation excludes the first call of each context. It includes VM
  execution and internal device work, but excludes explicit input upload and
  output readback. First-call samples remain in the raw report.
- Full runtime call includes explicit upload/readback. Its median includes
  first calls as well as warm calls.
- GGML CPU evaluation uses one thread and measures graph execution; graph
  construction/input setup is recorded separately.
- StableHLO interpretation is a correctness oracle and is not a deployment
  performance baseline. Logical upload/download counts are not total DDR/TCM
  traffic.

For the real-model profiles, warm VM invocation is approximately **3.06 ms**
for M=1 and **9.68 ms** for M=5. Same-profile cases use different model layers
and activations with shared code.

## Reproduce and inspect

See the [runner guide](m4-runner.md) for portable inputs and build instructions.
In the retained workspace, this runs a captured first-layer decode FFN:

~~~bash
source env.sh
RUN_OUTPUT="$(mktemp -d build/ffn-replay-XXXXXX)"
build/m1/coral-ffn-runner \
  --artifact build/m4-aot-results/aot-cvoknmn_/real-decode \
  --inputs build/m4-runner-results/three-way-p3m7_t6e/cases/model-step-1-layer-0/inputs.json \
  --repeat 3 --recreate 2 --output "$RUN_OUTPUT/result.json"
~~~

The aggregate report references per-profile CPU/Coral/reload reports. Each
`cases/CASE/` directory contains inputs, original GGML output, normalized IR,
fresh CPU output, Coral output, StableHLO output and `comparison.json`.
Raw stdout/stderr are retained alongside the aggregate report. These local
data and generated artifacts are ignored by Git.

Identity of the tested binaries:

- Coral runner: `90083e9e11cb6ebc413c4cad7fd636c8c80dd1fbedd5fe5dd96f783607a82093`
- GGML CPU oracle: `99bed6bf3bdd68a775cc055377524a7674f42c150b4619340d37c3ab10d5493b`
- Official StableHLO runner: `fe2cdd41ac4c4711997e267cac9f5202f2a96a5608b66d57afb8dfa0eebe9e7c`
- Aggregate report: `6dd12a45919d47b415636004ccb23a158fc3fadd4a702e722fb9bc06f254314c`

Next: connect complete FFN artifacts to GGML region plans and compare member
execution with complete-region execution. This runner acceptance does not
claim that GGML already performs fused-region calls.
