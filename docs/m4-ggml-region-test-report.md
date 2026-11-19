# M4 step 5: GGML complete FFN region acceptance

Later acceptance: the [compiler-free deployment gate](m4-deployment-test-report.md)
has now passed. Scope statements below describe this report's original test run.

Date: 2026-11-18. **Passed within the scope below.**
Implementation commit: `eb0b63a` on `codex/m4-ffn-aot`.

Complete FFNs now execute inside actual llama.cpp inference through the
unmodified GGML scheduler. Each eligible region issues one `FFNModule::Run`
and bypasses its original member kernels. Attention and other unsupported
operations continue on CPU; supported matmuls outside regions still use Coral.
The final M4 deployment gate with compiler entry points physically unavailable
has not been completed.

## Evidence and reproducibility

Retained local evidence (all paths are relative to the repository root):

- Backend suite: `build/m4-region-backend-results/backend-btflb88y/report.json`.
- Model suite: `build/m4-region-model-results/model-69qio4fj/report.json`.
- Per-run `summary.json`, `steps.jsonl`, `trace.jsonl`, `logits.f32`,
  `tokens.txt` and stdout/stderr logs are in the model evidence directory.
- `runner-exec.trace` records the isolated process's actual executable launches.
- Regression log: `build/m4-region-ctest.log`; **5/5 passed**.

The model is `stories260K-f32.gguf`: five layers, D=64, F=172, vocabulary 512.
The prompt is “Once upon a time”: five prefill tokens and three decode steps.
The model SHA-256 is
`270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d`.
The clean llama.cpp revision is
`9fba51d2c9a1b5cb5f0c47986c031334e2d7663a`.
Runner SHA-256: `daa89eedc76b152c455baac8ebb95146f059a66e0224a3ba3c7d99b754e7e6a7`.

To reproduce using the retained packages:

~~~bash
source env.sh
cmake --build build/m1 --target coral-llama-runner coral-ffn-region-test -j4

python tests/ggml/verify_ffn_region_backend.py \
  --matmuls build/m2-artifacts/model-aot-c2u8iuxg \
  --pointwise build/m3-pointwise-artifacts/pointwise-xc76cmwd \
  --ffn build/m4-aot-results/aot-cvoknmn_

python tests/ggml/verify_ffn_regions.py \
  --matmuls build/m2-artifacts/model-aot-c2u8iuxg \
  --pointwise build/m3-pointwise-artifacts/pointwise-xc76cmwd \
  --ffn build/m4-aot-results/aot-cvoknmn_
~~~

Both suites preserve existing build caches and write new evidence directories.
The model suite requires `strace`. On another machine, use equivalent artifacts
produced by the M2/M3 and [M4 AOT](m4-artifacts.md) preparation guides.

To inspect one real complete-region run yourself:

~~~bash
source env.sh
EVIDENCE=build/m4-region-model-results/model-69qio4fj
RUN_ROOT="$(mktemp -d build/m4-region-manual-XXXXXX)"
build/m1/coral-llama-runner \
  --mode coral --model models/stories260K-f32.gguf \
  --tokens "$EVIDENCE/cpu/tokens.txt" \
  --artifacts "$EVIDENCE/artifacts" --observation backend \
  --output "$RUN_ROOT/run"
python -m json.tool "$RUN_ROOT/run/summary.json"
rg '"kind":"ffn_region"' "$RUN_ROOT/run/trace.jsonl"
~~~

Expect `ffn_calls=20`, `ffn_members_skipped=80` and `calls_completed=104`.
The default node-observation mode disables fusion; this command explicitly
selects backend observation.

## Model correctness and route evidence

All five paired process runs use the CPU baseline's exact token replay.
Every complete-region run has four logit rows, 512 finite F32 values per row.

| Check | Result |
| --- | --- |
| Complete regions versus member logits | Bit-identical, maximum absolute error 0 |
| Complete regions versus GGML CPU logits | Maximum absolute error 8.58306884765625e-6 |
| Acceptance tolerance | abs(error) <= 1e-4 + 1e-4 * abs(CPU value) |
| Argmax, all four rows | Identical: 432, 383, 286, 261 |
| Fresh-process repetition | Bit-identical logits across all five runs |
| Executed FFNs per step | Five, 20 total |
| Live guard rejection in normal model run | Zero |
| Host executable loads / failed dispatches / observer errors | Zero / zero / zero |
| Instruction samples | One for every Coral dispatch |

The prefill has four M=5 regions and one M=1 region; decode has fifteen M=1
regions. Each M=5 program issues **three** Coral dispatches. Each M=1 program
issues **six** dispatches (five unique executable exports). One VM call is
not a claim of one device kernel.

Example actual trace fields from `regions-0/trace.jsonl`:

~~~json
{"kind":"ffn_region","step":0,"m":5,"d":64,"f":172,
 "dispatches":3,"instructions":132296,"weight_cache_hit":false,
 "program_sha256":"e82d1e7ac46b4f6755b3d3c4e22c5fb959b2a35bf41bbfe5396a0d4d895312d6"}
~~~

The original record also identifies its compute range and all four members:
`ffn_gate-0`, `ffn_up-0`, `ffn_swiglu-0`, `ffn_out-0`.
The verifier checks that no member name also appears as an individual
execution in the same step. Region traces sum to runtime counters; reservation
plans are not counted as execution.

## Calls, buffers and instructions

Counts below are for one complete prefill plus three decode steps.

| Metric | Member execution | Complete FFN regions |
| --- | ---: | ---: |
| All VM calls | 164 | 104 |
| Whole FFN calls | 0 | 20 |
| Individual matmul calls | 144 | 84 |
| Individual pointwise calls | 20 | 0 |
| Original member kernels bypassed | 0 | 80 |
| Coral dispatches | 276 | 256 |
| Kernel instructions | 1,496,248 | 1,504,628 |
| Loaded modules | 11 | 7 |
| Binding allocations | 49 | 43 |
| Logical upload bytes | 1,172,032 | 1,088,512 |
| Logical download bytes | 122,432 | 48,128 |
| GGML compute ranges | 84 | 84 |

Whole-FFN weights are prepared as five three-weight sets and reused 15 times,
including across prefill/decode profiles. Exact shapes share code, while
different layers retain separate weight identities.

Logical upload/download counters count runtime binding traffic. They do not
measure physical DDR, TCM or bus transfers. Fewer calls and bytes did **not**
reduce instruction count in this compiler configuration.

## Measured timing

Ubuntu 22.04 / WSL2, host diagnostic runner, one GGML CPU thread, MPACT simulation.
Five fresh processes per path, with member/region order alternating.
Both paths use `--observation backend`. Warm decode means steps 2 and 3
(the first decode can introduce a new module/profile). No statistical
significance or silicon performance claim is made.

| Measurement | Member median | Region median | Samples per path |
| --- | ---: | ---: | ---: |
| Prefill wall time | 63.7366 ms | 67.8839 ms | 5 |
| Warm decode wall time | 26.4225 ms | 26.56825 ms | 10 |
| Warm decode VM invocation time | 25.7566 ms | 25.9738 ms | 10 |

Warm decode wall ranges: member 26.2392–28.119 ms; region 25.8012–27.6634 ms.
A whole warm M=1 FFN invocation has median 3.06745 ms across 50 observations.

The region path shows **no measured latency improvement**: its warm median is
about 0.55% higher and prefill about 6.51% higher. This acceptance establishes
correct region routing and fewer outer calls, not a speedup. Runtime package
validation and initialization are included in cold inference; the measured
data does not isolate their individual contribution to the prefill difference.

## Boundary tests

The backend suite runs **84 real scheduled graphs, three executions each**,
plus a direct-compute control and a corrupt-package failure/retry control.
It tests split SwiGLU and both expanded SiLU/multiply operand orders at M=1/5.

- 180 complete FFN calls across mutable/immutable suites; 840 member kernels
  bypassed. Maximum synthetic absolute error is 7.45058e-9.
- Two chained regions in one split and two regions separated by a CPU operation.
- Graph reuse without reoptimization; reservation of a different graph whose
  tensor arena is destroyed before the actual allocation/compute.
- Changed activations and mutable WEIGHTS-tagged buffers; immutable snapshots
  reused only with the explicit promise and separate ordered weight identities.
- External consumers in a CPU split and external views: no fusion, correct output.
- Callbacks enabled explicitly, and actual callback subdivision even with a
  false no-callback declaration: no fusion, intermediate callbacks preserved.
- Metadata changed after allocation: live guard rejects on all three reuses,
  with original member execution retained.
- Missing full-region artifacts, disabled fusion and missing pointwise members:
  correct member/CPU routing without a complete FFN call.
- Corrupt package: failed graph, unchanged output marker, no VM attempts or
  member replay; a second compute on the failed backend also fails.
- Eleven invalid catalog forms rejected, including duplicate/invalid profiles,
  nested paths, symlink directories, missing packages and malformed records.

Real-model controls independently confirm callbacks, missing packages,
missing member capability and mixed profile coverage. Keeping only the M=1
package executes 16 regions and preserves member execution for four M=5 FFNs.
Corrupt VMFB and wrong-shape package controls fail without a success summary.

## Scope remaining

All child runner processes use `PATH=/nonexistent`. The isolated `strace` run
contains exactly one `execve`: the runner itself. There is no compiler or
reference-interpreter launch during inference. Compiler files still exist on
this development workstation; physically unavailable compiler deployment is
the remaining M4 gate.

The previous [standalone three-way acceptance](m4-runner-test-report.md)
established these complete artifacts against the official StableHLO interpreter.
This integration suite compares actual model inference to independent CPU and
member processes; it does not rerun that interpreter for the entire model.

No upstream llama.cpp patch, custom scheduler observer, private
`llama_context::get_sched()` or inference-time compilation was introduced.
