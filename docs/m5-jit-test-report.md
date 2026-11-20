# M5 steps 1–3: JIT acceptance report

Date: 2026-11-20. Status: **passed within the exact F32 FFN scope**.

Implemented milestones:

1. Shared canonical StableHLO compilation service, used by AOT and JIT misses.
2. Validated persistent artifact cache with atomic publication and recovery.
3. Opt-in GGML FFN JIT, isolated synchronous compiler worker, persistent request
   index and reuse of loaded modules in the existing VM/device.

P1 weight-layout preprocessing and P2 M buckets are not implemented here.
No upstream llama.cpp patch or scheduler observer was added. The locked
llama.cpp checkout remains clean at
`9fba51d2c9a1b5cb5f0c47986c031334e2d7663a`.
The fixture is `models/stories260K-f32.gguf`, SHA-256
`270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d`.

## Real-model evidence

Primary report: `build/m5-jit-arp2lua0/report.json`.

| Measurement | Cold cache | New process, populated cache |
| --- | ---: | ---: |
| Exact FFN profiles requested | 2 (M=5 and M=1) | 2 |
| Worker processes | 2 | 0 |
| Actual `iree-compile` invocations | 2 | 0 |
| Persistent cache hits | 0 | 2 |
| Loaded FFN module hits | 18 | 18 |
| Complete FFN calls | 20 | 20 |
| Original FFN member kernels bypassed | 80 | 80 |
| Independent matmul calls | 84 | 84 |
| Total VM calls | 104 | 104 |
| Coral dispatches / instruction samples | 256 / 256 | 256 / 256 |
| Simulator instructions | 1,504,628 | 1,504,628 |
| FFN weight preparations / reuse hits | 5 / 15 | 5 / 15 |
| Host numerical executable loads / failed dispatches | 0 / 0 | 0 / 0 |

`cold.exec.trace` records two worker launches and exactly two compiler
executions. `warm.exec.trace` contains only the model runner's own exec;
a disk hit does not start Python, MLIR, a compiler, or an artifact-check process.
C++ performs its own artifact validation before module loading.

Cold/warm logits are exactly equal. Maximum absolute error against GGML CPU:
`8.58306884765625e-6` under the frozen `1e-4 + 1e-4 * abs(reference)` bound.
Changing input token values while preserving shapes produces different
outputs, still agrees with CPU (maximum absolute error
`8.590519428253174e-6`), and starts zero workers/compilers.
The JIT-generated canonical hashes match the accepted M4 AOT semantic hashes.

The VM/device remains in the parent while the worker runs. The trace and
counters show seven module loads total, repeated execution through the same
backend, independent per-layer prepared weights, and no member replay.

## Failure and scheduler gates

All controls are retained under the primary evidence directory:

| Control | Observed result |
| --- | --- |
| Corrupt one cached VMFB | One worker, one recompile, one quarantine recovery; unchanged logits |
| Broken request index | One worker reimports, zero compiles; existing canonical artifact reused |
| Stale toolchain fingerprint | Rejected before worker launch or stale package use |
| Worker exits with an error | Graph fails and publishes no completion marker |
| Worker timeout | Graph fails; worker process group and child are killed |
| Retry the same failed backend | Rejected without relaunching a worker |
| Mutable weights | CPU comparisons pass across changing payloads; no stale prepared weights |
| Immutable weights | Independent layers retain independent snapshots and reuse them |
| Equivalent FFN spellings | Four additional import workers, zero extra compiles; still two canonical entries |
| External consumer/view, callbacks, callback subdivisions | Original members retained; no JIT request |
| Metadata changed after planning | Live guard rejects fusion; no JIT request |
| FFN explicitly disabled | Original members retained; no JIT request |
| Reservation/allocation | No JIT requests or worker launches |

The mutable/immutable modes each execute 18 scheduled graphs three times,
covering three FFN spellings, two M profiles, chained regions, multiple
CPU/Coral splits, destroyed reservation arenas and graph reuse. Together they
complete 180 whole-FFN calls and bypass 840 member kernels. Six rejection
modes add 36 graphs, each executed three times. This is 72 scheduled graphs /
216 successful scheduled executions, plus the failure/retry control.

Maximum error in the mutable/immutable scheduler cases:
`7.45058e-9` and `6.51926e-9` respectively.
Compiler snapshots contain metadata only; activation/weight payloads remain
in the inference process.

## Input and cache validation

| Gate | Result | Evidence |
| --- | --- | --- |
| Shared compilation service | AOT/direct identity agreement; 5 rejected requests; 14 runtime calls | `build/m5-service-6622t633/report.json` |
| Persistent cache | Cold/hit/fresh cache instance; 3 quarantine repairs; 2 compile failures; lock rejection; 8 identity dimensions; real changed-frontend miss | `build/m5-cache-fnala_ba/report.json` |
| GGML snapshot transport | 2 positives, 15 negatives; shared gate/up storage preserves positional ABI | `build/m5-snapshot-atjn2fsm/report.json` |
| Full AOT compiler regression | 14 normalized inputs, 10 rejected semantics, 6 compiled profiles; overwrite and failed publication rejected | `build/m5-aot-compiler-regression/aot-6m2_0ees/report.json` |
| Existing artifact contract | 1 positive, 54 negatives | `build/m5-regression-nwp_mitd/artifact.log` |
| Existing M1/M2/M3 CTest | 5/5 pass on rebuilt targets | `build/m5-regression-nwp_mitd/ctest.log` |
| Existing M4 backend | 84 graphs × 3 runs, 11 invalid catalogs | `build/m5-regression-nwp_mitd/backend/backend-y7kebd32/report.json` |
| Existing M4 model | 5 paired AOT member/region runs pass | `build/m5-regression-nwp_mitd/model/model-8t0_9yv1/report.json` |

The snapshot negatives cover schema/unknown fields, numeric wire types,
profile/type/stride mismatches, unsupported operations/parameters, cycles,
missing/duplicate members, wrong roles, duplicate JSON keys, malformed data
and oversized input. The worker feeds accepted metadata through the existing
per-op importer, standard verifier and FFN admission.

An initial AOT regression invocation used an older 11-case M3 fixture set;
the harness rejected the incomplete test set. The completed gate above uses
the full 14-case M3 acceptance fixtures. Both attempts remain on disk.

## Compiler-free AOT remains intact

`build/m5-deployment-results/deployment-yjmexpak/report.json` passes the
same physical compiler-absence gate as M4 using the new model runner:

- 27 isolated processes; runners use UID 65534 with no capabilities/network.
- Actual launch attempts for 15 compiler/tool entry points fail with ENOENT.
- The model still mixes 84 matmuls and 20 complete FFNs.
- All nine standalone FFN profiles and standalone matmul pass.
- The runtime root contains no compiler, frontend, Python, shell, JIT
  configuration or worker script.
- Development compilers and all build caches remain intact.

## Performance scope

The following cold/warm runs use `strace` for process evidence. They are single
diagnostic samples, not a benchmark or hardware latency claim.

| Host measurement | Cold | New process, populated cache |
| --- | ---: | ---: |
| Entire runner process | 37.406 s | 9.412 s |
| Initial toolchain fingerprint verification | 9.112 s | 9.084 s |
| JIT resolution (both profiles) | 27.970 s | 10.178 ms |
| VM module load time (all modules) | 3.832 ms | 3.670 ms |
| Total VM invocation time | 148.596 ms | 147.469 ms |

The cache-only tiny-profile test, without process tracing, measured an
8.925 s miss and 6.42 ms validated hit, excluding cache object/environment
fingerprint initialization. A separate untraced model smoke run measured
7.029 s of initial fingerprint verification and 22.162 s for both JIT
resolutions (`build/m5-jit-smoke-k9jns2o2/`).

The fresh-process fingerprint cost is substantial because the complete pinned
toolchain is checked, including the roughly 1.3 GiB `libIREECompiler.so`.
It occurs once per JIT client and does not recur on warm decode or loaded-module
hits. A persistent cache hit is therefore not equivalent to a millisecond
end-to-end startup. There is no new claim of NPU numerical speedup.

## Reproduction and retained artifacts

See [the JIT architecture and run guide](m5-jit.md). For this checkout, the main
test command was:

~~~bash
source env.sh
python tests/ggml/verify_ffn_jit.py \
  --matmuls build/m2-artifacts/model-aot-c2u8iuxg \
  --pointwise build/m3-pointwise-artifacts/pointwise-xc76cmwd \
  --aot build/m4-aot-results/aot-cvoknmn_
~~~

Every invocation creates new evidence directories. The validated cache,
configurations, generated packages, worker logs, imports, canonical IR,
manifests, traces and model outputs remain in ignored `build/`. No upstream
compiler rebuild or cache cleanup is required to replay the current artifacts.
