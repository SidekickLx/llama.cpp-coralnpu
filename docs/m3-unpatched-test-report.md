# Unpatched llama.cpp and FFN planning acceptance

Date: 2026-11-17. Branch: `codex/m3-ffn-ir`.

The local llama.cpp checkout is exactly the pinned revision
`9fba51d2c9a1b5cb5f0c47986c031334e2d7663a` with a clean working tree.
The project scheduler patch, observer exporter, application script and
internal `llama_context::get_sched()` dependency have been removed.

The incremental build reused existing dependencies, VMFBs and caches.
No compiler, simulator or model artifact was cleaned.

## Results

| Check | Result |
| --- | --- |
| M1 runtime, M2 adapter and planner CTest | 5/5 passed |
| FFN planner and real scheduler checks | 33 passed |
| M2 CPU fixed-token replay | Exact |
| M2 CPU/Coral logits | Maximum absolute error 7.1525574e-06 |
| M2 failure cases | 6 passed |
| GGML importer numerical / rejection cases | 12 / 99 passed |
| Real-model prefill/decode imports | 4 passed; maximum absolute error 2.8610229e-06 |

Numerical tolerance remains `abs(actual-reference) <= 1e-4 + 1e-4*abs(reference)`.

## Device and planning evidence

| Counter from the M2 Coral run | Value |
| --- | --- |
| Executed Coral matmuls | 144 |
| Successful Coral dispatches | 256 |
| Kernel instructions | 1447460 |
| Host executable load attempts | 0 |
| Graph optimization callbacks | 156 |
| FFN candidates inspected | 30 |
| FFN preparation plans | 0 |
| FFN rejections | 30 |

The diagnostic runner enables per-node evaluation callbacks. Those candidates
are rejected; current member capability also does not provide standalone
pointwise Coral execution. Every original operation remains in the ordinary
execution path. These counters are not evidence of fused FFN execution.
Optimization counters can include graph reservation and rebuilding.

Aggregate IREE invocation time was 136.872 ms in this run. This is a
single diagnostic simulator host-time measurement, not a warmed performance
distribution or a hardware speed claim. The planner test uses CPU-backed
probe devices to validate scheduler/allocator behavior, not a substitute NPU.

## Reproduction and retained evidence

Follow [the unpatched acceptance commands](m3-graph-export.md#unpatched-acceptance)
and [the planner tests](m3-fusion.md#tests).

Final evidence: `build/m3-unpatched-results/unpatched-gd3qfntx`.
It contains `report.json`, `fusion.json`, suite stdout/stderr and nested M2/
importer reports with logits, node traces, dispatch counts, MLIR and values.
CTest output is retained at `build/m3-upstream-ctest.log`.

The lifetime test registers actual upstream allocation dependencies and proves
that X remains intact through the down projection on repeated graph execution.
Discarding dependencies in the test control produces boundary storage overlap,
which live validation rejects. Cross-backend consumers, views, graph outputs,
callback subdivisions, stale generations/parameters and changed capability
also reject safely.

Full FFN member support/placement is the next gate. Fused Coral VMFB execution,
artifact compatibility and three-way numerical comparison remain M4.
