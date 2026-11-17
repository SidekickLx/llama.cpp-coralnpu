# M3: FFN legality on unmodified GGML

`src/planner/ggml_fusion.{h,cc}` uses the pinned upstream backend interfaces.
It recognizes three matmuls with split SwiGLU, or the expanded SiLU + multiply
form with either multiply operand order. Roles come from edges and parameters.
It creates process-local preparation plans only; no node is skipped and no
fused FFN is executed by this component.

## Planning before allocation

`PlanFFN` runs on the scheduled graph view supplied to `graph_optimize`.
A non-match, structured rejection or checked plan is returned. It checks:

- Static dense F32 dimensions, byte bounds, source slots and parameter words.
- Separate leaf weights, exact gate/up activation identity, topological order,
  distinct and contiguous members in one backend split.
- Original use counts retained by the view and
  `ggml_can_fuse_subgraph_ext`, including intermediate graph outputs.
- Explicit view/alias rejection and any already-bound boundary overlap.
- An explicit no-evaluation-callback policy and known support/artifacts for
  every member. Unknown capability is not permission to fuse.
- Availability of the upstream pre-allocation `add_alloc_dep` hook.

Only after all checks pass are unique X/weight inputs kept alive until the
down projection. Rejections add no allocation dependencies and change no
nodes, placement or tensor parameters. Caller-provided capability functions
must be pure queries of already-prepared backend support.

## Live guards

`CheckFFNExecution` validates the actual execution range, source metadata,
global use counts, capability, callback policy, generation and allocated
boundary ranges. It runs after allocation and must run for each future fused
invocation, including graph reuse. The owner increments the generation after
graph, schedule or allocation rebuilds and keeps borrowed tensors alive.

The plan carries no VMFB, prepared weight ownership or runtime authority.
M4 must add artifact/device compatibility and prepared-binding guards before
executing anything. Changing activation values alone does not stale a plan.

A callback can subdivide a scheduled split into smaller compute calls.
Even if the caller fails to update its observation policy, an incomplete
execution range cannot authorize the whole FFN. The default policy disables
fusion until the application explicitly declares callbacks absent.

## Tests

~~~bash
source env.sh
cmake --build build/m1 --target coral-ffn-planner-test -j4
ctest --test-dir build/m1 -R m3_ffn_planner --output-on-failure
~~~

All 33 cases passed. Evidence: `build/fusion-nso7lgtd`.

The tests construct real GGML graphs and use the unmodified scheduler and CPU
allocator. A test-only CPU backend hook exercises the production planner with
the actual upstream `graph_optimize`/`add_alloc_dep` callbacks. It always
executes original CPU nodes and is not evidence of fused Coral execution.

Coverage includes external consumers (also in another actual backend split),
external views, callback subdivision without dropped callbacks, graph reuse,
shape/stride/type/parameter failures, shared weight identity, missing member
capability/artifacts, stale metadata/generations and live output/input aliasing.

For lifetime validation, the real allocator receives the dependencies and X
is preserved through the down projection on both executions. A counterfactual
test discards those dependencies; the allocator reuses boundary storage and
the live guard rejects the resulting overlap. The cross-backend fixture uses
two CPU-backed test devices with distinct capabilities and shared host storage;
device-local transfer boundaries are not covered by that fixture.
