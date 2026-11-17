# M3: observing a real GGML graph

The runner now accepts `--export-graph`. For each successful prefill/decode
step it writes `graph-N.json` alongside the existing execution trace.
This is diagnostic metadata, not an executable IR or a fusion decision.

~~~bash
source env.sh
python scripts/apply_dependency_patches.py
cmake --build build/m1 --target coral-llama-runner -j4
build/m1/coral-llama-runner --mode cpu --model models/stories260K-f32.gguf --decode 3 --output build/m3-graph-example --export-graph
~~~

Use a new output directory on every run. ARTIFACT_DIR must be the existing
M2 artifact directory containing profiles.tsv and its compiled VMFBs. In Coral mode, supply the existing
`--artifacts ARTIFACT_DIR` argument as in the M2 walkthrough.

## Observation boundary

The pinned GGML scheduler rewrites cross-backend source edges before invoking
the existing per-node evaluation callback. That callback alone cannot recover
the original DAG. A small version-1 patch therefore provides two read-only
observation points:

1. Before splitting/allocation/edge rewriting: copy the original graph metadata.
2. Before each actual split executes: copy its backend, node order and rewritten
   dependencies. This fires on graph reuse as well as rebuild.

The exporter attaches through the pinned internal `llama_context::get_sched()`
accessor. That dependency and the patch must be reviewed when updating llama.cpp.
Callbacks never throw through C; errors are retained and reported by `Finish`.
A failed export prevents the runner's successful summary marker.

## Schema: `coral-ggml-graph-v1`

- `step`, `phase`: runner invocation coordinates.
- `graph_generation`, `graph_rebuilt`: original graph snapshot identity within
  this run. A reused graph retains its source snapshot.
- `source`: original graph node order and all reachable tensors, including
  constants, metadata operations, inputs, and alias/view dependencies.
- `splits`: ordered actual scheduler splits with backend names and their graph
  snapshots. A split snapshot's consumers are local to that snapshot's reachable
  tensor closure; use `source` for global original-graph consumer analysis.

Each tensor records a numeric ID, diagnostic name, operation, type, `ne`, byte
strides `nb`, raw integer op parameters, flags, contiguity, ordered `src` IDs,
`view_src`, byte view offset, reverse source consumers (including operand slot),
and optional buffer ID/type/capacity. Zero denotes an absent source/view.
Tensor IDs are assigned deterministically from node order, then dependencies,
and are scoped to a graph generation. Buffer IDs are process-local labels,
never addresses. Scheduler-created copies receive additional tensor IDs.
View aliases are represented explicitly by `view_src`, separately from the
source-consumer list. Source buffers may not yet be allocated.

No tensor payloads, pointer addresses, model paths, or environment paths are
exported. Custom operations with opaque pointer-bearing parameters are rejected.
Diagnostic names come from the model and are not proof of FFN semantics.

This observer supports the runner's single-microbatch invocation contract.
Multiple executed graphs within one step are rejected. It is opt-in, performs
host-side metadata work, and is not suitable for measuring baseline latency.
The execution trace remains the evidence of actual Coral dispatch; a split
observation alone does not prove successful execution.

## Validation

~~~bash
python tests/ggml/verify_graph_export.py --runner build/m1/coral-llama-runner --model models/stories260K-f32.gguf --artifacts ARTIFACT_DIR --output-root build/m3-graph-results
~~~

The real-model test checks original/rewritten edges, reverse consumers, views,
full node coverage, scheduler split membership, graph reuse, deterministic IDs,
and unchanged CPU logits. It also checks Coral results against CPU at M2
tolerance and confirms that FFN GLU remains on CPU with matmuls on Coral.
It discovers the five split SwiGLU shapes through dependencies, not tensor names.
This is a diagnostic acceptance test, not the production FFN matcher.

## Acceptance evidence

Validated with the locked stories260K F32 model, one prefill plus three decode
steps. The local evidence directory is
`build/m3-graph-results/graph-dny5co1e`; it contains per-run stdout/stderr,
graphs, traces, logits and `report.json`. These generated files remain ignored.

| Check | Result |
| --- | --- |
| Original graph per step | 189 nodes, 256 reachable tensors |
| CPU / Coral scheduler splits | 1 / 52 |
| Repeated CPU export | Identical metadata and bitwise-identical logits |
| Graph reuse | Decode steps 2 and 3 retain generation 2 |
| FFNs | Five, D=64, F=172 |
| Prefill M | 5 in layers 0–3; 1 in final layer |
| Decode M | 1 in every layer |
| Coral execution | 144 matmuls, 256 observed dispatches |
| Maximum CPU/Coral logit absolute error | 7.152557373046875e-6 |
| Existing M2 runner regression | Passed, including 6 negative cases |

The current Coral backend shares host-visible storage with CPU, so this model
needs no scheduler source-edge copies despite its 52 splits. The format records
rewritten edges when present; this real-model fixture does not exercise a
device-local copy boundary. No full FFN is fused or executed as a region yet.

Export timing includes metadata traversal/serialization and is diagnostic
overhead, not an inference performance benchmark. The M2 regression with export
disabled recorded 134.084 ms of aggregate IREE invocation time in this one run;
this is simulator host time and not a hardware latency claim.
