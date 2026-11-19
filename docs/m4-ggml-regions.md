# M4: Complete FFN execution in GGML

The out-of-tree Coral backend can execute a whole admitted FFN through
`runtime::FFNModule`. This is one VM invocation with four inputs and one output;
the compiled program may contain several Coral dispatches. The GGML graph and
the pinned llama.cpp source remain unchanged.

## Artifact selection and fallback

Keep the existing `profiles.tsv` matmul and `pointwise.tsv` member catalogs.
Add an optional `ffn.tsv` alongside them:

~~~text
coral-m4-ffn-v1
1 64 172 real-decode
5 64 172 real-prefill
~~~

Columns are M, D, F and one local child directory containing the complete
[validated FFN package](m4-artifacts.md). The loader rejects duplicate shapes,
invalid dimensions, nested paths, symlink directories and missing directories.
Registration reads the index; lazy module loading validates the complete
manifest, VMFB snapshot, ABI, hashes, placement and runtime compatibility.
Only one canonical FFN semantic contract/numeric policy is supported.

A missing FFN profile, `Options::enable_ffn = false`, or a failed legality/live
guard retains individual member execution. Missing member artifacts still
route unsupported members through the upstream scheduler (usually CPU) and
prevent a complete region. `supports_op` remains a pure query of **member**
capability; a full FFN package does not grant standalone operation support.
A listed but invalid package or failed invocation fails the graph and poisons
the backend; it never silently replays the members or publishes partial Y.

## Plan ownership and execution

The [planner](m3-fusion.md) checks global uses, views, output flags, source
roles, dimensions, parameters, contiguous members, member support and the
explicit no-evaluation-callback promise. Its allocation dependencies keep all
four inputs alive through the down projection.

A backend serves one scheduler at a time on one host thread. Optimization
collects all splits of a parent graph in one generation. A new optimization
batch after execution, or a different parent graph, discards old plans.
Each entry owns metadata copies and opaque node identities; it never retains
a reference to the scheduler split or its node array. No module is loaded
during reservation.

The pinned upstream scheduler assigns a fresh nonzero split UID after
optimization/allocation planning. On first compute, the backend matches the
exact split object, node sequence and use-count table, then binds that UID.
Later compute calls must have the same UID. Callback views have UID zero and
always use members, even when an application incorrectly declares callbacks
absent. Direct backend compute without a matching optimization also uses
members. Changes requiring rescheduling must go through the scheduler's normal
reset/allocation lifecycle; callers must not forge graph UIDs.

Before every invocation, including graph reuse, live guards recheck metadata,
global uses, capabilities, generation and boundary storage. Saved members
must first be found in the current live graph before dereferencing snapshots.
The region executes at its first member, after all boundary inputs are ready.
Its members are contiguous and have no observable intermediate consumers.
After a validated complete result is copied into Y, the member loop advances
past the entire region. Intermediate tensors are not materialized.

## Caches and evidence

Modules are cached per M/D/F in one device context. The fixed versioned
semantic contract and artifact validation establish the program identity.

FFN weight snapshots include the ordered gate/up/down tensor, buffer, data
identities and D/F in their cache key. Different layers cannot share weights
merely because shapes match. Reuse requires the explicit immutable-weight
promise, leaf tensors and WEIGHTS buffer usage. All weight storage must outlive
the backend. Without that promise, all three weights are freshly prepared
on every invocation, even when the buffers carry the WEIGHTS tag.

`ffn_calls` counts completed whole-program VM calls;
`ffn_members_skipped` counts original kernels bypassed.
`ffn_weight_preparations` counts three-weight sets, not tensors.
`ffn_missing_artifacts` is a planning count and can include reservation.
`ffn_guard_rejections` counts failed live checks of matched plans.
`ffn_region` trace records report actual program hash, member names, shape,
dispatches, instructions, cache reuse and VM invocation time. No synthetic
per-member execution events are emitted for unmaterialized intermediates.

## Running and testing

Use `coral-llama-runner` with `--observation backend` and a combined catalog.
`--ffn auto` is the default; `--ffn off` provides member execution for comparison.
The default `--observation nodes` installs per-node callbacks and disables
fusion, so it remains useful for inspecting intermediate tensors.

~~~bash
source env.sh
cmake --build build/m1 --target coral-llama-runner coral-ffn-region-test -j4
python tests/ggml/verify_ffn_region_backend.py \
  --matmuls build/m2-artifacts/model-aot-c2u8iuxg \
  --pointwise build/m3-pointwise-artifacts/pointwise-xc76cmwd \
  --ffn build/m4-aot-results/aot-cvoknmn_
~~~

The fixture paths above refer to retained local evidence. For a new workspace,
produce member artifacts using the M2/M3 guides and complete FFN packages using
`m4-artifacts.md`. The test copies packages to a fresh output directory and
preserves prior evidence; it does not compile or download artifacts.

See the [region acceptance report](m4-ggml-region-test-report.md) for the
real-model member/region comparison, execution evidence and measured timings.
The separate `tests/ggml/verify_ffn_regions.py` suite uses the same three
artifact-directory arguments as the backend suite above.
