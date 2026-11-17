# M3: upstream graph integration

The project uses the pinned llama.cpp checkout without project patches.
The custom scheduler graph observer, internal llama context accessor,
patch application script and `--export-graph` option have been removed.

## Observation boundaries

Standard `graph_optimize` receives the backend's scheduled split before
allocation. The view retains the parent graph's use counts, which upstream
`ggml_can_fuse_subgraph_ext` uses to reject externally consumed intermediates.
The `add_alloc_dep` callback can preserve inputs until a fusion anchor.

Standard `graph_compute` receives the actual execution range. Evaluation
callbacks may subdivide a split, so a planner must not assume that an earlier
optimization callback guarantees a complete region at execution time.

The public llama `cb_eval` callback and the Coral backend dispatch trace retain
numerical and execution evidence. They do not reconstruct the exact original
DAG before scheduler rewriting or enumerate every backend's split.

## Historical evidence

The earlier observer-based test remains in Git history at `dde700c`.
Its locally retained results are under `build/m3-graph-results/graph-dny5co1e`.
Those results describe that historical instrumented build and are not evidence
that the removed observer is still available.

## Unpatched acceptance

Build incrementally and run from the project root:

~~~bash
source env.sh
cmake --build build/m1 --target coral-llama-runner coral-adapter-test coral-ggml-import-test coral-model-import-test coral-ffn-planner-test -j4
python tests/ggml/verify_unpatched.py \
    --build-dir build/m1 \
    --check-library build/m3/libcoral-importer-check.so \
    --model models/stories260K-f32.gguf \
    --artifacts ARTIFACT_DIR \
    --output-root build/m3-unpatched-results
~~~

ARTIFACT_DIR is the existing M2 directory containing profiles.tsv and VMFBs.
The harness checks the pinned llama.cpp revision, its clean working tree and
the configured source path before running M2 and both importer suites.
The build must precede the test; source identity alone does not prove binary
provenance.

All three suites passed at
`build/m3-unpatched-results/unpatched-dtxlq54s`:
M2 CPU/Coral logits and six failure cases, 12 importer numerical cases and
99 rejections, and four real-model prefill/decode node imports.
Logs, values, MLIR and reports remain in that ignored evidence directory.

Fusion legality, member capability and full FFN device execution have their
own acceptance gates; removal of the observer does not establish them.

The final integrated run is described in
[the unpatched test report](m3-unpatched-test-report.md).
[FFN fusion checks](m3-fusion.md) now use the upstream optimization,
global-use and allocation-dependency interfaces.
