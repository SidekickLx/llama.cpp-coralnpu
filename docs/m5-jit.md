# M5: shared compilation, artifact cache and JIT

## Compilation contract

The common entry point is
`compile_region(canonical_ir, profile, target, numeric_policy)` in
`compiler/region_service.py`. `profile` is the exact (M, D, F) tuple.
`canonical_ir` contains normalized standard StableHLO, with four positional
F32 inputs and external weights. `target` is the complete runtime compatibility
object; the numeric policy remains `f32-no-fast-math-v1`.

The AOT CLI first calls `canonicalize_region` on actual imported StableHLO.
That operation uses the existing verifier and FFN semantic admission.
The common compile entry rechecks canonicality, compiles, verifies Coral-only
placement, constructs the existing M4 manifest and invokes the C++ artifact
checker before atomic publication. Unknown targets or policies are rejected.
Diagnostics and original source remain in sibling evidence directories.
M4 packages and runtime ABI remain compatible.

JIT will use this same entry on a persistent-cache miss. Neither a shape-only
template nor a second compilation pipeline is introduced.

The first service gate passed: AOT and direct-service callers produced matching
semantic identity, contract, compiler identity and reflection; five invalid
target/policy/shape/semantic/canonicality requests were rejected, and the
result executed 14 FFN calls across two device lifetimes.
Evidence: `build/m5-service-6622t633/report.json`.
Reproduce with `python tests/compiler/verify_region_service.py PATH_TO_IMPORTED_TINY_MLIR`.

## Persistent cache

`compiler/region_cache.py` stores completed packages at `entries/<SHA-256>`.
The key includes canonical StableHLO, exact profile/dtype/layout/ABI, full
target compatibility, numerical policy, pipeline version, compiler/frontend/
checker/Python/service hashes and linker/CRT hashes. Paths, graph IDs, tensor
addresses, weights and activation values are excluded.

Every persistent hit checks identity and compiler provenance, then runs the
same C++ artifact validator used by AOT. A miss invokes `compile_region`.
The entire package, including cache identity, is atomically renamed into
`entries/` only after validation. A damaged entry is moved to `quarantine/`;
compilation errors and incomplete work remain in `staging/`. No build output
is deleted. A failed compilation cannot create a hit or a permanent negative
cache entry; a later request may retry.

A nonblocking writer lock rejects overlapping cache writers. Concurrent
request deduplication and asynchronous compilation remain future work.
Read hits do not acquire the writer lock. Digests provide consistency, not
authentication of artifacts obtained from an untrusted publisher.

Reproduce: `python tests/compiler/verify_region_cache.py PATH_TO_IMPORTED_TINY_MLIR`.

The cache gate passed, including three quarantine/rebuild cases, two failed
compilation retries, ignored interrupted staging, writer contention, eight key
invalidation dimensions, and a real changed-frontend compile into a new entry.
The actual `libIREECompiler.so` and tool dependency closure are fingerprinted.
The tiny-profile miss took 8.925 s; the validated hit took 6.42 ms (excluding
cache-object/environment fingerprint initialization). These are host diagnostic
times, not NPU hardware latency.
Evidence: `build/m5-cache-fnala_ba/report.json`.

## Runtime architecture

JIT is explicitly enabled through `--jit-config`. The default remains AOT.
The application links neither MLIR nor the IREE compiler library.

~~~mermaid
flowchart TD
    A[Live GGML FFN and legality guards] --> B{Loaded module?}
    B -- yes --> R[IREE VM / HAL and Coral MPACT]
    B -- no --> C{Validated persistent request index and artifact?}
    C -- yes --> L[Validate VM reflection and load module]
    C -- no --> W[Separate synchronous Python worker]
    W --> I[Existing per-operator GGML to StableHLO importer]
    I --> V[StableHLO verifier and FFN semantic admission]
    V --> K{Canonical artifact cache hit?}
    K -- yes --> X[Validate existing package]
    K -- no --> P[Shared compile_region service]
    P --> X
    X --> D[Atomic artifact and request-index publication]
    D --> L
    L --> R
~~~

AOT catalog entries take precedence. A missing complete FFN package can use JIT
only after the same member capability, fusion, lifetime and live execution
checks as M4. Capability queries and graph reservation never invoke a worker.
Missing member support still routes through the original scheduler and prevents
fusion. This step does not add arbitrary-operator JIT, expand supported dtypes,
or make every new M automatically eligible without member coverage.

The C++ client serializes a bounded snapshot of the actual selected GGML nodes:
types, extents, strides, operation parameters and source-edge IDs. It contains
no tensor payloads, addresses, model names or source paths. Boundary producers
become function arguments. The worker reconstructs metadata-only GGML tensors
and calls the existing `ImportGGMLFFNFunction`. This transport is not a new
computational dialect, optimizer IR or evaluator. Standard StableHLO remains
the only compilation IR. The official reference interpreter remains an
external validation tool, not part of the inference process.

There are three code-cache levels:

1. A device-local loaded-module map reuses the exact FFN program within the
   running backend. Every execution still passes live guards.
2. A persistent request index maps a GGML snapshot plus pinned environment
   identity to its verified canonical artifact. A fresh-process hit validates
   the artifact in C++ without starting a worker, Python or a compiler.
3. The canonical artifact cache deduplicates equivalent GGML spellings after
   import/normalization. A new source spelling may launch a worker to import,
   yet reuse an existing canonical artifact without invoking `iree-compile`.

The prepared-weight cache is separate from these code caches. Different
same-shaped layers share code while retaining independent immutable weight
snapshots. Changing activation values never changes code identity; their
current contents are uploaded for every invocation.

A miss starts a short-lived worker with `posix_spawn` and explicit arguments,
not a shell. The parent retains its VM/device and synchronously waits. The
worker imports, looks up/compiles, validates and publishes; the parent checks
the response key, identity, integrity and compatibility, then the normal FFN
loader verifies VM reflection/signature. Only then is the program executed.
Worker failure or timeout fails the graph with no member replay or partial Y;
the backend rejects subsequent retries. Timeout kills the worker process group,
including compiler descendants. Logs and failed staging remain available.

An invalid indexed cache entry is recoverable: the worker quarantines it and
recompiles. In contrast, an explicitly listed invalid AOT package remains a
deployment error, as in M4. Publication provides atomic visibility. This step
does not promise fsync-based power-loss durability; lost/incomplete entries
are rejected and rebuilt after restart.

## Configuration and running

Use the existing README setup and member catalogs, then:

~~~bash
source env.sh
bash scripts/build_m3.sh
cmake --build build/m1 --target coral-llama-runner coral-ffn-region-test -j4

python compiler/jit_worker.py configure \
  --cache build/m5-cache \
  --output build/m5-jit.json

build/m1/coral-llama-runner \
  --mode coral \
  --model models/stories260K-f32.gguf \
  --artifacts PATH_TO_MEMBER_CATALOG \
  --observation backend \
  --jit-config build/m5-jit.json \
  --output build/my-first-jit-run
~~~

Use a member catalog without `ffn.tsv` to exercise JIT; existing FFN entries
are loaded as AOT. The first model run needs exact M=5 and M=1 profiles.
A second run with a new output directory reuses the same persistent cache.

The generated local configuration records absolute tool/cache paths and
content identities. Keep it under ignored `build/` or `.local/`. The runtime
hashes the pinned toolchain, including `libIREECompiler.so`, once when the first
JIT client is needed; stale configuration is rejected before using cached code.
After rebuilding/changing tools or compiler service code, run `configure` with
a new output filename. Configuration, tools and source are immutable for the
lifetime of a client; hot-swapping a compiler within a running session is not
supported. Digests are not publisher authentication.

This complete fingerprint check currently adds seconds to fresh-process JIT
initialization. It is measured separately from artifact lookup, compilation,
module load and VM invocation. A warm decode does not repeat it.
Compiler-free AOT execution uses no JIT configuration and performs none of
these toolchain checks.

Counters distinguish `jit_worker_launches`, actual `jit_compilations`,
`jit_disk_hits`, `jit_recoveries`, and `ffn_module_cache_hits`.
`jit_initialization_ms` covers environment verification; `jit_ms` covers
request lookup/worker/validation, excluding VM module load and inference.

## Acceptance

Run the real-model and scheduler gate with retained member/AOT evidence:

~~~bash
python tests/ggml/verify_ffn_jit.py \
  --matmuls PATH_TO_MATMUL_CATALOG \
  --pointwise PATH_TO_POINTWISE_CATALOG \
  --aot PATH_TO_M4_AOT_EVIDENCE
~~~

The gate traces actual process execution, compares CPU/JIT logits, checks
cross-process hits and changed activations, injects corrupt packages/indexes,
stale environments, failed workers and timeouts, and repeats scheduler
legality/lifetime tests with JIT enabled. P1 weight-layout optimization and
P2 M buckets remain the next M5 steps.

The completed steps 1–3 acceptance, measured costs and raw evidence locations
are recorded in [the test report](m5-jit-test-report.md).
