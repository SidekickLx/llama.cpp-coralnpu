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
