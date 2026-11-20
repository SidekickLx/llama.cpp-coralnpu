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
