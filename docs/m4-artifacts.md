# M4: complete FFN artifacts and runtime

## Step 1: artifact contract

Status: contract and loader validation implemented. A valid manifest is not
proof that bytes are executable; VM reflection and execution are later gates.

The wire format is [ffn-manifest.schema.json](../artifacts/ffn-manifest.schema.json).
The C++ validator additionally enforces role order, exact shape relations,
signed 32-bit tensor byte bounds, reflection identity, the supported target
and build compatibility. JSON Schema alone cannot establish those relations.

A package contains:

| File | Purpose |
| --- | --- |
| `manifest.json` | Version, completion state, FFN ABI, program identity, toolchain, target, integrity and placement summary |
| `region.stablehlo.mlir` | Verified, normalized standard computation; weights remain arguments |
| `region.vmfb` | Compiled IREE module, with `coral.m4.ffn.abi` function reflection |
| `region.metadata.json` | Source correspondence and diagnostic provenance, outside semantic identity |
| `placement.json` | Hashed per-dispatch device placement and matching summary |

Exactly four F32 dense row-major inputs are required, in activation/gate/up/down
order, followed by one output. The schema, program pipeline and numeric policy
are independently versioned. No M1/M2 module is implicitly an FFN artifact.

The reflection value binds ordered tensor roles, dimensions, numerical policy,
runtime compatibility digest, semantic program digest and pipeline version.
It excludes weights, activation values, graph IDs and machine paths. The
compiler embeds this value; the runtime must compare it before VM context
initialization. Artifact digests establish consistency, not publisher authenticity.

Compatibility is embedded at CMake configure time from the dependency lock,
target profile and pinned CRT/linker/runtime source hashes. Version 1 supports
Linux x86_64, MPACT, the locked RV32 ILP32 feature set and default 8 KiB ITCM /
32 KiB DTCM with 4 KiB stack and 4 KiB reserve. A different profile requires
an explicitly supported configuration, not a manifest edit. Compiler and actual
linker/CRT binary digests are also recorded for provenance.

The loader rejects unknown fields/versions, duplicate JSON keys, incorrect
numeric wire types, mismatched roles/shapes/target/options, incomplete packages,
inconsistent placement and bad SHA-256 values. Fixed filenames prevent manifest
path traversal. Member symlinks and nonregular files are rejected; file sizes
and JSON nesting are bounded. Validated VMFB bytes are retained as an owned
snapshot so execution never reopens an unchecked file.

`nlohmann/json` and the public-domain SHA-256 implementation come from the
locked llama.cpp vendor directory. They are compiled without linking GGML,
MLIR or the compiler. Their upstream licenses remain applicable.

~~~bash
source env.sh
cmake --build build/m1 --target coral-ffn-artifact-check -j4
python tests/artifacts/verify_ffn_artifact.py \
  --checker build/m1/coral-ffn-artifact-check
# Inspect the exact supported contract and target:
build/m1/coral-ffn-artifact-check --describe 1 64 172
~~~

Evidence: `build/m4-contract-results/contract-m52fub4u/report.json`.
One valid contract-only fixture and 54 negative cases pass. These fixture VMFB
bytes intentionally are not executable: this milestone tests integrity and
compatibility before the IREE loader, not device execution.
