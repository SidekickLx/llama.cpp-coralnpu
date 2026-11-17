# M3: complete FFN semantic acceptance

## Fixed role importer

`ImportGGMLFFNFunction` takes exactly four ordered boundary roles:
X, gate weight, up weight and down weight, plus the complete topological
member selection and one output. It exports `ffn` with the declared F32
contract and role attributes. It verifies the requested M/D/F dimensions,
the exact edge-to-role correspondence, member completeness and identities.
The existing per-operation importer handles parameter, type, layout and SSA
validation, followed by MLIR/StableHLO verification.

The matcher is shared with the planner in `src/planner/ggml_ffn_pattern.*`.
It reads GGML metadata and makes no scheduling or execution decision.
No additional computational IR, custom operation or numerical evaluator is
introduced.

Repeated source weight identities keep separate positional function arguments.
Matmul operands are assigned by their validated consumer role. This preserves
both original tensor identity metadata and the general four-argument FFN
semantics. The generic `ImportGGMLFunction` API retains its unique-input rule.

## Synthetic acceptance

~~~bash
source env.sh
bash scripts/build_m3.sh
cmake --build build/m1 --target coral-ggml-import-test coral-ffn-planner-test -j4
python tests/compiler/verify_ffn_import.py \
  --runner build/m1/coral-ggml-import-test \
  --check-library build/m3/libcoral-importer-check.so \
  --reference-runner build/m3/coral-stablehlo-reference \
  --output-root build/m3-ffn-import-results
~~~

Coverage includes split SwiGLU and both expanded SiLU/multiply orders, tail
dimensions, prefill/decode, independent same-shaped weights, shared weights,
reversed gate/up roles and independent projection ordering. Rejections cover
contract version/policy/shape, wrong roles, non-leaf weights, incomplete or
duplicated members, dependency order/cycles, wrong output, unknown parameters
and inconsistent shared identities.

The numerical baseline is actual GGML CPU execution; the evaluator is the
official StableHLO interpreter. Swapping input weight payloads after CPU
execution must produce a numerical failure. The shared-identity function is
also evaluated with distinct gate/up argument values and compared with the
corresponding GGML CPU case, proving neither argument was specialized away.

This first acceptance milestone validates the complete function semantics on
synthetic inputs. Real-model region capture and aggregate M3 acceptance follow.

Synthetic evidence: `build/m3-ffn-import-results/ffn-import-cktynamd/report.json`.
All 11 numerical cases, 12 rejection cases, the payload-swap failure and the
shared-argument counterexample pass. Maximum absolute error is
`1.1175870895385742e-8`, with `atol=rtol=1e-4`. The original importer suite
(12 numerical / 99 rejection cases) and all 33 planner cases also pass after
the matcher extraction. Importer regression evidence:
`build/m3-importer-results/import-gv_vmaq6`.
