#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify complete FFN import, positional roles, shared identities and tail shapes."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

POSITIVE = [
    "ffn-region-split", "ffn-region-expanded", "ffn-region-expanded-reversed",
    "ffn-region-tail-split", "ffn-region-tail-expanded",
    "ffn-region-decode", "ffn-region-prefill",
    "ffn-region-shared", "ffn-region-shared-expanded",
    "ffn-region-gate-up-swapped", "ffn-region-up-first",
]
NEGATIVE = {
    "bad-region-contract": "ffn-contract", "bad-region-policy": "ffn-contract",
    "bad-region-shape": "ffn-contract", "bad-region-roles": "ffn-roles",
    "bad-region-weight-producer": "ffn-roles",
    "bad-region-missing": "ffn-members", "bad-region-duplicate": "ffn-members",
    "bad-region-order": "dependency", "bad-region-cycle": "ffn-pattern",
    "bad-region-output": "ffn-pattern", "bad-region-parameter": "parameters",
    "bad-region-shared-id": "identity",
}
ROLES = ("activation", "gate_weight", "up_weight", "down_weight")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runner", "check-library", "reference-runner", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="ffn-import-", dir=args.output_root.resolve()))
    print(f"Evidence: {work}", flush=True)
    reports = []
    for name in [*POSITIVE, *NEGATIVE, "bad-region-payload-swap"]:
        output = work / name
        with (work / f"{name}.stdout.txt").open("w") as out, (work / f"{name}.stderr.txt").open("w") as err:
            result = subprocess.run(
                [str(args.runner.resolve()), str(args.check_library.resolve()), name, str(output)],
                stdout=out, stderr=err, timeout=90, check=False)
        text = (work / f"{name}.stderr.txt").read_text()
        expected = 2 if name in NEGATIVE else 1 if name == "bad-region-payload-swap" else 0
        if result.returncode != expected:
            raise RuntimeError(f"{name}: exit {result.returncode}: {text[-4000:]}")
        report = json.loads((output / "report.json").read_text())
        if name in NEGATIVE:
            if report["status"] != "rejected" or f"[{NEGATIVE[name]}]" not in text:
                raise RuntimeError(f"{name}: incorrect rejection: {text}")
            if (output / "imported.mlir").exists(): raise RuntimeError("failed import published module")
        elif name == "bad-region-payload-swap":
            if report["status"] != "failed" or not report["comparisons"][0]["failing_indices"]:
                raise RuntimeError("swapped payloads did not trigger numerical mismatch")
        else:
            if report["status"] != "passed" or len(report["comparisons"]) != 1:
                raise RuntimeError("incomplete FFN comparison")
            ir = (output / "imported.mlir").read_text()
            if ir.count("stablehlo.dot_general") != 3 or 'func.func @ffn(' not in ir:
                raise RuntimeError("not a complete FFN function")
            if 'coral.ffn_abi = "coral.ffn.f32.v1"' not in ir or "custom_call" in ir:
                raise RuntimeError("invalid standard FFN ABI")
            for role in ROLES:
                if f'coral.ffn_role = "{role}"' not in ir:
                    raise RuntimeError("missing positional role")
            inputs = json.loads((output / "inputs.json").read_text())["inputs"]
            if len(inputs) != 4: raise RuntimeError("shared identity collapsed an ABI argument")
            reports.append(report)
    def values(name):
        return json.loads((work / name / "reference-output.json").read_text())["outputs"][0]["values"]
    reference = values("ffn-region-split")
    for name in ("ffn-region-expanded", "ffn-region-expanded-reversed", "ffn-region-up-first"):
        if values(name) != reference: raise RuntimeError("equivalent FFN spelling changed output")
    if values("ffn-region-shared") != values("ffn-region-shared-expanded"):
        raise RuntimeError("shared weights differ across FFN spellings")
    if max(abs(a-b) for a, b in zip(values("ffn-region-gate-up-swapped"), reference)) < 1e-4:
        raise RuntimeError("same-shaped gate/up weights did not distinguish roles")
    # A shared source identity must not specialize away one positional weight.
    # Replay the shared-source function with the distinct-weight case's inputs;
    # compare with that case's real GGML CPU result, not another implementation.
    counterfactual = work / "shared-distinct-arguments.json"
    with (work / "shared-distinct.stdout.txt").open("w") as out, (work / "shared-distinct.stderr.txt").open("w") as err:
        subprocess.run(
            [str(args.reference_runner.resolve()),
             str(work / "ffn-region-shared/imported.mlir"), "ffn",
             str(work / "ffn-region-split/inputs.json"), str(counterfactual)],
            stdout=out, stderr=err, timeout=90, check=True)
    actual = json.loads(counterfactual.read_text())["outputs"][0]["values"]
    expected = json.loads((work / "ffn-region-split/ggml-output.json").read_text())["outputs"][0]["values"]
    if len(actual) != len(expected) or any(abs(a-b) > 1e-4 + 1e-4 * abs(b)
                                          for a, b in zip(actual, expected)):
        raise RuntimeError("shared identity collapsed independent FFN arguments")
    report = {"status": "passed", "shared_argument_independence": True,
              "positive_cases": len(POSITIVE),
              "rejected_cases": len(NEGATIVE), "payload_swap_detected": True,
              "max_abs_error": max(r["comparisons"][0]["max_abs_error"] for r in reports),
              "cases": reports}
    (work / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}), flush=True)
if __name__ == "__main__":
    main()
