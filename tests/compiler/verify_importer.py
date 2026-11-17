#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run real GGML CPU graphs through the direct StableHLO importer."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

POSITIVE = ["matmul-small", "matmul-decode", "matmul-prefill",
            "matmul-explicit-f32", "matmul-shared"]
NEGATIVE = {
    "bad-dtype-input": "tensor-type", "bad-dtype-output": "tensor-type",
    "bad-quantized": "tensor-type", "bad-stride": "layout", "bad-view": "layout",
    "bad-view-offset": "layout", "bad-extra": "layout", "bad-batch": "shape",
    "bad-zero": "shape", "bad-overflow": "shape", "bad-contracting": "shape",
    "bad-result-shape": "shape", "bad-missing-source": "operands",
    "bad-extra-source": "operands", "bad-operator": "operator",
    "bad-dependency": "dependency", "bad-precision": "parameters",
    "bad-id-zero": "identity", "bad-id-duplicate": "identity",
    "bad-output-id": "identity", "bad-graph-id": "request",
}
for word in range(16):
    NEGATIVE[f"bad-param-{word}"] = "parameters"

POSITIVE += ["silu-ranges", "mul-ranges", "glu-asymmetric", "glu-reversed",
             "ffn-split", "ffn-expanded", "ffn-expanded-reversed"]
NEGATIVE.update({
    "bad-silu-activation": "parameters", "bad-silu-inplace": "layout",
    "bad-silu-extra-source": "operands", "bad-glu-swap": "parameters",
    "bad-glu-clamp": "parameters", "bad-glu-oai": "parameters",
    "bad-glu-packed": "operands", "bad-glu-shape": "shape",
    "bad-glu-missing-source": "operands", "bad-glu-dtype": "tensor-type",
    "bad-mul-broadcast": "shape", "bad-mul-result-shape": "shape",
    "bad-ffn-order": "dependency", "bad-ffn-self-cycle": "dependency",
})
for operator in ("silu", "mul", "glu"):
    for word in range(16):
        NEGATIVE[f"bad-{operator}-param-{word}"] = "parameters"


def run(args, work):
    reports = []
    for name in [*POSITIVE, *NEGATIVE]:
        output = work / name
        result = subprocess.run(
            [str(args.runner.resolve()), str(args.check_library.resolve()), name, str(output)],
            capture_output=True, text=True, timeout=60, check=False)
        (work / (name + ".stdout.txt")).write_text(result.stdout)
        (work / (name + ".stderr.txt")).write_text(result.stderr)
        rejected = name in NEGATIVE
        expected = 2 if rejected else 0
        if result.returncode != expected:
            raise RuntimeError(f"{name}: expected {expected}, got {result.returncode}: {result.stderr}")
        report = json.loads((output / "report.json").read_text())
        if rejected:
            if "[" + NEGATIVE[name] + "]" not in result.stderr or report["status"] != "rejected":
                raise RuntimeError(f"{name}: incorrect rejection: {result.stderr}")
            if (output / "imported.mlir").exists():
                raise RuntimeError(f"{name}: failed import published a module")
        else:
            if report["status"] != "passed":
                raise RuntimeError(f"{name}: failed numerical comparison")
            text = (output / "imported.mlir").read_text()
            for field in ("coral.ggml_graph_instance", "coral.ggml_graph_revision",
                          "coral.ggml_tensor_id", "coral.ggml_op_params"):
                if field not in text:
                    raise RuntimeError(f"{name}: missing source metadata {field}")
            if "custom_call" in text or "diagnostic-only" in text:
                raise RuntimeError(f"{name}: imported opaque semantics or relied on names")
            if name == "matmul-explicit-f32" and "MUL_MAT" not in text:
                raise RuntimeError("missing matmul source provenance")
            if any(item["failing_indices"] for item in report["comparisons"]):
                raise RuntimeError(f"{name}: nonempty failing indices")
            reports.append(report)
    # A reversed MUL preserves the expanded FFN, but reversing GLU gate/up
    # changes the program. Check both so symmetric fixtures cannot hide a bug.
    split = json.loads((work / "ffn-split/reference-output.json").read_text())["outputs"][-1]["values"]
    for name in ("ffn-expanded", "ffn-expanded-reversed"):
        expanded = json.loads((work / name / "reference-output.json").read_text())["outputs"][-1]["values"]
        if split != expanded:
            raise RuntimeError("split and expanded FFN differ")
    normal = json.loads((work / "glu-asymmetric/reference-output.json").read_text())["outputs"][0]["values"]
    reversed_glu = json.loads((work / "glu-reversed/reference-output.json").read_text())["outputs"][0]["values"]
    if max(abs(a-b) for a, b in zip(normal, reversed_glu)) < 1:
        raise RuntimeError("GLU fixture does not distinguish gate/up ordering")
    summary = {"status": "passed", "positive_cases": len(POSITIVE),
               "negative_cases": len(NEGATIVE), "cases": reports}
    (work / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": "passed", "positive_cases": len(POSITIVE),
                      "negative_cases": len(NEGATIVE), "evidence": str(work)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runner", "check-library", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="import-", dir=args.output_root.resolve()))
    run(args, work)

if __name__ == "__main__":
    main()
