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
