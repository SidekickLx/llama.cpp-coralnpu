#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify actual GGUF prefill/decode node imports against captured GGML CPU data."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runner", "check-library", "model", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    model = json.loads((ROOT / "dependencies.lock.json").read_text())["model"]
    if hashlib.sha256(args.model.read_bytes()).hexdigest() != model["sha256"]:
        raise RuntimeError("this acceptance fixture requires the locked stories260K F32 model")
    args.output_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="model-import-", dir=args.output_root.resolve()))
    output = work / "capture"
    result = subprocess.run(
        [str(args.runner.resolve()), str(args.model.resolve()),
         str(args.check_library.resolve()), str(output)],
        capture_output=True, text=True, timeout=90, check=False)
    (work / "stdout.txt").write_text(result.stdout)
    (work / "stderr.txt").write_text(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"model import test exited {result.returncode}: {result.stderr[-4000:]}")
    summary = json.loads((output / "summary.json").read_text())
    if summary["status"] != "passed" or summary["captures"] != 4 or summary["steps"] != 2:
        raise RuntimeError("incomplete real-model captures")
    cases = []
    for step in range(2):
        for operator in ("matmul", "glu"):
            label = f"step-{step}-{operator}"
            path = output / label
            report = json.loads((path / "report.json").read_text())
            if report["status"] != "passed" or len(report["comparisons"]) != 1:
                raise RuntimeError(f"{label}: incorrect comparison result")
            if report["comparisons"][0]["failing_indices"]:
                raise RuntimeError(f"{label}: failed values")
            text = (path / "imported.mlir").read_text()
            spelling = "MUL_MAT" if operator == "matmul" else "GLU"
            if f'coral.ggml_op = "{spelling}"' not in text:
                raise RuntimeError(f"{label}: missing GGML provenance")
            inputs = json.loads((path / "inputs.json").read_text())["inputs"]
            results = json.loads((path / "reference-output.json").read_text())["outputs"]
            rows = 5 if step == 0 else 1
            if results[0]["shape"][0] != rows:
                raise RuntimeError(f"{label}: incorrect prefill/decode dimension")
            if operator == "glu" and results[0]["shape"][1] != 172:
                raise RuntimeError("expected real FFN width 172")
            report["input_shapes"] = [tensor["shape"] for tensor in inputs]
            report["output_shape"] = results[0]["shape"]
            cases.append(report)
    report = {"status": "passed", "model_sha256": model["sha256"], "cases": cases}
    (work / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "passed", "real_model_cases": 4,
                      "evidence": str(work),
                      "max_abs_error": max(case["comparisons"][0]["max_abs_error"] for case in cases)}))


if __name__ == "__main__":
    main()
