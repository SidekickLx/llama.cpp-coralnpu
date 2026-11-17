#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Accept complete real-model FFNs against the official StableHLO interpreter."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/ggml"))
from verify_unpatched import require_clean

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def values(path, key="outputs"):
    return json.loads(path.read_text())[key]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runner", "check-library", "model", "tool", "reference-runner", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    revision = require_clean()
    lock = json.loads((ROOT / "dependencies.lock.json").read_text())
    model_hash = hashlib.sha256(args.model.read_bytes()).hexdigest()
    require(model_hash == lock["model"]["sha256"], "model differs from the locked acceptance fixture")
    runner = args.runner.resolve(strict=True)
    cache = (runner.parent / "CMakeCache.txt").read_text()
    require(f"LLAMA_SOURCE_DIR:PATH={(ROOT / 'repos/llama.cpp').resolve()}" in cache,
            "build must use the clean locked llama.cpp source")
    args.output_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="model-ffn-", dir=args.output_root.resolve()))
    output = work / "capture"
    print(f"Evidence: {work}", flush=True)
    start = time.perf_counter()
    with (work / "stdout.txt").open("w") as out, (work / "stderr.txt").open("w") as err:
        result = subprocess.run(
            [str(runner), str(args.model.resolve()), str(args.check_library.resolve()),
             str(output), "--ffn-regions"], stdout=out, stderr=err, timeout=600, check=False)
    wall_ms = (time.perf_counter() - start) * 1000
    require(result.returncode == 0,
            f"real FFN capture failed: {(work / 'stderr.txt').read_text()[-4000:]}")
    summary = json.loads((output / "summary.json").read_text())
    require(summary["status"] == "passed" and summary["mode"] == "ffn-regions" and
            summary["steps"] == 4 and summary["layers"] == 5 and summary["captures"] == 20,
            "incomplete model FFN coverage")
    cases = []
    anchors = None
    prefill_shapes = set()
    for step in range(4):
        current_anchors = set()
        for layer in range(5):
            label = f"step-{step}-ffn-{layer}"
            path = output / label
            report = json.loads((path / "report.json").read_text())
            origin = json.loads((path / "capture.json").read_text())
            require(origin["step"] == step and origin["region"] == layer and
                    len(origin["members"]) == 4, "invalid real split-SwiGLU provenance")
            current_anchors.add(origin["anchor"])
            require(report["status"] == "passed" and len(report["comparisons"]) == 1 and
                    not report["comparisons"][0]["failing_indices"], "numerical comparison failed")
            ir = (path / "imported.mlir").read_text()
            require(ir.count("stablehlo.dot_general") == 3 and
                    ir.count("coral.ffn_role") == 4 and 'func.func @ffn(' in ir and
                    "custom_call" not in ir, "incomplete standard FFN program")
            with (path / "verify.stdout.txt").open("w") as out, (path / "verify.stderr.txt").open("w") as err:
                subprocess.run([str(args.tool.resolve()), "verify", str(path / "imported.mlir")],
                               stdout=out, stderr=err, timeout=60, check=True)
            inputs = values(path / "inputs.json", "inputs")
            actual = values(path / "reference-output.json")
            expected = values(path / "ggml-output.json")
            require(len(inputs) == 4 and len(actual) == len(expected) == 1, "incorrect function ABI")
            m, d = inputs[0]["shape"]
            require(d == 64 and inputs[1]["shape"] == inputs[2]["shape"] == [172, 64] and
                    inputs[3]["shape"] == [64, 172] and
                    actual[0]["shape"] == expected[0]["shape"] == [m, 64],
                    "unexpected real FFN dimensions")
            require(m in (1, 5) if step == 0 else m == 1, "wrong prefill/decode shape")
            if step == 0:
                prefill_shapes.add(m)
            a, b = actual[0]["values"], expected[0]["values"]
            require(len(a) == len(b) == m * 64, "incomplete output payload")
            require(all(math.isfinite(x) and math.isfinite(y) and
                        abs(x-y) <= 1e-4 + 1e-4 * abs(y) for x, y in zip(a, b)),
                    "stored reference payload differs from GGML")
            cases.append({"case": label, "anchor": origin["anchor"], "shape": [m, 64, 172],
                          "elements": len(a), "max_abs_error": max(abs(x-y) for x, y in zip(a, b)),
                          "import_ms": report["import_ms"], "reference_ms": report["reference_ms"]})
        require(len(current_anchors) == 5, "duplicate or missing FFN anchors")
        if anchors is not None:
            require(current_anchors == anchors, "layer coverage changed between steps")
        anchors = current_anchors
    require(5 in prefill_shapes, "missing actual prefill profile")
    for step in (1, 2):
        first = values(output / f"step-{step}-ffn-0/inputs.json", "inputs")[0]["values"]
        second = values(output / f"step-{step+1}-ffn-0/inputs.json", "inputs")[0]["values"]
        require(first != second, "reused decode graph captured stale activations")

    # Deliberately swap the two same-shaped real weights in a separate replay.
    # A type-correct program must still reveal this semantic binding error.
    source = output / "step-0-ffn-0"
    swapped = json.loads((source / "inputs.json").read_text())
    swapped["inputs"][1], swapped["inputs"][2] = swapped["inputs"][2], swapped["inputs"][1]
    (work / "swapped-inputs.json").write_text(json.dumps(swapped) + "\n")
    with (work / "swapped.stdout.txt").open("w") as out, (work / "swapped.stderr.txt").open("w") as err:
        subprocess.run([str(args.reference_runner.resolve()), str(source / "imported.mlir"), "ffn",
                        str(work / "swapped-inputs.json"), str(work / "swapped-output.json")],
                       stdout=out, stderr=err, timeout=180, check=True)
    a = values(work / "swapped-output.json")[0]["values"]
    b = values(source / "ggml-output.json")[0]["values"]
    mismatches = sum(abs(x-y) > 1e-4 + 1e-4 * abs(y) for x, y in zip(a, b))
    require(mismatches > 0, "real gate/up role swap was numerically invisible")
    require_clean()
    report = {"status": "passed", "llama_revision": revision, "llama_worktree_clean": True,
              "model_sha256": model_hash, "runner_sha256": hashlib.sha256(runner.read_bytes()).hexdigest(),
              "check_library_sha256": hashlib.sha256(args.check_library.read_bytes()).hexdigest(),
              "real_ffn_regions": len(cases), "elements": sum(case["elements"] for case in cases),
              "atol": 1e-4, "rtol": 1e-4, "serialized_modules_verified": len(cases),
              "max_abs_error": max(case["max_abs_error"] for case in cases),
              "swapped_weight_mismatches": mismatches,
              "capture_and_reference_wall_ms": wall_ms,
              "import_ms": sum(case["import_ms"] for case in cases),
              "reference_ms": sum(case["reference_ms"] for case in cases), "cases": cases}
    (work / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}), flush=True)
if __name__ == "__main__":
    main()
