#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify semantic admission, normalization and full FFN AOT publication."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]

def require(value, message):
    if not value: raise RuntimeError(message)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True,
                        help="Complete M3 verify_ffn_import evidence directory")
    parser.add_argument("--real-fixtures", type=Path, required=True,
                        help="Complete M3 verify_model_ffn evidence directory")
    parser.add_argument("--tool", type=Path, default=ROOT / "build/m3/coral-stablehlo-tool")
    parser.add_argument("--output-root", type=Path, default=ROOT / "build/m4-aot-results")
    args = parser.parse_args()
    tool = args.tool.resolve(strict=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="aot-", dir=args.output_root.resolve()))
    print(f"Evidence: {work}", flush=True)
    successful = {}
    def invoke(command, label, success=True, timeout=600):
        with (work / (label + ".stdout.txt")).open("w") as out, \
             (work / (label + ".stderr.txt")).open("w") as err:
            result = subprocess.run(list(map(str, command)), stdout=out, stderr=err,
                                    timeout=timeout, cwd=ROOT, check=False)
        require((result.returncode == 0) == success, f"{label}: unexpected exit; see {work}")
    for source in sorted(args.fixtures.glob("ffn-region-*/imported.mlir")):
        tensors = json.loads((source.parent / "inputs.json").read_text())["inputs"]
        dims = [tensors[0]["shape"][0], tensors[0]["shape"][1], tensors[1]["shape"][0]]
        label = source.parent.name
        output = work / (label + ".mlir")
        invoke([tool, "prepare-ffn", source.resolve(), *dims, output, work / (label + ".json")], label)
        successful[label] = hashlib.sha256(output.read_bytes()).hexdigest()
    require(len(successful) == 14, "missing M3 synthetic fixtures")
    same = ["ffn-region-split", "ffn-region-expanded", "ffn-region-expanded-reversed",
            "ffn-region-gate-up-swapped", "ffn-region-up-first", "ffn-region-shared",
            "ffn-region-shared-expanded"]
    require(len({successful[label] for label in same}) == 1,
            "source spelling, graph IDs or shared identities leaked into semantic identity")
    baseline = (args.fixtures / "ffn-region-split/imported.mlir").read_text()
    negatives = {
        "wrong-gate": baseline.replace("dot_general %arg0, %arg1", "dot_general %arg0, %arg2"),
        "wrong-up": baseline.replace("dot_general %arg0, %arg2", "dot_general %arg0, %arg1"),
        "wrong-output": baseline.replace("return %8", "return %arg0"),
        "wrong-one": baseline.replace("dense<1.000000e+00>", "dense<2.000000e+00>"),
        "wrong-exp": baseline.replace("stablehlo.exponential", "stablehlo.abs"),
        "fast-policy": baseline.replace("f32_no_fast_math_v1", "fast"),
        "f16": baseline.replace("f32", "f16"),
        "wrong-role": baseline.replace('coral.ffn_role = "gate_weight"', 'coral.ffn_role = "up_weight"'),
        "extra-op": baseline.replace("    return", "    %unused = stablehlo.negate %arg0 : tensor<2x3xf32>\n    return"),
        "unknown-attribute": baseline.replace('coral.ffn_abi =', 'unsafe_hint = true, coral.ffn_abi ='),
    }
    for label, text in negatives.items():
        source = work / (label + ".input.mlir")
        source.write_text(text)
        output = work / (label + ".mlir")
        invoke([tool, "prepare-ffn", source, 2, 3, 4, output, work / (label + ".json")], label, False)
        require(not output.exists(), "rejected program published semantic IR")
    normalized = work / "ffn-region-split.mlir"
    invoke([tool, "prepare-ffn", normalized, 2, 3, 4,
            work / "roundtrip.mlir", work / "roundtrip.json"], "roundtrip")
    require(normalized.read_bytes() == (work / "roundtrip.mlir").read_bytes(), "normalization is not stable")
    # All compiler inputs come from actual GGML imports, including real model profiles.
    cases = [
        ("tiny", args.fixtures / "ffn-region-split/imported.mlir", (2, 3, 4)),
        ("real-decode", args.real_fixtures / "capture/step-1-ffn-0/imported.mlir", (1, 64, 172)),
        ("real-prefill", args.real_fixtures / "capture/step-0-ffn-0/imported.mlir", (5, 64, 172)),
    ]
    for m in (1, 16, 32):
        cases.append((f"profile-m{m}", args.fixtures / f"ffn-region-profile-m{m}/imported.mlir", (m, 128, 256)))
    artifacts = []
    for label, source, dims in cases:
        output = work / label
        command = [sys.executable, ROOT / "compiler/compile_ffn.py", "--input", source.resolve(),
                   "--m", dims[0], "--d", dims[1], "--f", dims[2], "--output", output]
        invoke(command, "compile-" + label)
        report = json.loads((output / "build-report.json").read_text())
        require(report["status"] == "passed" and report["placement"]["host_dispatches"] == 0,
                "incomplete Coral compilation")
        artifacts.append({"case": label, "directory": str(output.relative_to(work)), **report})
        print(f"Compiled {label}: {report['placement']['coral_dispatches']} Coral dispatches", flush=True)
    existing = (work / "tiny/manifest.json").read_bytes()
    invoke([sys.executable, ROOT / "compiler/compile_ffn.py", "--input", normalized,
            "--m", 2, "--d", 3, "--f", 4, "--output", work / "tiny"], "overwrite", False)
    require((work / "tiny/manifest.json").read_bytes() == existing, "existing artifact overwritten")
    invoke([sys.executable, ROOT / "compiler/compile_ffn.py", "--input", work / "wrong-gate.input.mlir",
            "--m", 2, "--d", 3, "--f", 4, "--output", work / "rejected"], "rejected-publish", False)
    require(not (work / "rejected").exists(), "failed compilation published an artifact")
    report = {"status": "passed", "normalized_programs": len(successful),
              "semantic_rejections": len(negatives), "normalization_stable": True,
              "overwrite_rejected": True, "failed_publication_rejected": True, "artifacts": artifacts}
    (work / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "passed", "compiled_profiles": len(artifacts), "evidence": str(work)}))

if __name__ == "__main__":
    main()
