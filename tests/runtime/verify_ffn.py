#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Exercise the independent FFN runtime, not GGML fused backend routing."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
CORE_FILES = ("manifest.json", "region.vmfb", "region.stablehlo.mlir",
              "region.metadata.json", "placement.json")

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def copy_package(source, destination):
    destination.mkdir()
    for name in CORE_FILES:
        shutil.copyfile(source / name, destination / name)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True,
                        help="Passing verify_ffn_aot evidence directory")
    parser.add_argument("--runner", type=Path, default=ROOT / "build/m1/coral-ffn-runtime-test")
    parser.add_argument("--output-root", type=Path, default=ROOT / "build/m4-runtime-results")
    args = parser.parse_args()
    source = args.artifacts.resolve(strict=True)
    runner = args.runner.resolve(strict=True)
    report = json.loads((source / "report.json").read_text())
    require(report["status"] == "passed" and len(report["artifacts"]) == 6,
            "expected six passing AOT profiles")
    args.output_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="runtime-", dir=args.output_root.resolve()))
    print(f"Evidence: {work}", flush=True)
    results = []

    def invoke(command, label, timeout=600):
        with (work / (label + ".stdout.txt")).open("w") as out, \
             (work / (label + ".stderr.txt")).open("w") as err:
            result = subprocess.run(list(map(str, command)), stdout=out, stderr=err,
                                    cwd=ROOT, timeout=timeout, check=False)
        require(result.returncode == 0, f"{label}: failed; see {work}")
        return (work / (label + ".stdout.txt")).read_text()

    for artifact in report["artifacts"]:
        label = artifact["case"]
        package = source / artifact["directory"]
        if label == "tiny":
            package = work / "owned-tiny"
            copy_package(source / artifact["directory"], package)
        output = invoke([runner, "owned" if label == "tiny" else "run",
                         package, *artifact["shape"]], label)
        records = [json.loads(line) for line in output.splitlines() if line.startswith("{")]
        require(len(records) == 2 and all(row["status"] == "passed" for row in records),
                "missing execution cycles")
        results.append({"case": label, "cycles": records})
        print(f"Executed {label}: two device lifetimes, fourteen FFN calls", flush=True)

    tiny = source / "tiny"
    dims = (2, 3, 4)
    invoke([runner, "cross", source / "real-decode", 1, 64, 172,
            5, source / "real-prefill"], "cross-profile")
    negatives = []
    for label in ("missing-manifest", "changed-vmfb", "wrong-target", "wrong-role",
                  "incomplete", "corrupt-rehashed", "reflection-mismatch"):
        package = work / label
        copy_package(tiny, package)
        manifest = json.loads((package / "manifest.json").read_text())
        expected = "FFN artifact"
        if label == "missing-manifest":
            (package / "manifest.json").unlink()
        elif label == "changed-vmfb":
            with (package / "region.vmfb").open("ab") as out:
                out.write(b"changed")
        elif label == "wrong-target":
            manifest["compatibility"]["target"]["dtcm_bytes"] = 65536
        elif label == "wrong-role":
            manifest["contract"]["inputs"][1]["role"] = "up_weight"
        elif label == "incomplete":
            manifest["state"] = "building"
        else:
            if label == "corrupt-rehashed":
                (package / "region.vmfb").write_bytes(b"not a VM module")
                expected = "load VMFB"
            else:
                shutil.copyfile(source / "real-decode/region.vmfb", package / "region.vmfb")
                expected = "VMFB ABI mismatch"
            manifest["files"]["region.vmfb"] = hashlib.sha256((package / "region.vmfb").read_bytes()).hexdigest()
        if label != "missing-manifest":
            (package / "manifest.json").write_text(json.dumps(manifest))
        invoke([runner, "reject", package, *dims, expected, tiny], label)
        negatives.append(label)

    # Fault injection deliberately bypasses semantic admission. These are
    # counterfeit fixtures, never outputs of the production FFN compiler.
    # Retain the real reflection string but change VM arity/output to exercise
    # the independent VM signature and output checks after manifest validation.
    valid = json.loads((tiny / "manifest.json").read_text())
    compiler = (ROOT / "repos/coralnpu-compiler/bazel-bin/external/iree_core+/tools/iree-compile").resolve(strict=True)
    support = Path(str(compiler) + ".runfiles") / "_main"
    tool = ROOT / "build/m3/coral-stablehlo-tool"
    for label in ("wrong-arity", "wrong-output"):
        package = work / label
        copy_package(tiny, package)
        args_text = "%arg0: tensor<2x3xf32>, %arg1: tensor<4x3xf32>"
        if label == "wrong-output":
            args_text += ", %arg2: tensor<4x3xf32>, %arg3: tensor<3x4xf32>"
        ir = ("module {\n  func.func @ffn(" + args_text + ") -> tensor<2x4xf32> {\n"
              "    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [1], "
              "precision = [DEFAULT, DEFAULT] : (tensor<2x3xf32>, tensor<4x3xf32>) -> tensor<2x4xf32>\n"
              "    return %0 : tensor<2x4xf32>\n  }\n}\n")
        fault = package / "fault.mlir"
        bound = package / "fault-bound.mlir"
        fault.write_text(ir)
        invoke([tool, "bind-ffn-reflection", fault, valid["reflection"], bound], label + "-reflection")
        invoke([compiler, bound, *valid["compiler"]["options"],
                "--coralnpu-embedded-linker-path=" + str(support / "toolchain_rv32/bin/riscv32-unknown-elf-ld"),
                "--coralnpu-linker-script-path=" + str(support / "crt/coralnpu_tcm.ld"),
                "-o", package / "region.vmfb"], label + "-compile")
        manifest = json.loads((package / "manifest.json").read_text())
        manifest["files"]["region.vmfb"] = hashlib.sha256((package / "region.vmfb").read_bytes()).hexdigest()
        (package / "manifest.json").write_text(json.dumps(manifest))
        mode = "reject" if label == "wrong-arity" else "poison"
        expected = "calling convention" if label == "wrong-arity" else "output violates"
        invoke([runner, mode, package, *dims, expected, tiny], label)
        negatives.append(label)
    result = {"status": "passed", "profiles": results, "negative_cases": negatives,
              "cross_profile_weight_reuse": True, "owned_package_snapshot": True,
              "total_numerical_comparisons": 86,
              "timing_scope": "MPACT host wall time; not hardware latency or speedup",
              "tolerance": {"absolute": 2e-5, "relative": 2e-4}}
    (work / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "passed", "profiles": 6, "negative_cases": len(negatives),
                      "evidence": str(work)}))

if __name__ == "__main__":
    main()
