#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the complete M3 acceptance gate using retained builds and AOT artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests/ggml"))
from verify_unpatched import require_clean


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def invoke(command, label, work):
    start = time.perf_counter()
    with (work / (label + ".stdout.txt")).open("w") as out, \
         (work / (label + ".stderr.txt")).open("w") as err:
        result = subprocess.run(list(map(str, command)), cwd=ROOT,
                                stdout=out, stderr=err, timeout=1800, check=False)
    require(result.returncode == 0, f"{label} failed; see retained logs in {work}")
    return (time.perf_counter() - start) * 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build/m1")
    parser.add_argument("--compiler-dir", type=Path, default=ROOT / "build/m3")
    parser.add_argument("--model", type=Path, default=ROOT / "models/stories260K-f32.gguf")
    parser.add_argument("--matmuls", type=Path, required=True)
    parser.add_argument("--pointwise", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "build/m3-acceptance-results")
    args = parser.parse_args()
    revision = require_clean()
    build, compiler = args.build_dir.resolve(), args.compiler_dir.resolve()
    model, matmuls, pointwise = args.model.resolve(), args.matmuls.resolve(), args.pointwise.resolve()
    tool = compiler / "coral-stablehlo-tool"
    reference = compiler / "coral-stablehlo-reference"
    library = compiler / "libcoral-importer-check.so"
    binaries = [build / name for name in (
        "coral-matmul-runner", "coral-llama-runner", "coral-adapter-test",
        "coral-ffn-planner-test", "coral-ggml-import-test", "coral-model-import-test",
        "coral-pointwise-test", "coral-member-adapter-test")]
    binaries += [tool, reference, library]
    hashes = {binary.name: sha256(binary) for binary in binaries}
    require(matmuls != pointwise, "supply the separate matmul and pointwise catalogs")
    for path in (matmuls / "profiles.tsv", pointwise / "pointwise.tsv"):
        require(path.is_file(), f"missing artifact catalog: {path}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="acceptance-", dir=args.output_root.resolve()))
    print(f"Evidence: {work}", flush=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    report = {"status": "running", "scope": "M3 static F32 FFN acceptance",
              "project_revision": head, "project_worktree_dirty": dirty,
              "llama_revision": revision, "llama_worktree_clean": True,
              "dependencies_sha256": sha256(ROOT / "dependencies.lock.json"),
              "model_sha256": sha256(model), "binary_sha256": hashes, "suites": []}
    report_path = work / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    suites = [
        ("stablehlo", "tests/compiler/verify_stablehlo.py", ["--tool", tool]),
        ("reference", "tests/compiler/verify_reference.py",
         ["--tool", tool, "--reference", reference]),
        ("unpatched", "tests/ggml/verify_unpatched.py",
         ["--build-dir", build, "--check-library", library, "--model", model,
          "--artifacts", matmuls]),
        ("ffn-import", "tests/compiler/verify_ffn_import.py",
         ["--runner", build / "coral-ggml-import-test", "--check-library", library,
          "--reference-runner", reference]),
        ("model-ffn", "tests/compiler/verify_model_ffn.py",
         ["--runner", build / "coral-model-import-test", "--check-library", library,
          "--model", model, "--tool", tool, "--reference-runner", reference]),
        ("pointwise", "tests/runtime/verify_pointwise.py",
         ["--runner", build / "coral-pointwise-test", "--artifacts", pointwise]),
        ("member-adapter", "tests/ggml/verify_member_adapter.py",
         ["--runner", build / "coral-member-adapter-test", "--matmuls", matmuls,
          "--pointwise", pointwise]),
        ("ffn-members", "tests/ggml/verify_ffn_members.py",
         ["--runner", build / "coral-llama-runner", "--model", model,
          "--matmuls", matmuls, "--pointwise", pointwise]),
    ]
    started = time.perf_counter()
    try:
        # A reduced or unconfigured CTest invocation must not silently pass.
        listing = subprocess.check_output(
            ["ctest", "--test-dir", str(build), "--show-only=json-v1"], text=True)
        (work / "ctest-tests.json").write_text(listing)
        registered = {item["name"] for item in json.loads(listing)["tests"]}
        required = {"m1_matmul_m1", "m1_matmul_m16", "m1_matmul_m32",
                    "m2_adapter_boundaries", "m3_ffn_planner"}
        require(required <= registered, "configure all M1/M2/FFN CTest fixtures first")
        wall = invoke(["ctest", "--test-dir", build, "--output-on-failure",
                       "--output-junit", work / "ctest.xml"], "ctest", work)
        cases = list(ET.parse(work / "ctest.xml").getroot().iter("testcase"))
        require(required <= {case.attrib["name"] for case in cases} and
                all(case.find("failure") is None and case.find("error") is None and
                    case.find("skipped") is None for case in cases), "incomplete CTest acceptance")
        report["suites"].append({"name": "ctest", "status": "passed", "tests": len(cases),
                                 "report": "ctest.xml", "wall_ms": wall})
        print(f"ctest: passed ({len(cases)} tests)", flush=True)
        for label, script, options in suites:
            wall = invoke([sys.executable, ROOT / script, *options,
                           "--output-root", work / label], label, work)
            reports = list((work / label).glob("*/report.json"))
            require(len(reports) == 1, f"{label}: expected one suite report")
            child = json.loads(reports[0].read_text())
            require(child["status"] == "passed", f"{label}: suite report did not pass")
            report["suites"].append({"name": label, "status": "passed",
                                     "report": str(reports[0].relative_to(work)), "wall_ms": wall})
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            print(f"{label}: passed", flush=True)
        require_clean()
        require(all(sha256(binary) == hashes[binary.name] for binary in binaries),
                "a tested binary changed during acceptance")
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        report["wall_ms"] = (time.perf_counter() - started) * 1000
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "suites": len(report["suites"]),
                      "wall_ms": report["wall_ms"], "evidence": str(work)}), flush=True)


if __name__ == "__main__":
    main()
