#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Exercise the artifact boundary without claiming fixture bytes are executable."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "compiler"))
from ffn_artifact import make_manifest, write_json

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checker", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "build/m4-contract-results")
    args = parser.parse_args()
    checker = args.checker.resolve(strict=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="contract-", dir=args.output_root.resolve()))
    description = json.loads(subprocess.check_output(
        [str(checker), "--describe", "2", "3", "4"], text=True))
    fixture = work / "valid"
    fixture.mkdir()
    (fixture / "region.vmfb").write_bytes(b"contract-only fixture, not executable")
    (fixture / "region.stablehlo.mlir").write_text("module {}\n")
    write_json(fixture / "region.metadata.json", {"fixture": "contract-only"})
    placement = {"numerical_target": "coralnpu", "host_dispatches": 0, "coral_dispatches": 1}
    write_json(fixture / "placement.json", {
        "summary": placement, "executables": [{"name": "fixture_dispatch", "target": "coralnpu"}]})
    support = {name: "1" * 64 for name in ("linker", "linker_script", "iree_archive", "crt_archive")}
    valid = make_manifest(fixture, description, "2" * 64, support, placement)
    write_json(fixture / "manifest.json", valid)
    results = []

    def run(label, directory, expected=1, dims=("2", "3", "4")):
        result = subprocess.run([str(checker), str(directory), *dims],
                                capture_output=True, text=True, timeout=20)
        (work / (label + ".stdout.txt")).write_text(result.stdout)
        (work / (label + ".stderr.txt")).write_text(result.stderr)
        if result.returncode != expected:
            raise RuntimeError(f"{label}: expected exit {expected}, got {result.returncode}: {result.stderr}")
        results.append({"case": label, "expected_exit": expected})

    run("valid", fixture, 0)
    mutations = [
        ("schema", ["schema"], "v999"), ("publication", ["state"], "building"),
        ("abi", ["contract", "abi"], "coral.m1"),
        ("entry", ["contract", "entry"], "matmul"),
        ("policy", ["contract", "numeric_policy"], "fast"),
        ("shape", ["contract", "shape", "m"], 3),
        ("float-dimension", ["contract", "shape", "m"], 2.0),
        ("bool-dimension", ["contract", "shape", "m"], True),
        ("negative-dimension", ["contract", "shape", "m"], -1),
        ("role", ["contract", "inputs", 1, "role"], "up_weight"),
        ("dtype", ["contract", "inputs", 0, "dtype"], "f16"),
        ("layout", ["contract", "inputs", 0, "layout"], "column-major"),
        ("output", ["contract", "output", "shape"], [3, 2]),
        ("arity", ["contract", "inputs"], valid["contract"]["inputs"][:3]),
        ("host", ["compatibility", "host"], "linux-aarch64"),
        ("isa", ["compatibility", "target", "features"], "+m"),
        ("target-abi", ["compatibility", "target", "abi"], "lp64"),
        ("dtcm", ["compatibility", "target", "dtcm_bytes"], 65536),
        ("itcm", ["compatibility", "target", "itcm_bytes"], 16384),
        ("stack", ["compatibility", "target", "stack_bytes"], 2048),
        ("highmem", ["compatibility", "target", "highmem"], True),
        ("float-target", ["compatibility", "target", "itcm_bytes"], 8192.0),
        ("lock", ["compatibility", "dependency_lock_sha256"], "0" * 64),
        ("compiler-revision", ["compatibility", "repositories", "coralnpu-compiler", "revision"], "0" * 40),
        ("pipeline", ["program", "pipeline"], "unknown"),
        ("semantic", ["program", "semantic_sha256"], "3" * 64),
        ("compiler-digest", ["compiler", "binary_sha256"], "not-a-digest"),
        ("compiler-options", ["compiler", "options"], []),
        ("support-identity", ["compiler", "support_sha256", "crt_archive"], "short"),
        ("host-placement", ["placement", "host_dispatches"], 1),
        ("zero-dispatch", ["placement", "coral_dispatches"], 0),
        ("dispatch-count", ["placement", "coral_dispatches"], 2),
        ("numerical-target", ["placement", "numerical_target"], "local"),
        ("reflection", ["reflection"], "wrong"),
        ("vmfb-digest", ["files", "region.vmfb"], "4" * 64),
        ("uppercase-digest", ["files", "region.vmfb"], "A" * 64),
    ]
    for label, path, value in mutations:
        directory = work / label
        shutil.copytree(fixture, directory)
        manifest = copy.deepcopy(valid)
        node = manifest
        for part in path[:-1]: node = node[part]
        node[path[-1]] = value
        (directory / "manifest.json").write_text(json.dumps(manifest))
        run(label, directory)
    for label in ("missing-field", "extra-field", "path-traversal", "duplicate-key", "malformed",
                  "symlink-manifest", "symlink-vmfb", "empty-vmfb", "changed-ir",
                  "missing-file", "bad-placement-report", "invalid-metadata", "oversized-manifest"):
        directory = work / label
        shutil.copytree(fixture, directory)
        manifest = copy.deepcopy(valid)
        path = directory / "manifest.json"
        if label == "missing-field":
            del manifest["contract"]["abi"]
        elif label == "extra-field": manifest["unexpected"] = 1
        elif label == "path-traversal": manifest["files"]["../outside.vmfb"] = "0" * 64
        elif label == "duplicate-key":
            path.write_text('{"schema":"bad",' + json.dumps(valid)[1:])
        elif label == "malformed": path.write_text("{")
        elif label == "oversized-manifest": path.write_bytes(b" " * (1024 * 1024 + 1))
        elif label in ("symlink-manifest", "symlink-vmfb"):
            name = "manifest.json" if label == "symlink-manifest" else "region.vmfb"
            (directory / name).unlink()
            (directory / name).symlink_to(fixture / name)
        elif label == "empty-vmfb": (directory / "region.vmfb").write_bytes(b"")
        elif label == "changed-ir": (directory / "region.stablehlo.mlir").write_text("changed")
        elif label == "missing-file": (directory / "region.metadata.json").unlink()
        elif label in ("bad-placement-report", "invalid-metadata"):
            from ffn_artifact import digest
            name = "placement.json" if label == "bad-placement-report" else "region.metadata.json"
            (directory / name).write_text("{}" if label == "bad-placement-report" else "[]")
            manifest["files"][name] = digest((directory / name).read_bytes())
        if label in ("missing-field", "extra-field", "path-traversal",
                     "bad-placement-report", "invalid-metadata"):
            path.write_text(json.dumps(manifest))
        run(label, directory)
    run("request-shape", fixture, dims=("1", "3", "4"))
    for label, dims in (("zero", ("0", "3", "4")), ("overflow", (str(2**63-1), "3", "4")),
                        ("negative", ("-1", "3", "4")), ("decimal", ("2.0", "3", "4"))):
        run("spec-" + label, fixture, dims=dims)
    report = {"status": "passed", "positive": 1, "negative": len(results)-1, "cases": results}
    write_json(work / "report.json", report)
    print(json.dumps({"status": "passed", "negative": len(results)-1, "evidence": str(work)}))

if __name__ == "__main__":
    main()
