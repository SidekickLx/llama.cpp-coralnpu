#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Validate standalone runner I/O, batch execution and failure publication."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from ffn_compare import compare

ROOT = Path(__file__).resolve().parents[2]

def require(value, message):
    if not value:
        raise RuntimeError(message)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, default=ROOT / "build/m1/coral-ffn-runner")
    parser.add_argument("--cpu-reference", type=Path, default=ROOT / "build/m1/coral-ffn-cpu-reference")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "build/m4-runner-cli-results")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="cli-", dir=args.output_root.resolve()))
    print(f"Evidence: {work}", flush=True)
    original = (args.fixture / "inputs.json").read_bytes()
    input_file = work / "inputs.json"
    input_file.write_bytes(original)
    baseline = json.loads(original)
    changed = copy.deepcopy(baseline)
    changed["inputs"][0]["values"] = [value + 0.25 for value in changed["inputs"][0]["values"]]
    changed["inputs"][1]["values"] = [-0.7 * value for value in changed["inputs"][1]["values"]]
    alternate = work / "alternate.json"
    alternate.write_text(json.dumps(changed))
    checks = []
    def invoke(command, label, expected=0):
        result = subprocess.run(list(map(str, command)), capture_output=True, text=True, timeout=60)
        (work / (label + ".stdout.txt")).write_text(result.stdout)
        (work / (label + ".stderr.txt")).write_text(result.stderr)
        require(result.returncode == expected, f"{label}: unexpected exit: {result.stderr[-2000:]}")
        checks.append(label)
    engines = [("coral", [args.runner.resolve(), "--artifact", args.artifact.resolve()]),
               ("cpu", [args.cpu_reference.resolve()])]
    for label, command in engines:
        output = work / (label + ".json")
        options = ["--inputs", input_file, "--inputs", alternate, "--repeat", "2", "--output", output]
        if label == "coral":
            options += ["--recreate", "2"]
        invoke(command + options, label)
    coral = json.loads((work / "coral.json").read_text())
    cpu = json.loads((work / "cpu.json").read_text())
    captured = json.loads((args.fixture / "ggml-output.json").read_text())["outputs"][0]
    for i in range(2):
        require(compare(coral["cases"][i]["outputs"][0], cpu["cases"][i]["outputs"][0])["status"] == "passed",
                "batch numerical comparison failed")
        expected_hash = hashlib.sha256((input_file if i == 0 else alternate).read_bytes()).hexdigest()
        require(coral["cases"][i]["input_sha256"] == cpu["cases"][i]["input_sha256"] == expected_hash,
                "input evidence identity mismatch")
    require(compare(cpu["cases"][0]["outputs"][0], captured)["status"] == "passed", "captured CPU replay mismatch")
    require(coral["cases"][0]["outputs"] != coral["cases"][1]["outputs"], "changed inputs produced stale output")
    require(len(coral["cycles"]) == 2 and all(c["stats"]["module_loads"] == 1 and
            c["stats"]["calls_completed"] == 4 and c["stats"]["binding_allocations"] == 7
            for c in coral["cycles"]), "batch/module reuse evidence failed")
    failures = {}
    mutations = {
        "arity": ("inputs", None),
        "dtype": ("dtype", "f16"), "float-shape": ("shape", [2.0, 3]),
        "boolean-shape": ("shape", [True, 3]), "zero-shape": ("shape", [0, 3]),
        "negative-shape": ("shape", [-2, 3]), "huge-shape": ("shape", [2**63, 3]),
        "rank": ("shape", [6]), "count": ("values", [0.0]),
    }
    for label, (field, value) in mutations.items():
        bad = copy.deepcopy(baseline)
        if field == "inputs":
            bad["inputs"].pop()
        else:
            bad["inputs"][0][field] = value
        failures[label] = json.dumps(bad)
    for label, value in (("bool-value", True), ("string-value", "1.0"),
                         ("null-value", None), ("f32-overflow", 1e40), ("nan-value", float("nan"))):
        bad = copy.deepcopy(baseline)
        bad["inputs"][0]["values"][0] = value
        failures[label] = json.dumps(bad)
    bad = copy.deepcopy(baseline)
    bad["inputs"][3]["shape"] = bad["inputs"][1]["shape"]
    failures["role-shape"] = json.dumps(bad)
    bad = copy.deepcopy(baseline)
    bad["extra"] = 1
    failures["unknown-field"] = json.dumps(bad)
    failures["malformed"] = "{"
    failures["duplicate-key"] = '{"inputs":[],' + original.decode().lstrip()[1:]
    failures["nested"] = '{"inputs":' + "[" * 30 + "0" + "]" * 30 + "}"
    for label, text in failures.items():
        path = work / (label + ".input.json")
        path.write_text(text)
        for engine, command in engines:
            output = work / (engine + "-" + label + ".json")
            invoke(command + ["--inputs", path, "--output", output], engine + "-" + label, 1)
            require(not output.exists(), "rejected input published output")
    for engine, command in engines:
        for label, options in (
            ("zero-repeat", ["--repeat", "0"]),
            ("bad-repeat", ["--repeat", "2junk"]),
            ("negative-repeat", ["--repeat", "-1"]),
            ("duplicate-option", ["--repeat", "1", "--repeat", "2"]),
            ("unknown-option", ["--unknown", "1"])):
            output = work / (engine + "-" + label + ".json")
            invoke(command + ["--inputs", input_file, "--output", output, *options], engine + "-" + label, 1)
            require(not output.exists(), "bad CLI published output")
        output = work / (engine + "-existing.json")
        output.write_bytes(b"preserve this result")
        invoke(command + ["--inputs", input_file, "--output", output], engine + "-overwrite", 1)
        require(output.read_bytes() == b"preserve this result", "overwrote existing output")
        invoke(command + ["--inputs", input_file, "--output", input_file], engine + "-input-overwrite", 1)
        require(input_file.read_bytes() == original, "overwrote input")
        link = work / (engine + "-symlink.json")
        link.symlink_to(input_file)
        output = work / (engine + "-link-output.json")
        invoke(command + ["--inputs", link, "--output", output], engine + "-input-symlink", 1)
        require(not output.exists(), "symlink input published output")
    result = {"status": "passed", "checks": checks, "batch_cases": 2,
              "numerical_comparisons": 3, "negative_processes": len(checks) - 2,
              "runner_sha256": hashlib.sha256(args.runner.read_bytes()).hexdigest(),
              "cpu_reference_sha256": hashlib.sha256(args.cpu_reference.read_bytes()).hexdigest()}
    (work / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "checks"}))

if __name__ == "__main__":
    main()
