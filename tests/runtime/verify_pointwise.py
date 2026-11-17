#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Check standalone FFN members against GGML CPU with no compiler in PATH."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

def invoke(command, work, label, expected=0, message=None):
    env = os.environ.copy()
    env["PATH"] = "/nonexistent"
    with (work / f"{label}.stdout.log").open("w") as out, (work / f"{label}.stderr.log").open("w") as err:
        result = subprocess.run(list(map(str, command)), stdout=out, stderr=err,
                                env=env, timeout=180, check=False)
    text = (work / f"{label}.stderr.log").read_text()
    if result.returncode != expected or (message and message not in text):
        raise RuntimeError(f"{label}: exit {result.returncode}: {text[-4000:]}")
    return [json.loads(line) for line in (work / f"{label}.stdout.log").read_text().splitlines()]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("build/m3-pointwise-results"))
    args = parser.parse_args()
    runner = args.runner.resolve(strict=True)
    artifacts = args.artifacts.resolve(strict=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="pointwise-", dir=args.output_root)).resolve()
    print(f"Evidence: {work}", flush=True)
    lines = (artifacts / "pointwise.tsv").read_text().splitlines()
    if lines[0] != "coral-m3-pointwise-v1": raise ValueError("invalid pointwise catalog")
    results = []
    hashes = {}
    for line in lines[1:]:
        op, m, n, filename = line.split()
        records = invoke([runner, op, m, n, artifacts / filename], work, filename)
        if len(records) != 2: raise RuntimeError("missing context recreation")
        for record in records:
            if (record["status"] != "passed" or record["calls"] != 4 or
                record["dispatches"] < 4 or record["instruction_samples"] != record["dispatches"] or
                record["kernel_instructions"] <= 0 or record["host_loads"]):
                raise RuntimeError("missing Coral execution evidence")
        results.extend(records)
        hashes[filename] = hashlib.sha256((artifacts / filename).read_bytes()).hexdigest()
        invoke([runner, op, str(int(m) + 1), n, artifacts / filename], work,
               filename + ".wrong-shape", expected=1, message="ABI mismatch")
        print(f"PASS {filename}", flush=True)
    corrupt = work / "corrupt.vmfb"
    corrupt.write_bytes(b"not a module")
    invoke([runner, "silu", 1, 172, corrupt], work, "corrupt", expected=1, message="load VMFB")
    invoke([runner, "silu", 1, 172, work / "missing.vmfb"], work, "missing",
           expected=1, message="cannot open VMFB")
    report = {"status": "passed", "profiles": len(hashes), "results": results,
              "negative_processes": len(hashes) + 2, "invalid_binding_checks": len(results) * 4,
              "atol": 2e-6, "rtol": 1e-5,
              "max_abs_error": max(r["max_abs_error"] for r in results),
              "calls": sum(r["calls"] for r in results),
              "dispatches": sum(r["dispatches"] for r in results),
              "kernel_instructions": sum(r["kernel_instructions"] for r in results),
              "invoke_ms": sum(r["invoke_ms"] for r in results),
              "runner_sha256": hashlib.sha256(runner.read_bytes()).hexdigest(),
              "vmfb_sha256": hashes}
    (work / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("results", "vmfb_sha256")}), flush=True)
if __name__ == "__main__":
    main()
