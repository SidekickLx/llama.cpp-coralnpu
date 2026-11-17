#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run isolated GGML adapter boundary tests using retained M1 VMFBs."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

CASES = [
    "capabilities",
    "dense_decode", "dense_prefill", "explicit_f32",
    "weight_stride", "activation_stride", "f16_weight", "bf16_weight",
    "broadcast", "batch3d",
    "precision_bf16_acc", "precision_f16_acc", "precision_bf16_src", "precision_q8_src",
    "missing_profile", "mixed_graph",
    "cache_default_mutable", "cache_optin_mutable", "cache_view_mutable", "cache_view_readonly", "cache_immutable",
    "preflight_failure", "abi_failure",
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--m1-vmfb", required=True, type=Path)
    parser.add_argument("--m16-vmfb", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("build/m2-results"))
    args = parser.parse_args()
    runner = args.runner.resolve(strict=True)
    sources = {1: args.m1_vmfb.resolve(strict=True), 16: args.m16_vmfb.resolve(strict=True)}
    args.output_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="adapter-", dir=args.output_root)).resolve()
    print(f"Results: {work}", flush=True)
    artifacts = work / "artifacts"
    artifacts.mkdir()
    for m, source in sources.items():
        shutil.copyfile(source, artifacts / f"m{m}.vmfb")
    (artifacts / "profiles.tsv").write_text(
        "coral-m2-v1\n1\t16\t8\tm1.vmfb\n16\t16\t8\tm16.vmfb\n", encoding="utf-8"
    )
    # Capability queries must not read VMFB bytes or require artifact files
    # after registration. The C++ test renames only this private directory's files.
    query_artifacts = work / "query-artifacts"
    query_artifacts.mkdir()
    (query_artifacts / "m1.vmfb").write_bytes(b"deliberately not a VMFB")
    (query_artifacts / "profiles.tsv").write_text(
        "coral-m2-v1\n1\t16\t8\tm1.vmfb\n", encoding="utf-8"
    )
    abi_artifacts = work / "abi-artifacts"
    abi_artifacts.mkdir()
    shutil.copyfile(sources[16], abi_artifacts / "m1.vmfb")
    (abi_artifacts / "profiles.tsv").write_text(
        "coral-m2-v1\n1\t16\t8\tm1.vmfb\n", encoding="utf-8"
    )
    env = os.environ.copy()
    env["PATH"] = "/nonexistent"
    results = []
    for case in CASES:
        directory = query_artifacts if case == "capabilities" else (
            abi_artifacts if case == "abi_failure" else artifacts
        )
        try:
            process = subprocess.run(
                [str(runner), case, str(directory)], capture_output=True, text=True,
                timeout=180, env=env, check=False,
            )
        except subprocess.TimeoutExpired as error:
            for stream in ("stdout", "stderr"):
                payload = getattr(error, stream) or b""
                if isinstance(payload, bytes):
                    payload = payload.decode("utf-8", errors="replace")
                (work / f"{case}.{stream}.log").write_text(payload, encoding="utf-8")
            raise RuntimeError(f"{case}: timeout; logs retained at {work}") from error
        (work / f"{case}.stdout.log").write_text(process.stdout, encoding="utf-8")
        (work / f"{case}.stderr.log").write_text(process.stderr, encoding="utf-8")
        if process.returncode != 0:
            raise RuntimeError(f"{case}: exit {process.returncode}\n{process.stderr[-4000:]}")
        result = json.loads(process.stdout)
        if result["status"] != "passed" or result["case"] != case or not result["checks"]:
            raise RuntimeError(f"{case}: invalid completion evidence")
        if sum(event["dispatches"] for event in result["events"]) != result["dispatches"]:
            raise RuntimeError(f"{case}: dispatch evidence mismatch")
        if result["instruction_samples"] != result["dispatches"]:
            raise RuntimeError(f"{case}: missing instruction samples")
        results.append(result)
        print(f"PASS {case}: {len(result['checks'])} checks, {result['dispatches']} dispatches", flush=True)
    report = {
        "status": "passed",
        "cases": len(results),
        "checks": sum(len(result["checks"]) for result in results),
        "atol": 1e-6, "rtol": 1e-5,
        "max_abs_error": max(result["max_abs_error"] for result in results),
        "matmuls": sum(result["matmuls"] for result in results),
        "dispatches": sum(result["dispatches"] for result in results),
        "kernel_instructions": sum(result["kernel_instructions"] for result in results),
        "runner_sha256": sha256(runner),
        "vmfb_sha256": {f"m{m}": sha256(source) for m, source in sources.items()},
        "results": results,
    }
    # Success is published only after every fresh process has passed.
    (work / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}), flush=True)


if __name__ == "__main__":
    main()
