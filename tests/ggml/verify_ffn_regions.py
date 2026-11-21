#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Accept complete GGML FFN execution against CPU/member inference in fresh processes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import tempfile
from verify_ffn_region_backend import make_catalog
from verify_ffn_members import compare
from verify_runner import ATOL, RTOL, invoke, read_run, require
from verify_unpatched import ROOT, require_clean

def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]

def distribution(values):
    return {"samples": len(values), "median": statistics.median(values),
            "minimum": min(values), "maximum": max(values)}

def inspect_regions(output, expected, dispatch_counts=None):
    if dispatch_counts is None:
        dispatch_counts = {1: 6, 5: 3}
    trace = lines(output / "trace.jsonl")
    regions = [row for row in trace if row["kind"] == "ffn_region"]
    require(len(regions) == expected, "wrong complete-region execution count")
    for step in range(4):
        selected = [row for row in regions if row["step"] == step]
        names = []
        for row in selected:
            require(len(row["program_sha256"]) == 64, "missing executed semantic identity")
            require((row["d"], row["f"]) == (64, 172), "wrong FFN dimensions")
            require(row["m"] in (1, 5) and (step == 0 or row["m"] == 1), "wrong profile")
            ops = [member["op"] for member in row["members"]]
            require(ops == ["MUL_MAT", "MUL_MAT", "GLU", "MUL_MAT"], "wrong FFN coverage")
            require(row["dispatches"] == dispatch_counts[row["m"]], "unexpected dispatch count")
            names.extend(member["name"] for member in row["members"])
        require(len(set(names)) == len(names), "overlapping regions")
        individual = [row["name"] for row in trace
                      if row["kind"] == "node" and row["step"] == step]
        require(not set(names).intersection(individual), "a member was also executed individually")
        if expected == 20:
            require(len(selected) == 5, "missing model layer region")
    return regions

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, default=Path("build/m1/coral-llama-runner"))
    parser.add_argument("--model", type=Path, default=Path("models/stories260K-f32.gguf"))
    parser.add_argument("--matmuls", type=Path, required=True)
    parser.add_argument("--pointwise", type=Path, required=True)
    parser.add_argument("--ffn", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output-root", type=Path, default=Path("build/m4-region-model-results"))
    args = parser.parse_args()
    require(args.repeats >= 3, "at least three paired runs are required")
    revision = require_clean()
    runner, model = args.runner.resolve(strict=True), args.model.resolve(strict=True)
    cache = (runner.parent / "CMakeCache.txt").read_text()
    require(f"LLAMA_SOURCE_DIR:PATH={(ROOT / 'repos/llama.cpp').resolve()}" in cache,
            "runner must use the locked unmodified llama.cpp source")
    args.output_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="model-", dir=args.output_root)).resolve()
    print(f"Evidence: {work}", flush=True)
    catalog = work / "artifacts"
    make_catalog(args.matmuls, args.pointwise, args.ffn, catalog)
    common = ["--model", model]
    invoke(runner, ["--mode", "cpu", *common, "--output", work / "cpu"], "cpu", work)
    baseline, reference = read_run(work / "cpu", "cpu")
    require(baseline["prefill_tokens"] == 5 and baseline["decode_steps"] == 3,
            "fixture requires five prefill tokens and three decode steps")
    replay = ["--tokens", work / "cpu/tokens.txt"]
    expected_tokens = (work / "cpu/tokens.txt").read_bytes()
    expected_argmax = [row["argmax_token"] for row in lines(work / "cpu/steps.jsonl")]
    results = {}
    for repeat in range(args.repeats):
        # Alternate the order to reduce a systematic member/region warmup bias.
        modes = ("members", "regions") if repeat % 2 == 0 else ("regions", "members")
        pair = {}
        for mode in modes:
            name = f"{mode}-{repeat}"
            output = work / name
            invoke(runner, ["--mode", "coral", *common, *replay, "--artifacts", catalog,
                            "--observation", "backend", "--ffn", "off" if mode == "members" else "auto",
                            "--output", output], name, work)
            summary, actual = read_run(output, "coral")
            require((output / "tokens.txt").read_bytes() == expected_tokens, "replay inputs changed")
            steps = lines(output / "steps.jsonl")
            require([row["argmax_token"] for row in steps] == expected_argmax, "argmax changed")
            error = compare(actual, reference)
            stats = summary["stats"]
            full = mode == "regions"
            inspect_regions(output, 20 if full else 0)
            require(stats["ffn_members_skipped"] == (80 if full else 0), "wrong skipped members")
            require(stats["calls_completed"] == (104 if full else 164), "unexpected VM calls")
            require(stats["coral_dispatches"] == (256 if full else 276), "unexpected dispatches")
            require(stats["matmuls"] == (84 if full else 144), "wrong matmul route")
            require(stats["pointwise_calls"] == (0 if full else 20), "wrong pointwise route")
            require(stats["ffn_guard_rejections"] == 0, "model failed a live guard")
            if full:
                require(stats["ffn_weight_preparations"] == 5 and stats["ffn_weight_cache_hits"] == 15,
                        "layer weights were not reused across profiles/decode")
            if repeat:
                require((output / "logits.f32").read_bytes() ==
                        (work / f"{mode}-0/logits.f32").read_bytes(), "fresh process changed logits")
            results[name] = {"max_abs_error_vs_cpu": error, "stats": stats}
            pair[mode] = actual
        results[f"pair-{repeat}"] = {"max_abs_error_regions_vs_members": compare(pair["regions"], pair["members"])}
    controls = {}
    for mode in ("callbacks", "missing-all", "missing-prefill", "missing-member", "corrupt", "wrong-shape"):
        artifacts = catalog
        if mode != "callbacks":
            artifacts = work / f"{mode}-artifacts"
            shutil.copytree(catalog, artifacts)
            if mode == "missing-all":
                (artifacts / "ffn.tsv").unlink()
            elif mode == "missing-prefill":
                (artifacts / "ffn.tsv").write_text("coral-m4-ffn-v1\n1 64 172 real-decode\n")
            elif mode == "missing-member":
                (artifacts / "pointwise.tsv").unlink()
            elif mode == "corrupt":
                (artifacts / "real-prefill/region.vmfb").write_bytes(b"invalid region")
            else:
                (artifacts / "ffn.tsv").write_text("coral-m4-ffn-v1\n1 64 172 real-prefill\n5 64 172 real-prefill\n")
        output = work / mode
        success = mode not in ("corrupt", "wrong-shape")
        invoke(runner, ["--mode", "coral", *common, *replay, "--artifacts", artifacts,
                        "--observation", "nodes" if mode == "callbacks" else "backend",
                        "--output", output], mode, work, expect_success=success)
        if not success:
            require(not (output / "summary.json").exists(), "failed graph published completion")
            controls[mode] = {"rejected": True}
            continue
        summary, actual = read_run(output, "coral")
        error = compare(actual, reference)
        inspect_regions(output, 16 if mode == "missing-prefill" else 0)
        controls[mode] = {"max_abs_error_vs_cpu": error, "stats": summary["stats"]}
    # Process evidence complements PATH isolation. It does not claim that the
    # compiler binaries have been physically removed from this workstation.
    strace = shutil.which("strace")
    require(strace is not None, "strace is required for process execution evidence")
    command = [strace, "-f", "-e", "trace=execve", "-o", str(work / "runner-exec.trace"),
               str(runner), "--mode", "coral", "--model", str(model), "--tokens", str(work / "cpu/tokens.txt"),
               "--artifacts", str(catalog), "--observation", "backend", "--output", str(work / "isolated")]
    env = dict(os.environ, PATH="/nonexistent")
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=300, check=False)
    (work / "isolated.stdout.log").write_text(result.stdout)
    (work / "isolated.stderr.log").write_text(result.stderr)
    require(result.returncode == 0, "isolated inference failed")
    executions = [line for line in (work / "runner-exec.trace").read_text().splitlines() if "execve(" in line]
    require(len(executions) == 1 and str(runner) in executions[0], "runner launched a child executable")
    isolated, actual = read_run(work / "isolated", "coral")
    compare(actual, reference)
    inspect_regions(work / "isolated", 20)
    timing = {}
    for mode in ("members", "regions"):
        prefill, warm, vm_warm, region_warm = [], [], [], []
        for repeat in range(args.repeats):
            output = work / f"{mode}-{repeat}"
            steps = lines(output / "steps.jsonl")
            prefill.append(steps[0]["wall_ms"])
            warm.extend(row["wall_ms"] for row in steps[2:])
            vm_warm.extend(row["stats"]["invoke_ms"] for row in steps[2:])
            if mode == "regions":
                region_warm.extend(row["invoke_ms"] for row in lines(output / "trace.jsonl")
                                   if row["kind"] == "ffn_region" and row["step"] >= 2)
        timing[mode] = {"prefill_wall_ms": distribution(prefill),
                        "warm_decode_wall_ms": distribution(warm),
                        "warm_decode_vm_ms": distribution(vm_warm)}
        if region_warm:
            timing[mode]["warm_ffn_vm_ms"] = distribution(region_warm)
    require_clean()
    report = {"status": "passed", "llama_revision": revision, "llama_worktree_clean": True,
              "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
              "runner_sha256": hashlib.sha256(runner.read_bytes()).hexdigest(),
              "paired_process_runs": args.repeats, "atol": ATOL, "rtol": RTOL,
              "results": results, "controls": controls, "timing": timing,
              "isolated_execution": {"execve_count": len(executions), "ffn_calls": isolated["stats"]["ffn_calls"],
                                     "compiler_physically_absent": False}}
    (work / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "passed", "report": str(work / "report.json"),
                      "timing": timing, "region_stats": results["regions-0"]["stats"],
                      "errors": {key: row.get("max_abs_error_vs_cpu", row.get("max_abs_error_regions_vs_members"))
                                 for key, row in results.items()}}), flush=True)
if __name__ == "__main__":
    main()
