#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure the diagnostic M2 runner with fixed-token CPU/MPACT process pairs."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import select
import signal
import statistics
import subprocess
import tempfile
import time

from verify_runner import ATOL, RTOL, read_run, require


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values):
    ordered = sorted(values)
    index = (len(ordered) - 1) * 0.95
    lower = math.floor(index)
    fraction = index - lower
    p95 = ordered[lower] * (1 - fraction) + ordered[min(lower + 1, len(ordered) - 1)] * fraction
    return {"n": len(values), "min": min(values), "p50": statistics.median(values),
            "p95": p95, "max": max(values), "mean": statistics.mean(values),
            "stddev": statistics.pstdev(values)}


def measure(runner, model, tokens, artifacts, mode, directory):
    env = os.environ.copy()
    env["PATH"] = "/nonexistent"
    command = ["/usr/bin/time", "-f", "%M", "-o", str(directory) + ".rss-kib.txt",
               str(runner), "--mode", mode, "--model", str(model),
               "--tokens", str(tokens), "--output", str(directory)]
    if mode == "coral":
        command += ["--artifacts", str(artifacts)]
    with open(str(directory) + ".stdout.log", "w") as out, open(str(directory) + ".stderr.log", "w") as err:
        start = time.perf_counter_ns()
        process = subprocess.Popen(command, stdout=out, stderr=err, env=env, start_new_session=True)
        # Python wait(timeout) polls with increasing sleeps on POSIX and
        # overstates short process latency. A Linux pidfd signals actual exit.
        pidfd = os.pidfd_open(process.pid)
        try:
            ready, _, _ = select.select([pidfd], [], [], 300)
            if not ready:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise RuntimeError(f"{mode} measurement timed out")
            status = process.wait()
        finally:
            os.close(pidfd)
        elapsed = (time.perf_counter_ns() - start) / 1e6
    require(status == 0, f"runner failed: {directory.name}; see its stderr log")
    peak_rss_kib = int(Path(str(directory) + ".rss-kib.txt").read_text().strip())
    summary, values = read_run(directory, mode)
    steps = [json.loads(line) for line in (directory / "steps.jsonl").read_text().splitlines()]
    require((directory / "tokens.txt").read_bytes() == tokens.read_bytes(), "input tokens changed")
    return {"mode": mode, "process_wall_ms": elapsed, "peak_rss_kib": peak_rss_kib,
            "summary": summary, "steps": steps}, values


def summarize(records):
    result = {}
    for mode in ("cpu", "coral"):
        runs = [run for run in records if run["mode"] == mode and not run["warmup"]]
        metrics = {
            "process_wall_ms": [run["process_wall_ms"] for run in runs],
            "peak_rss_mib": [run["peak_rss_kib"] / 1024 for run in runs],
            "prefill_wall_ms": [run["steps"][0]["wall_ms"] for run in runs],
            "first_decode_wall_ms": [run["steps"][1]["wall_ms"] for run in runs],
            "warm_decode_wall_ms": [step["wall_ms"] for run in runs for step in run["steps"][2:]],
            "total_step_wall_ms": [sum(step["wall_ms"] for step in run["steps"]) for run in runs],
            "module_load_ms": [run["summary"]["stats"]["module_load_ms"] for run in runs],
            "vm_invoke_ms": [run["summary"]["stats"]["invoke_ms"] for run in runs],
            "non_step_process_ms": [
                run["process_wall_ms"] - sum(step["wall_ms"] for step in run["steps"]) for run in runs],
            "warm_invoke_share_percent": [
                100 * step["stats"]["invoke_ms"] / step["wall_ms"]
                for run in runs for step in run["steps"][2:]],
        }
        result[mode] = {name: distribution(values) for name, values in metrics.items()}
        warm_ms = result[mode]["warm_decode_wall_ms"]["p50"]
        result[mode]["warm_decode_tokens_per_second_from_p50"] = 1000 / warm_ms
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup-pairs", type=int, default=2)
    parser.add_argument("--output-root", type=Path, default=Path("build/m2-benchmarks"))
    args = parser.parse_args()
    require(args.repeats >= 2 and args.warmup_pairs >= 0, "invalid measurement count")
    runner, model, tokens, artifacts = [path.resolve() for path in
                                       (args.runner, args.model, args.tokens, args.artifacts)]
    args.output_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="paired-", dir=args.output_root)).resolve()
    print(f"Results: {work}", flush=True)
    root = Path(__file__).resolve().parents[2]
    source_files = ["CMakeLists.txt", "src/ggml/CMakeLists.txt", "src/ggml/ggml_coral.cc",
                    "src/ggml/ggml_coral.h", "src/ggml/llama_runner.cc", "src/support/json.h",
                    "src/runtime/matmul_runtime.cc", "src/runtime/matmul_runtime.h",
                    "tests/ggml/verify_runner.py", "tests/ggml/benchmark_runner.py"]
    identity = {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "runner_sha256": digest(runner), "model_sha256": digest(model), "tokens_sha256": digest(tokens),
        "source_sha256": {name: digest(root / name) for name in source_files},
        "vmfb_sha256": {path.name: digest(path) for path in sorted(artifacts.glob("*.vmfb"))},
        "manifest_sha256": digest(artifacts / "profiles.tsv"),
    }
    environment = {"system": platform.system(), "machine": platform.machine(),
                   "kernel": platform.release(), "available_logical_cpus": os.cpu_count(),
                   "python": platform.python_version(), "inference_path": "/nonexistent"}
    # Do not include hostname, username, absolute paths or machine identifiers.
    records = []
    max_error = 0.0
    argmax_matches = 0
    start_time = datetime.now(timezone.utc).isoformat()
    for pair in range(args.warmup_pairs + args.repeats):
        order = ("cpu", "coral") if pair % 2 == 0 else ("coral", "cpu")
        paired = {}
        for mode in order:
            record, values = measure(runner, model, tokens, artifacts, mode, work / f"{pair:02d}-{mode}")
            record.update(pair=pair, warmup=pair < args.warmup_pairs)
            records.append(record)
            paired[mode] = (record, values)
        cpu, coral = paired["cpu"], paired["coral"]
        require(len(cpu[1]) == len(coral[1]), "different logit counts")
        for reference, actual in zip(cpu[1], coral[1]):
            error = abs(reference - actual)
            require(error <= ATOL + RTOL * abs(reference), "logit tolerance exceeded")
            max_error = max(max_error, error)
        for a, b in zip(cpu[0]["steps"], coral[0]["steps"]):
            require(a["argmax_token"] == b["argmax_token"], "argmax changed")
            argmax_matches += 1
        with (work / "runs.json").open("w") as file:
            json.dump(records, file, indent=2)
        print(f"Pair {pair + 1}: CPU/Coral logits and dispatch evidence passed", flush=True)

    fields = ["pair", "warmup", "mode", "step", "phase", "wall_ms", "coral_nodes",
              "dispatches", "instructions", "module_loads", "weight_cache_hits",
              "binding_allocations", "logical_upload_bytes", "logical_download_bytes",
              "vm_invoke_ms", "module_load_ms", "process_wall_ms", "peak_rss_kib"]
    with (work / "measurements.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for run in records:
            for step in run["steps"]:
                stats = step["stats"]
                writer.writerow({
                    "pair": run["pair"], "warmup": run["warmup"], "mode": run["mode"],
                    "step": step["step"], "phase": step["phase"], "wall_ms": step["wall_ms"],
                    "coral_nodes": step["coral_nodes"], "dispatches": stats["coral_dispatches"],
                    "instructions": stats["kernel_instructions"], "module_loads": stats["module_loads"],
                    "weight_cache_hits": stats["weight_cache_hits"],
                    "binding_allocations": stats["binding_allocations"],
                    "logical_upload_bytes": stats["logical_upload_bytes"],
                    "logical_download_bytes": stats["logical_download_bytes"],
                    "vm_invoke_ms": stats["invoke_ms"], "module_load_ms": stats["module_load_ms"],
                    "process_wall_ms": run["process_wall_ms"], "peak_rss_kib": run["peak_rss_kib"],
                })
    # Changing an input or executable during the experiment invalidates it.
    require(digest(runner) == identity["runner_sha256"] and digest(model) == identity["model_sha256"]
            and digest(tokens) == identity["tokens_sha256"], "experiment input changed")
    report = {"schema": "coral-m2-benchmark-v1", "started_at_utc": start_time,
              "repeats": args.repeats, "warmup_pairs": args.warmup_pairs,
              "trace_enabled": True, "fresh_process_per_run": True, "alternating_order": True,
              "process_wait_method": "Linux pidfd + select (no polling sleeps)",
              "p95_method": "linear interpolation at (n-1)*0.95",
              "warm_decode_definition": "decode steps 2 and later, within each fresh process",
              "max_abs_logit_error_all_pairs": max_error, "argmax_checks": argmax_matches,
              "atol": ATOL, "rtol": RTOL, "identity": identity, "environment": environment,
              "metrics": summarize(records)}
    (work / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"max_abs_error": max_error, "metrics": report["metrics"]}), flush=True)


if __name__ == "__main__":
    main()
