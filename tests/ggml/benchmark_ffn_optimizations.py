#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Paired simulator measurements for exact, layout and row-capacity policies."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import tempfile
import time

from verify_runner import read_run, require
from verify_ffn_members import copy_catalog, compare
from verify_unpatched import ROOT, require_clean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native-catalog", "layout-catalog", "bucket-evidence"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    require(args.repeats >= 3, "at least three paired runs")
    revision = require_clean()
    work = Path(tempfile.mkdtemp(prefix="m5-performance-", dir=ROOT / "build"))
    print(f"Evidence: {work}", flush=True)
    data = json.loads((args.bucket_evidence / "report.json").read_text())
    require(data["status"] == "passed", "bucket evidence has not passed")
    catalogs = {"native-exact": args.native_catalog.resolve(strict=True),
                "p1-exact": args.layout_catalog.resolve(strict=True)}
    for label, layout, capacities in (("p2-only-8", "native", (8, 8)),
                                      ("p1-p2-8", "transposed", (8, 8)),
                                      ("p1-p2-1-8", "transposed", (1, 8))):
        catalog = work / label
        catalog.mkdir()
        for index in ("profiles.tsv", "pointwise.tsv"):
            copy_catalog(args.native_catalog, catalog, index)
        record = [c for c in data["cases"] if c["case"] == layout + "-m5-d64-f172"][0]
        shutil.copytree(args.bucket_evidence / "cache/entries" / record["key"], catalog / "capacity-8")
        if capacities[0] == 1:
            shutil.copytree(args.layout_catalog / "real-decode", catalog / "capacity-1")
        rows = ["coral-m5-ffn-v2"]
        for m, capacity in zip((1, 5), capacities):
            rows.append(f"{m} 64 172 capacity-{capacity} {layout} {capacity}")
        (catalog / "ffn.tsv").write_text("\n".join(rows) + "\n")
        catalogs[label] = catalog
    runner = ROOT / "build/m1/coral-llama-runner"
    model = ROOT / "models/stories260K-f32.gguf"

    def run(label, arguments):
        start = time.perf_counter()
        # Pipes let communicate observe EOF promptly. wait(timeout) with file
        # redirection polls at up to 50 ms and distorts short process timings.
        result = subprocess.run(list(map(str, [runner, "--model", model, *arguments,
                                  "--output", work / label])), cwd=ROOT,
                                capture_output=True, text=True, timeout=60, check=False)
        elapsed = (time.perf_counter() - start) * 1000
        (work / (label + ".stdout.log")).write_text(result.stdout)
        (work / (label + ".stderr.log")).write_text(result.stderr)
        require(result.returncode == 0, label + " failed")
        return elapsed

    run("cpu", ["--mode", "cpu"])
    _, expected = read_run(work / "cpu", "cpu")
    order = list(catalogs)
    measurements = {name: [] for name in order}
    for repeat in range(args.repeats):
        rotation = order[repeat % len(order):] + order[:repeat % len(order)]
        for name in rotation:
            label = name + "-" + str(repeat)
            wall = run(label, ["--mode", "coral", "--artifacts", catalogs[name],
                               "--observation", "backend", "--tokens", work / "cpu/tokens.txt"])
            summary, actual = read_run(work / label, "coral")
            error = compare(actual, expected)
            stats = summary["stats"]
            require(stats["jit_requests"] == 0 and stats["ffn_calls"] == 20 and
                    stats["matmuls"] == 84 and stats["ffn_weight_preparations"] == 5 and
                    stats["ffn_weight_cache_hits"] == 15, "incomplete comparable workload")
            trace = [json.loads(line) for line in (work / label / "trace.jsonl").read_text().splitlines()]
            regions = [row for row in trace if row["kind"] == "ffn_region"]
            require(len(regions) == 20, "missing FFN trace")
            measurements[name].append({"process_wall_ms": wall, "stats": stats,
                                       "ffn_invoke_ms": sum(row["invoke_ms"] for row in regions),
                                       "ffn_m1_median_ms": statistics.median(
                                           row["invoke_ms"] for row in regions if row["m"] == 1),
                                       "ffn_m5_median_ms": statistics.median(
                                           row["invoke_ms"] for row in regions if row["m"] == 5),
                                       "max_abs_vs_cpu": error})
        print(f"Pair {repeat+1}/{args.repeats}: passed", flush=True)
    results = {}
    for name, samples in measurements.items():
        result = {"runs": samples}
        for metric in ("process_wall_ms", "ffn_invoke_ms", "ffn_m1_median_ms", "ffn_m5_median_ms"):
            values = [row[metric] for row in samples]
            result[metric] = {"median": statistics.median(values), "min": min(values), "max": max(values)}
        for metric in ("invoke_ms", "module_load_ms", "ffn_boundary_ms"):
            result[metric] = statistics.median(row["stats"][metric] for row in samples)
        for metric in ("kernel_instructions", "coral_dispatches", "logical_upload_bytes",
                       "logical_download_bytes", "ffn_padded_rows", "module_loads"):
            values = {row["stats"][metric] for row in samples}
            require(len(values) == 1, "non-deterministic accounting: " + name + metric)
            result[metric] = values.pop()
        results[name] = result
    report = {"status": "passed", "llama_revision": revision, "repeats": args.repeats,
              "runner_sha256": hashlib.sha256(runner.read_bytes()).hexdigest(),
              "strategies": results, "timing_scope": "local MPACT simulator; AOT code; fresh process per run",
              "pair_order": "rotate strategy order on each repeat; one process at a time"}
    (work / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "passed", "report": str(work / "report.json")}), flush=True)


if __name__ == "__main__":
    main()
