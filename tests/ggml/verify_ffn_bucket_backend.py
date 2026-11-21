#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Accept explicit AOT row capacities in the unmodified GGML scheduler."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from verify_ffn_members import copy_catalog, compare
from verify_ffn_regions import inspect_regions
from verify_runner import read_run, require
from verify_unpatched import ROOT, require_clean
sys.path.insert(0, str(ROOT / "compiler"))
from region_service import invoke


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bucket-evidence", "matmuls", "pointwise"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    revision = require_clean()
    source = args.bucket_evidence.resolve(strict=True)
    report = json.loads((source / "report.json").read_text())
    require(report["status"] == "passed", "bucket evidence has not passed")
    selected = [c for c in report["cases"] if c["case"] == "transposed-m5-d64-f172"][0]
    package = source / "cache/entries" / selected["key"]
    work = Path(tempfile.mkdtemp(prefix="m5-p2-backend-", dir=ROOT / "build"))
    print(f"Evidence: {work}", flush=True)
    catalog = work / "artifacts"
    catalog.mkdir()
    copy_catalog(args.matmuls, catalog, "profiles.tsv")
    copy_catalog(args.pointwise, catalog, "pointwise.tsv")
    shutil.copytree(package, catalog / "capacity-8")
    (catalog / "ffn.tsv").write_text(
        "coral-m5-ffn-v2\n1 64 172 capacity-8 transposed 8\n5 64 172 capacity-8 transposed 8\n")
    runner = ROOT / "build/m1/coral-llama-runner"
    model = ROOT / "models/stories260K-f32.gguf"
    invoke([runner, "--mode", "cpu", "--model", model, "--output", work / "cpu"], work, "cpu")
    invoke([runner, "--mode", "coral", "--model", model, "--artifacts", catalog,
            "--jit-config", work / "must-not-open.json", "--observation", "backend",
            "--tokens", work / "cpu/tokens.txt", "--output", work / "regions-0"], work, "regions")
    _, expected = read_run(work / "cpu", "cpu")
    summary, actual = read_run(work / "regions-0", "coral")
    error = compare(actual, expected)
    inspect_regions(work / "regions-0", 20, {1: 3, 5: 3})
    stats = summary["stats"]
    require(stats["ffn_module_cache_hits"] == 19 and stats["jit_requests"] == 0 and
            stats["jit_initialization_ms"] == 0 and stats["ffn_weight_preparations"] == 5 and
            stats["ffn_weight_cache_hits"] == 15 and stats["ffn_padded_rows"] ==
            stats["ffn_cropped_rows"] == 124, "capacity/module/weight reuse failed")
    for mode in ("mutable", "immutable", "weight-generation", "external-consumer",
                 "external-view", "callbacks", "callback-subdivision"):
        invoke([ROOT / "build/m1/coral-ffn-region-test", mode, catalog], work, mode)
    # Catalog row corruption must fail at registration before numerical work.
    bad = work / "bad-catalog"
    bad.mkdir()
    copy_catalog(args.matmuls, bad, "profiles.tsv")
    copy_catalog(args.pointwise, bad, "pointwise.tsv")
    shutil.copytree(package, bad / "capacity-8")
    invalid_rows = [
        "5 64 172 capacity-8 transposed 4",
        "5 64 172 capacity-8 transposed 0",
        "5 64 172 capacity-8 transposed 536870911",
        "5 64 172 capacity-8 transposed",
        "5 64 172 capacity-8 unknown 8",
        "5 64 172 capacity-8 native 8 extra",
        "5 64 172 capacity-8 native 8\n5 64 172 capacity-8 native 8",
    ]
    for i, row in enumerate(invalid_rows):
        (bad / "ffn.tsv").write_text("coral-m5-ffn-v2\n" + row + "\n")
        result = subprocess.run([ROOT / "build/m1/coral-ffn-region-test", "catalog", bad],
                                capture_output=True, text=True)
        require(result.returncode != 0, "invalid capacity catalog accepted")
        (work / f"catalog-negative-{i}.log").write_text(result.stderr)
    # A different explicitly listed package with the same capacity must still
    # be loaded/validated, even after a healthy prefill module is already cached.
    shutil.copytree(package, bad / "corrupt")
    (bad / "corrupt/region.vmfb").write_bytes(b"deliberately corrupt bucket package")
    (bad / "ffn.tsv").write_text(
        "coral-m5-ffn-v2\n5 64 172 capacity-8 transposed 8\n1 64 172 corrupt transposed 8\n")
    result = subprocess.run([runner, "--mode", "coral", "--model", model, "--artifacts", bad,
                             "--jit-config", work / "must-not-open.json", "--observation", "backend",
                             "--tokens", work / "cpu/tokens.txt", "--output", work / "bad-model"],
                            capture_output=True, text=True, timeout=60)
    (work / "bad-model.stderr.log").write_text(result.stderr)
    require(result.returncode != 0 and "region.vmfb: SHA-256 mismatch" in result.stderr and
            not (work / "bad-model/summary.json").exists(), "listed package escaped validation")
    trace = [json.loads(line) for line in (work / "bad-model/trace.jsonl").read_text().splitlines()]
    require(sum(row["kind"] == "ffn_region" for row in trace) == 4,
            "control did not fail after four healthy prefill FFNs")
    result = {"status": "passed", "llama_revision": revision, "model_stats": stats,
              "max_abs_vs_cpu": error, "catalog_negative_cases": len(invalid_rows),
              "different_aot_package_revalidated": True,
              "same_capacity_module_shared": True}
    (work / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "passed", "report": str(work / "report.json")}), flush=True)


if __name__ == "__main__":
    main()
