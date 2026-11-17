#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Validate standalone FFN backend capabilities, execution and catalog failures."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from verify_ffn_members import copy_catalog
from verify_runner import invoke, require

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--matmuls", type=Path, required=True)
    parser.add_argument("--pointwise", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("build/m3-member-adapter-results"))
    args = parser.parse_args()
    runner = args.runner.resolve(strict=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="adapter-", dir=args.output_root)).resolve()
    print(f"Evidence: {work}", flush=True)
    results = []
    for mode in ("capabilities", "execute", "preflight"):
        catalog = work / mode
        catalog.mkdir()
        copy_catalog(args.matmuls, catalog, "profiles.tsv")
        copy_catalog(args.pointwise, catalog, "pointwise.tsv")
        result = invoke(runner, [mode, catalog], mode, work)
        record = json.loads(result.stdout)
        require(record["status"] == "passed" and record["checks"] > 0, "missing checks")
        results.append(record)
    original = (args.pointwise / "pointwise.tsv").read_text().splitlines()
    header, row = original[:2]
    op, m, n, filename = row.split()
    cases = {
        "header": "unknown\n" + row,
        "empty": header,
        "duplicate": "\n".join([header, row, row]),
        "unknown": f"{header}\nunknown {m} {n} {filename}",
        "zero": f"{header}\n{op} 0 {n} {filename}",
        "overflow": f"{header}\n{op} 2147483647 {n} {filename}",
        "extra": f"{header}\n{row} extra",
        "traversal": f"{header}\n{op} {m} {n} ../outside.vmfb",
        "missing": f"{header}\n{op} {m} {n} missing.vmfb",
        "symlink": f"{header}\n{op} {m} {n} escape.vmfb",
    }
    (work / "outside.vmfb").write_bytes(b"outside catalog")
    for name, text in cases.items():
        catalog = work / name
        catalog.mkdir()
        copy_catalog(args.matmuls, catalog, "profiles.tsv")
        copy_catalog(args.pointwise, catalog, "pointwise.tsv")
        (catalog / "pointwise.tsv").write_text(text + "\n")
        if name == "symlink": (catalog / "escape.vmfb").symlink_to(work / "outside.vmfb")
        invoke(runner, ["catalog", catalog], name, work, expect_success=False)
    report = {"status": "passed", "results": results, "negative_catalogs": len(cases),
              "checks": sum(result["checks"] for result in results),
              "runner_sha256": hashlib.sha256(runner.read_bytes()).hexdigest()}
    (work / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
if __name__ == "__main__":
    main()
