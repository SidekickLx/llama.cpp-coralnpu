#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Exercise complete FFN admission/execution with the unmodified GGML scheduler."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile
from verify_runner import invoke, require
from verify_ffn_members import copy_catalog
from verify_unpatched import require_clean

def make_catalog(matmuls, pointwise, ffn, output):
    output.mkdir()
    copy_catalog(matmuls, output, "profiles.tsv")
    copy_catalog(pointwise, output, "pointwise.tsv")
    for name in ("real-decode", "real-prefill"):
        shutil.copytree(ffn / name, output / name)
    (output / "ffn.tsv").write_text(
        "coral-m4-ffn-v1\n1 64 172 real-decode\n5 64 172 real-prefill\n")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, default=Path("build/m1/coral-ffn-region-test"))
    parser.add_argument("--matmuls", type=Path, required=True)
    parser.add_argument("--pointwise", type=Path, required=True)
    parser.add_argument("--ffn", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("build/m4-region-backend-results"))
    args = parser.parse_args()
    revision = require_clean()
    args.output_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="backend-", dir=args.output_root)).resolve()
    print(f"Evidence: {work}", flush=True)
    artifacts = work / "artifacts"
    make_catalog(args.matmuls, args.pointwise, args.ffn, artifacts)
    runner = args.runner.resolve(strict=True)
    cases = {}
    scheduled = 0
    for mode in ("mutable", "immutable", "external-consumer", "external-view",
                 "callbacks", "callback-subdivision", "stale-metadata", "disabled",
                 "missing", "missing-member", "corrupt"):
        catalog = artifacts
        if mode in ("missing", "missing-member", "corrupt"):
            catalog = work / f"{mode}-artifacts"
            shutil.copytree(artifacts, catalog)
            if mode == "missing":
                (catalog / "ffn.tsv").unlink()
            elif mode == "missing-member":
                (catalog / "pointwise.tsv").unlink()
            else:
                (catalog / "real-decode/region.vmfb").write_bytes(b"invalid region VMFB")
        result = invoke(runner, [mode, catalog], mode, work)
        records = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
        report = records[-1]
        require(report["status"] == "passed", f"{mode}: failed")
        scheduled += sum("case" in row for row in records)
        cases[mode] = report
    invalid = {
        "header": "other\n1 64 172 real-decode\n",
        "duplicate": "coral-m4-ffn-v1\n1 64 172 real-decode\n1 64 172 real-decode\n",
        "zero": "coral-m4-ffn-v1\n0 64 172 real-decode\n",
        "negative": "coral-m4-ffn-v1\n-1 64 172 real-decode\n",
        "overflow": "coral-m4-ffn-v1\n2147483647 64 172 real-decode\n",
        "empty": "coral-m4-ffn-v1\n",
        "extra": "coral-m4-ffn-v1\n1 64 172 real-decode extra\n",
        "traversal": "coral-m4-ffn-v1\n1 64 172 ../artifacts/real-decode\n",
        "dot": "coral-m4-ffn-v1\n1 64 172 .\n",
        "missing-package": "coral-m4-ffn-v1\n1 64 172 absent\n",
        "symlink": "coral-m4-ffn-v1\n1 64 172 linked\n",
    }
    bad = work / "bad-catalog"
    shutil.copytree(artifacts, bad)
    (bad / "linked").symlink_to(artifacts / "real-decode", target_is_directory=True)
    for name, content in invalid.items():
        (bad / "ffn.tsv").write_text(content)
        invoke(runner, ["catalog", bad], f"catalog-{name}", work, expect_success=False)
    require_clean()
    report = {"status": "passed", "llama_revision": revision,
              "llama_worktree_clean": True, "scheduled_graphs": scheduled,
              "executions_per_graph": 3, "catalog_rejections": len(invalid),
              "cases": cases, "compiler_path": "/nonexistent",
              "atol": 1e-4, "rtol": 1e-4}
    (work / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
if __name__ == "__main__":
    main()
