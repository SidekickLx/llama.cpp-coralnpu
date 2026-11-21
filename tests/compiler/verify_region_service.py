#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Exercise AOT and direct service callers against one semantic contract."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "compiler"))
from region_service import Toolchain, canonicalize_region, compile_region


def main():
    source = Path(sys.argv[1]).resolve(strict=True)
    work = Path(tempfile.mkdtemp(prefix="m5-service-", dir=ROOT / "build"))
    print(f"Evidence: {work}", flush=True)
    toolchain = Toolchain()
    profile = (2, 3, 4)
    description = toolchain.describe(profile)
    target = description["compatibility"]
    policy = description["contract"]["numeric_policy"]
    canonical = canonicalize_region(source.read_bytes(), profile,
                                    toolchain=toolchain, evidence=work / "prepare")
    api = compile_region(canonical, profile, target, policy, output=work / "api")
    subprocess.run([sys.executable, ROOT / "compiler/compile_ffn.py", "--input", source,
                    "--m", "2", "--d", "3", "--f", "4", "--output", work / "aot"], check=True)
    a = json.loads((api / "manifest.json").read_text())
    b = json.loads((work / "aot/manifest.json").read_text())
    assert a["program"] == b["program"] and a["contract"] == b["contract"]
    assert a["compiler"] == b["compiler"] and a["reflection"] == b["reflection"]
    negatives = [
        ("target", canonical, profile, {}, policy),
        ("policy", canonical, profile, target, "fast"),
        ("shape", canonical, (1, 3, 4), target, policy),
        ("semantics", canonical.replace(b"stablehlo.exponential", b"stablehlo.abs"), profile, target, policy),
        ("noncanonical", b"\n" + canonical, profile, target, policy),
    ]
    for label, ir, dims, compatibility, numeric in negatives:
        try:
            compile_region(ir, dims, compatibility, numeric, output=work / label)
        except (RuntimeError, subprocess.CalledProcessError):
            assert not (work / label).exists()
        else:
            raise AssertionError(label + " was accepted")
    subprocess.run([ROOT / "build/m1/coral-ffn-runtime-test", "run", api, "2", "3", "4"],
                   check=True, stdout=(work / "runtime.log").open("w"))
    result = {"status": "passed", "callers": ["AOT", "service"], "negative_cases": 5,
              "runtime_calls": 14, "program_sha256": a["program"]["semantic_sha256"]}
    (work / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
