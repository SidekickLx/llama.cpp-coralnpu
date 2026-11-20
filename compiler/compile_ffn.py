#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Normalize imported StableHLO and call the shared FFN compilation service."""
import argparse
from pathlib import Path
import tempfile
from region_service import ROOT, Toolchain, canonicalize_region, compile_region, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("m", "d", "f"):
        parser.add_argument("--" + name, type=int, required=True)
    defaults = Toolchain()
    for name in ("tool", "checker", "compiler"):
        parser.add_argument("--" + name, type=Path, default=getattr(defaults, name))
    args = parser.parse_args()
    require(not args.output.exists() and not args.output.is_symlink(), "output must be a new directory")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    evidence = Path(tempfile.mkdtemp(prefix=args.output.name + ".source-", dir=args.output.parent.resolve()))
    toolchain = Toolchain(args.tool, args.checker, args.compiler)
    profile = (args.m, args.d, args.f)
    description = toolchain.describe(profile)
    canonical = canonicalize_region(args.input.read_bytes(), profile,
                                    toolchain=toolchain, evidence=evidence / "normalization")
    output = compile_region(canonical, profile, description["compatibility"],
                            description["contract"]["numeric_policy"],
                            output=args.output, toolchain=toolchain)
    print(f"Source evidence: {evidence}", flush=True)
    print(f"Artifact: {output}", flush=True)


if __name__ == "__main__":
    main()
