#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Ahead-of-time compile verified F32 FFN members; retain all artifacts and logs."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("build/m3-pointwise-artifacts"))
    parser.add_argument("--shapes", nargs="+", default=["1x172", "5x172"])
    args = parser.parse_args()
    tool = args.tool.resolve(strict=True)
    compiler = args.compiler.resolve(strict=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="pointwise-", dir=args.output_root)).resolve()
    rows = ["coral-m3-pointwise-v1"]
    for shape in args.shapes:
        m, n = map(int, shape.split("x"))
        if min(m, n) <= 0:
            raise ValueError("dimensions must be positive")
        for op in ("silu", "multiply", "swiglu"):
            stem = f"{op}_m{m}_n{n}"
            mlir = work / f"{stem}.mlir"
            subprocess.run([str(tool), "emit-pointwise", op, str(m), str(n), str(mlir)], check=True)
            command = [
                str(compiler), str(mlir), "--iree-input-type=stablehlo",
                "--iree-execution-model=async-internal",
                "--iree-hal-target-device=local",
                "--iree-hal-local-target-device-backends=llvm-cpu",
                "--iree-llvmcpu-target-cpu=host", "--iree-hal-target-device=coralnpu",
                "--coralnpu-fast-math=false", "--coralnpu-roofline-speedup-threshold=0",
                "--coralnpu-dump-affinity-profile-format=pretty",
                "-o", str(work / f"{stem}.vmfb"),
            ]
            (work / f"{stem}.command.json").write_text(json.dumps(command, indent=2) + "\n")
            with (work / f"{stem}.compile.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
            rows.append(f"{op}\t{m}\t{n}\t{stem}.vmfb")
            print(f"Compiled {stem}", flush=True)
    (work / "pointwise.tsv").write_text("\n".join(rows) + "\n")
    print(f"Artifacts: {work}", flush=True)

if __name__ == "__main__":
    main()
