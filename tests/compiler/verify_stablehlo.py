#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Exercise the typed builder and upstream MLIR/StableHLO verifier."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

def invoke(tool, args, work, label, success=True):
    result = subprocess.run([str(tool), *map(str, args)], capture_output=True,
                            text=True, timeout=30, check=False)
    (work / (label + ".stdout.txt")).write_text(result.stdout)
    (work / (label + ".stderr.txt")).write_text(result.stderr)
    expected = 0 if success else 1
    if result.returncode != expected:
        raise RuntimeError(f"{label}: expected {expected}, got {result.returncode}: {result.stderr}")
    if not success and not result.stderr.strip():
        raise RuntimeError(f"{label}: no diagnostic")
    return result

DOT = """module {
  func.func @matmul(%x: tensor<2x3xf32>, %w: tensor<4x3xf32>) -> tensor<2x4xf32> {
    %y = stablehlo.dot_general %x, %w, contracting_dims = [1] x [1],
      precision = [DEFAULT, DEFAULT] :
      (tensor<2x3xf32>, tensor<4x3xf32>) -> tensor<2x4xf32>
    return %y : tensor<2x4xf32>
  }
}
"""

def run(tool, work):
    shapes = [(1, 3, 4), (2, 3, 4), (1, 64, 172), (5, 64, 172)]
    for m, d, f in shapes:
        path = work / f"ffn-{m}-{d}-{f}.mlir"
        invoke(tool, ["emit-ffn", m, d, f, path], work, path.stem)
        invoke(tool, ["verify", path], work, path.stem + "-verify")
        text = path.read_text()
        if text.count("stablehlo.dot_general") != 3 or "custom_call" in text:
            raise RuntimeError("FFN must contain three standard dot operations")
        for op in ("broadcast_in_dim", "negate", "exponential", "add", "divide", "multiply"):
            if "stablehlo." + op not in text:
                raise RuntimeError(f"missing canonical SiLU/multiply op: {op}")
    path = work / "matmul.mlir"
    invoke(tool, ["emit-matmul", 2, 3, 4, path], work, "matmul")
    invoke(tool, ["verify", path], work, "matmul-verify")
    original = path.read_bytes()
    invoke(tool, ["emit-matmul", 2, 3, 4, path], work, "no-overwrite", False)
    if path.read_bytes() != original:
        raise RuntimeError("output was overwritten")
    negative = {
        "contracting-size": DOT.replace("4x3xf32", "4x5xf32"),
        "operand-type": """module {
          func.func @bad(%a: tensor<2xf32>, %b: tensor<2xi32>) -> tensor<2xf32> {
            %r = "stablehlo.add"(%a, %b) :
              (tensor<2xf32>, tensor<2xi32>) -> tensor<2xf32>
            return %r : tensor<2xf32>
          }
        }""",
        "contracting-axis": DOT.replace("[1] x [1]", "[2] x [1]"),
        "result-shape": DOT.replace("2x4xf32", "2x5xf32"),
        "undefined-ssa": DOT.replace("%x, %w,", "%missing, %w,"),
        "return-type": """module {
          func.func @bad(%x: tensor<2xf32>) -> tensor<3xf32> {
            return %x : tensor<2xf32>
          }
        }""",
        "broadcast-rank": """module {
          func.func @bad(%x: tensor<f32>) -> tensor<2x3xf32> {
            %y = stablehlo.broadcast_in_dim %x, dims = [0] :
              (tensor<f32>) -> tensor<2x3xf32>
            return %y : tensor<2x3xf32>
          }
        }""",
    }
    diagnostics = {}
    for name, text in negative.items():
        path = work / (name + ".mlir")
        path.write_text(text)
        result = invoke(tool, ["verify", path], work, name, False)
        diagnostics[name] = result.stderr.splitlines()[0]
    for index, dims in enumerate(((0, 3, 4), (1, -3, 4), (2**63 - 1, 3, 4))):
        path = work / f"invalid-{index}.mlir"
        invoke(tool, ["emit-ffn", *dims, path], work, f"invalid-{index}", False)
        if path.exists():
            raise RuntimeError("invalid contract produced an output")
    report = {"status": "passed", "positive_modules": 5, "negative_modules": len(negative),
              "invalid_specs": 3, "overwrite_rejected": True, "diagnostics": diagnostics}
    (work / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "passed", "evidence": str(work)}))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="verify-", dir=args.output_root.resolve()))
    run(args.tool.resolve(), work)

if __name__ == "__main__":
    main()
