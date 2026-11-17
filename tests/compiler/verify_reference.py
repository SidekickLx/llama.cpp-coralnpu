#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare the upstream interpreter with independent, scalar F32 arithmetic."""
import argparse
import json
import math
from pathlib import Path
import random
import re
import struct
import tempfile

from verify_stablehlo import invoke, DOT

ATOL = RTOL = 1e-4


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def tensor(shape, values):
    return {"dtype": "f32", "shape": shape, "values": values}


def dot(x, w):
    m, k = x["shape"]
    n, wk = w["shape"]
    if wk != k:
        raise RuntimeError("invalid oracle dimensions")
    result = []
    for row in range(m):
        for column in range(n):
            accumulator = f32(0)
            for contraction in range(k):
                product = f32(x["values"][row*k+contraction] * w["values"][column*k+contraction])
                accumulator = f32(accumulator + product)
            result.append(accumulator)
    return tensor([m, n], result)


def oracle_ffn(inputs):
    x, wg, wu, wd = inputs
    gate, up = dot(x, wg), dot(x, wu)
    silu = tensor(gate["shape"], [
        f32(value / f32(1.0 + f32(math.exp(f32(-value)))))
        for value in gate["values"]])
    hidden = tensor(gate["shape"], [
        f32(a*b) for a, b in zip(silu["values"], up["values"])])
    return [gate, up, silu, hidden, dot(hidden, wd)]


def compare(actual, expected):
    if actual["dtype"] != "f32" or actual["shape"] != expected["shape"]:
        raise RuntimeError("wrong result type/shape")
    if len(actual["values"]) != len(expected["values"]):
        raise RuntimeError("wrong result payload count")
    maximum = 0.0
    for a, b in zip(actual["values"], expected["values"]):
        if not math.isfinite(a) or abs(a-b) > ATOL + RTOL * abs(b):
            raise RuntimeError(f"reference result differs from scalar F32 oracle: {a} vs {b}")
        maximum = max(maximum, abs(a-b))
    return maximum


def evaluate(reference, module, entry, inputs, work, label, success=True):
    input_path, output_path = work / (label + "-inputs.json"), work / (label + "-outputs.json")
    input_path.write_text(json.dumps({"inputs": inputs}, allow_nan=False))
    invoke(reference, [module, entry, input_path, output_path], work, label, success)
    if not success:
        if output_path.exists():
            raise RuntimeError("failed reference evaluation published output")
        return None
    report = json.loads(output_path.read_text())
    if report["schema"] != "coral-stablehlo-reference-v1" or report["entry"] != entry:
        raise RuntimeError("invalid result schema")
    return report


def debug_module(text, m, d, f):
    # Test-only exposure of intermediates in an ordinary StableHLO function.
    # This does not alter the production four-input/one-output FFN ABI.
    dots = re.findall(r"^\s*(%\w+) = stablehlo.dot_general", text, re.M)
    silu = re.findall(r"^\s*(%\w+) = stablehlo.divide", text, re.M)
    hidden = re.findall(r"^\s*(%\w+) = stablehlo.multiply", text, re.M)
    if len(dots) != 3 or len(silu) != 1 or len(hidden) != 1:
        raise RuntimeError("unexpected FFN structure")
    values = [dots[0], dots[1], silu[0], hidden[0], dots[2]]
    types = [f"tensor<{m}x{f}xf32>"] * 4 + [f"tensor<{m}x{d}xf32>"]
    text = text.replace("@ffn(", "@ffn_debug(", 1)
    text, count = re.subn(r"-> tensor<[^>]+> attributes \{[^}]+\} \{",
                          "-> (" + ", ".join(types) + ") {", text, count=1)
    if count != 1:
        raise RuntimeError("missing production FFN signature")
    text, count = re.subn(r"return %\w+ : tensor<[^>]+>",
                          "return " + ", ".join(values) + " : " + ", ".join(types), text)
    if count != 1:
        raise RuntimeError("missing production FFN return")
    return text


def run(tool, reference, work):
    report = {"status": "passed", "atol": ATOL, "rtol": RTOL, "cases": []}
    matmul = work / "matmul.mlir"
    invoke(tool, ["emit-matmul", 2, 3, 4, matmul], work, "emit-matmul")
    inputs = [tensor([2, 3], [1, -2, 3, 0.5, 4, -1]),
              tensor([4, 3], [2, 0, -1, -3, 1, 2, 0, 4, 1, 1, 2, 3])]
    result = evaluate(reference, matmul, "matmul", inputs, work, "matmul")
    maximum = compare(result["outputs"][0], dot(*inputs))
    report["cases"].append({"name": "non-square-matmul", "max_abs_error": maximum,
                            "evaluation_ms": result["evaluation_ms"]})
    rng = random.Random(317)
    for m, d, f in ((1, 3, 4), (2, 3, 4), (1, 64, 172), (5, 64, 172)):
        label = f"ffn-{m}-{d}-{f}"
        module = work / (label + ".mlir")
        invoke(tool, ["emit-ffn", m, d, f, module], work, label + "-emit")
        inputs = []
        for shape in ([m, d], [f, d], [f, d], [d, f]):
            inputs.append(tensor(shape, [f32(rng.uniform(-0.5, 0.5))
                                        for _ in range(math.prod(shape))]))
        expected = oracle_ffn(inputs)
        result = evaluate(reference, module, "ffn", inputs, work, label)
        final_error = compare(result["outputs"][0], expected[-1])
        diagnostic = work / (label + "-debug.mlir")
        diagnostic.write_text(debug_module(module.read_text(), m, d, f))
        observed = evaluate(reference, diagnostic, "ffn_debug", inputs, work, label + "-debug")
        if len(observed["outputs"]) != 5:
            raise RuntimeError("missing named intermediate outputs")
        errors = {}
        for name, actual, desired in zip(("gate", "up", "silu", "hidden", "output"),
                                         observed["outputs"], expected):
            errors[name] = compare(actual, desired)
        report["cases"].append({"name": label, "max_abs_error": final_error,
                                "intermediate_max_abs_error": errors,
                                "evaluation_ms": result["evaluation_ms"]})

    valid = [tensor([2, 3], [1] * 6), tensor([4, 3], [2] * 12)]
    failures = [
        ("missing-input", "matmul", valid[:1]),
        ("shape-mismatch", "matmul", [tensor([3, 2], [1] * 6), valid[1]]),
        ("payload-size", "matmul", [tensor([2, 3], [1]), valid[1]]),
        ("zero-shape", "matmul", [tensor([0, 3], []), valid[1]]),
        ("wrong-entry", "missing", valid),
        ("overflow-input", "matmul", [tensor([2, 3], [1e100] * 6), valid[1]]),
        ("wrong-dtype", "matmul", [dict(valid[0], dtype="i32"), valid[1]]),
    ]
    for label, entry, inputs in failures:
        evaluate(reference, matmul, entry, inputs, work, label, False)
    unsupported = work / "unsupported.mlir"
    unsupported.write_text("""module {
      func.func @unsupported(%x: tensor<2x3xf32>) -> tensor<2x3xf32> {
        %y = stablehlo.sine %x : tensor<2x3xf32>
        return %y : tensor<2x3xf32>
      }
    }""")
    # This is valid standard StableHLO, but outside this phase's admission set.
    invoke(tool, ["verify", unsupported], work, "unsupported-verifies")
    evaluate(reference, unsupported, "unsupported", valid[:1], work, "unsupported", False)
    mixed = work / "mixed-dot.mlir"
    mixed.write_text(DOT.replace("4x3xf32", "4x3xi32"))
    invoke(tool, ["verify", mixed], work, "mixed-dot-verifies")
    evaluate(reference, mixed, "matmul", valid, work, "mixed-dot-rejected", False)
    nonfinite = work / "nonfinite.mlir"
    nonfinite.write_text("""module {
      func.func @overflow(%x: tensor<1xf32>) -> tensor<1xf32> {
        %y = stablehlo.multiply %x, %x : tensor<1xf32>
        return %y : tensor<1xf32>
      }
    }""")
    evaluate(reference, nonfinite, "overflow", [tensor([1], [3e38])], work, "nonfinite-output", False)
    report["negative_cases"] = len(failures) + 3
    (work / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "passed", "evidence": str(work), "negative_cases": report["negative_cases"],
                      "max_abs_error": max(case["max_abs_error"] for case in report["cases"])}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("tool", "reference", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="reference-", dir=args.output_root.resolve()))
    run(args.tool.resolve(), args.reference.resolve(), work)


if __name__ == "__main__":
    main()
