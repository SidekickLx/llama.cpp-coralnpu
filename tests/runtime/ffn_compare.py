# SPDX-License-Identifier: MIT
"""Strict tensor payload comparison shared by standalone FFN acceptance tools."""
import math

def compare(actual, expected, atol=1e-4, rtol=1e-4):
    for tensor in (actual, expected):
        if tensor.get("dtype") != "f32" or len(tensor.get("shape", [])) != 2:
            raise RuntimeError("expected rank-two F32 result")
        if any(type(dim) is not int or dim <= 0 for dim in tensor["shape"]):
            raise RuntimeError("invalid result dimensions")
        if len(tensor.get("values", [])) != math.prod(tensor["shape"]):
            raise RuntimeError("incomplete result payload")
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in tensor["values"]):
            raise RuntimeError("non-finite or invalid result payload")
    if actual["shape"] != expected["shape"]:
        raise RuntimeError("result shapes differ")
    max_abs = max_rel = max_ratio = 0.0
    squared = 0.0
    failures = []
    failed_count = 0
    for i, (a, b) in enumerate(zip(actual["values"], expected["values"])):
        error = abs(a-b)
        tolerance = atol + rtol * abs(b)
        max_abs = max(max_abs, error)
        if b:
            max_rel = max(max_rel, error / abs(b))
        max_ratio = max(max_ratio, error / tolerance)
        squared += error * error
        if error > tolerance:
            failed_count += 1
            if len(failures) < 32:
                failures.append(i)
    return {"status": "passed" if failed_count == 0 else "failed",
            "elements": len(actual["values"]), "max_abs_error": max_abs,
            "max_relative_error_nonzero_reference": max_rel,
            "max_tolerance_ratio": max_ratio, "rmse": math.sqrt(squared / len(actual["values"])),
            "failed_count": failed_count, "first_failing_indices": failures,
            "atol": atol, "rtol": rtol}
