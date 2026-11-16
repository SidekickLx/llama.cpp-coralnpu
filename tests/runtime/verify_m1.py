#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Validate an M1 runner and precompiled artifacts without rebuilding them."""

import argparse
import json
import math
from pathlib import Path
import random
import struct
import subprocess
import tempfile


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_f32(path, values):
    path.write_bytes(struct.pack(f"<{len(values)}f", *values))


def invoke(runner, arguments, label, work, expected_exit=0, error_text=None):
    try:
        result = subprocess.run(
            [str(runner), *map(str, arguments)],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        # TimeoutExpired may contain bytes even when text=True. Preserve all
        # partial evidence before reporting the timeout to the caller.
        for stream, output in (("stdout", error.stdout), ("stderr", error.stderr)):
            if isinstance(output, bytes):
                output = output.decode("utf-8", errors="replace")
            (work / f"{label}.{stream}.log").write_text(output or "", encoding="utf-8")
        raise RuntimeError(f"{label}: runner timed out after 180 seconds") from error
    (work / f"{label}.stdout.log").write_text(result.stdout, encoding="utf-8")
    (work / f"{label}.stderr.log").write_text(result.stderr, encoding="utf-8")
    require(
        result.returncode == expected_exit,
        f"{label}: exit {result.returncode}, expected {expected_exit}\n{result.stderr[-4000:]}",
    )
    if error_text is not None:
        require(error_text in result.stderr, f"{label}: expected error {error_text!r}")
    records = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    print(json.dumps({"invocation": label, "exit_code": result.returncode}), flush=True)
    return records


def check_stats(records, cycles, calls, m, prepared_weights):
    stats = [record for record in records if "calls_completed" in record]
    require(len(stats) == cycles, "wrong number of context statistics records")
    for record in stats:
        require(record["module_loads"] == 1, "module was not loaded exactly once")
        require(record["calls_attempted"] == calls, "unexpected call attempts")
        require(record["calls_completed"] == calls, "some calls did not finish")
        require(record["coral_dispatches"] >= calls, "missing Coral dispatches")
        require(
            record["instruction_samples"] == record["coral_dispatches"],
            "missing instruction samples",
        )
        require(record["kernel_instructions"] > 0, "no simulated kernel instructions")
        for field in ("failed_dispatches", "observer_errors", "host_executable_load_attempts"):
            require(record[field] == 0, f"unexpected {field}")
        require(record["binding_allocations"] == prepared_weights + 1, "buffers not reused")
        require(
            record["upload_bytes"] == (prepared_weights * 8 * 16 + calls * m * 16) * 4,
            "unexpected logical upload volume",
        )
        require(record["download_bytes"] == calls * m * 8 * 4, "unexpected readback volume")
    cases = [record for record in records if "case" in record]
    require(len(cases) == cycles * calls, "missing numerical checks")
    for record in records:
        if "case" in record:
            require(record["passed"] and record["mismatches"] == 0, "CPU reference mismatch")


def run_suite(runner, artifacts, work):
    # These are the six independently compiled target/shape combinations.
    for target in ("zvt", "rvv"):
        for m in (1, 16, 32):
            module = artifacts / f"m{m}_{target}.vmfb"
            base = ["--module", module, "--m", m, "--k", 16, "--n", 8]
            label = f"m{m}_{target}"
            records = invoke(
                runner, [*base, "--repeat", 3, "--recreate", 2, "--self-test"],
                f"{label}_self_test", work,
            )
            check_stats(records, cycles=2, calls=8, m=m, prepared_weights=2)

            # Exercise the public file ABI with non-dyadic, signed F32 values.
            # Round input data to F32 before computing the independent reference.
            rng = random.Random(1000 + m)
            x = [rng.uniform(-1, 1) for _ in range(m * 16)]
            w = [rng.uniform(-1, 1) for _ in range(8 * 16)]
            x_path = work / f"{label}_x.f32"
            w_path = work / f"{label}_w.f32"
            y_path = work / f"{label}_y.f32"
            write_f32(x_path, x)
            write_f32(w_path, w)
            x = struct.unpack(f"<{len(x)}f", x_path.read_bytes())
            w = struct.unpack(f"<{len(w)}f", w_path.read_bytes())
            records = invoke(
                runner, [*base, "--input", x_path, "--weights", w_path,
                         "--output", y_path, "--repeat", 3],
                f"{label}_file_io", work,
            )
            check_stats(records, cycles=1, calls=3, m=m, prepared_weights=1)
            output = y_path.read_bytes()
            require(len(output) == m * 8 * 4, "wrong raw output size")
            actual = struct.unpack(f"<{m * 8}f", output)
            for row in range(m):
                for col in range(8):
                    expected = math.fsum(x[row * 16 + k] * w[col * 16 + k] for k in range(16))
                    expected = struct.unpack("<f", struct.pack("<f", expected))[0]
                    value = actual[row * 8 + col]
                    require(
                        math.isfinite(value) and abs(value - expected) <= 1e-6 + 1e-5 * abs(expected),
                        f"{label}: independent raw-output comparison failed",
                    )

    module = artifacts / "m1_zvt.vmfb"
    base = ["--module", module, "--repeat", 1]
    corrupt = work / "corrupt.vmfb"
    corrupt.write_bytes(b"not an IREE module")
    short_input = work / "short.f32"
    write_f32(short_input, [0.0])
    nan_input = work / "nan.f32"
    write_f32(nan_input, [float("nan")] + [0.0] * 15)
    weights = work / "negative_w.f32"
    write_f32(weights, [1.0] * 128)
    existing_output = work / "existing.f32"
    existing_output.write_bytes(b"keep this file")

    negative_cases = [
        ("wrong_shape", [*base, "--m", 2], 1, "ABI mismatch"),
        ("missing_export", [*base, "--function", "missing"], 1, "find matmul export"),
        ("corrupt_module", ["--module", corrupt], 1, "load VMFB"),
        ("missing_module", ["--module", work / "missing.vmfb"], 1, "cannot open VMFB"),
        ("zero_dimension", [*base, "--m", 0], 2, "positive integer"),
        ("unpaired_inputs", [*base, "--input", short_input], 2, "supplied together"),
        ("short_input", [*base, "--input", short_input, "--weights", weights], 1, "wrong size"),
        ("nan_input", [*base, "--input", nan_input, "--weights", weights], 2, "finite F32"),
        ("existing_output", [*base, "--output", existing_output], 1, "output exists"),
        ("missing_abi", ["--module", artifacts / "m1_noabi.vmfb"], 1, "ABI mismatch"),
        ("host_kernel", ["--module", artifacts / "m1_host.vmfb"], 1, "rejects host numerical"),
    ]
    for label, arguments, exit_code, error in negative_cases:
        records = invoke(runner, arguments, label, work, exit_code, error)
        if label == "host_kernel":
            stats = [record for record in records if "calls_completed" in record]
            require(bool(stats), "missing host rejection statistics")
            require(
                sum(record["host_executable_load_attempts"] for record in stats) > 0,
                "host executable loader was not exercised",
            )
            require(
                all(record["calls_completed"] == 0 for record in stats),
                "host kernel rejection published a successful result",
            )
    require(existing_output.read_bytes() == b"keep this file", "existing output was overwritten")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    runner = args.runner.resolve()
    artifacts = args.artifacts.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    # Keep every run's evidence, including failures; never delete build artifacts.
    work = Path(tempfile.mkdtemp(prefix="verification-", dir=args.work_dir.resolve()))
    print(f"Verification evidence: {work}", flush=True)
    run_suite(runner, artifacts, work)
    (work / "result.json").write_text(
        json.dumps({"passed": True, "profiles": 6, "runner_processes": 23}, indent=2) + "\n",
        encoding="utf-8",
    )
    print("All M1 runtime checks passed.", flush=True)


if __name__ == "__main__":
    main()
