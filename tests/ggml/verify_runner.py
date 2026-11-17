#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Exercise the M2 model runner in fresh processes, without building or compiling."""

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile

# Frozen model-logit tolerance for the strict F32 path.
ATOL = 1e-4
RTOL = 1e-4


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def invoke(runner, args, label, work, expect_success=True):
    # Neither inference path may depend on a compiler or shell in PATH.
    env = os.environ.copy()
    env["PATH"] = "/nonexistent"
    try:
        result = subprocess.run(
            [str(runner), *map(str, args)],
            capture_output=True, text=True, env=env, timeout=300, check=False,
        )
    except subprocess.TimeoutExpired as error:
        for name, value in (("stdout", error.stdout), ("stderr", error.stderr)):
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")
            (work / f"{label}.{name}.log").write_text(value or "", encoding="utf-8")
        raise RuntimeError(f"{label}: timed out") from error
    for name, value in (("stdout", result.stdout), ("stderr", result.stderr)):
        (work / f"{label}.{name}.log").write_text(value, encoding="utf-8")
    require((result.returncode == 0) == expect_success,
            f"{label}: unexpected exit {result.returncode}\n{result.stderr[-3000:]}")
    return result


def read_run(directory, mode):
    summary = json.loads((directory / "summary.json").read_text())
    steps = [json.loads(line) for line in (directory / "steps.jsonl").read_text().splitlines()]
    trace = [json.loads(line) for line in (directory / "trace.jsonl").read_text().splitlines()]
    require(summary["status"] == "completed" and summary["mode"] == mode, "wrong completion marker")
    require(summary["registered_devices"] == (["CPU", "CORAL"] if mode == "coral" else ["CPU"]),
            "unexpected inference device")
    require(summary["coral_registered"] == (mode == "coral"), "wrong backend registration")
    require(len(steps) == summary["decode_steps"] + 1 == summary["logit_rows"], "missing logits/steps")
    payload = (directory / "logits.f32").read_bytes()
    require(len(payload) == 4 * summary["vocab_size"] * len(steps), "wrong logit file size")
    values = struct.unpack(f"<{len(payload) // 4}f", payload)
    require(all(math.isfinite(value) for value in values), "non-finite logits")
    require(summary["observed_profiles"] > 0, "no real matmul profiles")
    buffers = {item["id"] for item in trace if item["kind"] == "buffer"}
    for index, step in enumerate(steps):
        require(step["step"] == index and step["logit_row"] == index, "wrong step numbering")
        require(step["phase"] == ("prefill" if index == 0 else "decode"), "wrong phase")
        nodes = [item for item in trace if item["kind"] == "node" and item["step"] == index]
        coral = [node for node in nodes if node["placement"] == "CORAL"]
        cpu = [node for node in nodes if node["placement"] == "CPU"]
        require(len(nodes) == len(cpu) + len(coral), "unknown placement")
        require(len(coral) == step["coral_nodes"], "Coral node trace mismatch")
        if summary.get("observation", "nodes") == "nodes":
            require(len(cpu) == step["cpu_nodes"] and len(cpu) > 0,
                    "missing CPU observations")
        else:
            require(not cpu and step["cpu_nodes"] is None and step["cpu_matmuls"] is None,
                    "backend observation must not invent CPU counts")
        for node in nodes:
            for field in ("src0", "src1", "output"):
                tensor = node[field]
                if tensor is not None:
                    require(tensor["buffer_id"] == 0 or tensor["buffer_id"] in buffers, "unknown buffer")
        stats = step["stats"]
        for field in ("failures", "failed_dispatches", "observer_errors", "host_executable_load_attempts"):
            require(stats[field] == 0, f"unexpected {field}")
        require(sum(node["dispatches"] for node in coral) == stats["coral_dispatches"], "dispatch trace mismatch")
        require(sum(node["instructions"] for node in coral) == stats["kernel_instructions"], "instruction trace mismatch")
        if mode == "coral":
            require(len(coral) > 0 and stats["kernel_instructions"] > 0, "no actual NPU execution")
            require(stats["instruction_samples"] == stats["coral_dispatches"], "missing instruction samples")
            matmuls = sum(node["op"] == "MUL_MAT" for node in coral)
            require(stats["matmuls"] == matmuls, "matmul trace mismatch")
            require(stats.get("pointwise_calls", 0) == len(coral) - matmuls, "member trace mismatch")
            require(len(coral) == stats["calls_completed"], "incomplete Coral calls")
            if index >= 2:
                require(stats["module_loads"] == 0, "warm decode reloaded a module")
                require(stats["weight_cache_hits"] > 0, "warm decode did not reuse model weights")
        else:
            require(not coral and stats["module_loads"] == 0, "CPU baseline used Coral")
    return summary, values


def run_suite(runner, model, artifacts, work):
    cpu = work / "cpu"
    invoke(runner, ["--mode", "cpu", "--model", model, "--output", cpu], "cpu", work)
    baseline, reference = read_run(cpu, "cpu")
    replay = work / "cpu-replay"
    invoke(runner, ["--mode", "cpu", "--model", model, "--output", replay,
                    "--tokens", cpu / "tokens.txt"], "cpu-replay", work)
    _, repeated = read_run(replay, "cpu")
    require(reference == repeated, "fixed-token CPU replay changed logits")
    require((cpu / "tokens.txt").read_bytes() == (replay / "tokens.txt").read_bytes(), "replay changed inputs")

    marker = (cpu / "summary.json").read_bytes()
    result = invoke(runner, ["--mode", "cpu", "--model", model, "--output", cpu],
                    "existing-output", work, False)
    require("new directory" in result.stderr and (cpu / "summary.json").read_bytes() == marker,
            "existing results were not preserved")

    malformed = work / "malformed-tokens.txt"
    malformed.write_text("coral-m2-tokens-v1\n2 3\n1 2\n", encoding="utf-8")
    invoke(runner, ["--mode", "cpu", "--model", model, "--output", work / "bad-input",
                    "--tokens", malformed], "bad-input", work, False)
    require(not (work / "bad-input" / "summary.json").exists(), "bad input published success")

    invalid_vocab = work / "out-of-vocabulary.txt"
    invalid_vocab.write_text("coral-m2-tokens-v1\n1 2\n2147483647\n0\n0\n", encoding="utf-8")
    result = invoke(runner, ["--mode", "cpu", "--model", model, "--output", work / "bad-vocab",
                             "--tokens", invalid_vocab], "bad-vocab", work, False)
    require("outside this model" in result.stderr, "out-of-vocabulary tokens were not rejected")

    missing = work / "missing-artifacts"
    missing.mkdir()
    (missing / "profiles.tsv").write_text("coral-m2-v1\n1\t16\t8\tmissing.vmfb\n", encoding="utf-8")
    invoke(runner, ["--mode", "coral", "--model", model, "--output", work / "bad-artifacts",
                    "--artifacts", missing], "bad-artifacts", work, False)
    require(not (work / "bad-artifacts" / "summary.json").exists(), "missing artifacts published success")

    report = {"cpu_replay_exact": True, "negative_cases": 4, "coral_checked": False}
    if artifacts:
        coral = work / "coral"
        invoke(runner, ["--mode", "coral", "--model", model, "--output", coral,
                        "--tokens", cpu / "tokens.txt", "--artifacts", artifacts], "coral", work)
        candidate, actual = read_run(coral, "coral")
        checks = candidate["stats"]
        require(checks["graph_optimizations"] > 0 and checks["ffn_candidates"] > 0,
                "upstream FFN planning hook was not exercised")
        require(checks["ffn_preparation_plans"] == 0 and
                checks["ffn_rejections"] == checks["ffn_candidates"],
                "diagnostic callback or unsupported FFN was incorrectly admitted")
        require(candidate["vocab_size"] == baseline["vocab_size"] and len(actual) == len(reference),
                "incompatible logit shapes")
        require((coral / "tokens.txt").read_bytes() == (cpu / "tokens.txt").read_bytes(), "Coral inputs differ")
        errors = [abs(value - expected) for value, expected in zip(actual, reference)]
        mismatches = sum(error > ATOL + RTOL * abs(expected) for error, expected in zip(errors, reference))
        require(mismatches == 0, f"CPU/Coral logits mismatch: {mismatches}, max error {max(errors)}")

        # Valid files with no matching model shape must not pass as a Coral run.
        unmatched = work / "unmatched-artifacts"
        unmatched.mkdir()
        first = next(artifacts.glob("*.vmfb"))
        shutil.copyfile(first, unmatched / "unused.vmfb")
        (unmatched / "profiles.tsv").write_text("coral-m2-v1\n1\t1\t1\tunused.vmfb\n", encoding="utf-8")
        result = invoke(runner, ["--mode", "coral", "--model", model, "--output", work / "no-offload",
                                 "--tokens", cpu / "tokens.txt", "--artifacts", unmatched],
                        "no-offload", work, False)
        require("requires actual NPU dispatch" in result.stderr, "CPU-only execution was accepted as Coral")
        require(not (work / "no-offload" / "summary.json").exists(), "no-offload published success")
        # A matching manifest with broken VMFB bytes must propagate a runtime
        # error, not silently fall back or publish a successful result.
        broken = work / "corrupt-artifacts"
        broken.mkdir()
        manifest = (artifacts / "profiles.tsv").read_bytes()
        (broken / "profiles.tsv").write_bytes(manifest)
        for row in manifest.decode().splitlines()[1:]:
            if row.strip() and not row.startswith("#"):
                (broken / row.split()[3]).write_bytes(b"not a VMFB")
        result = invoke(runner, ["--mode", "coral", "--model", model, "--output", work / "corrupt-run",
                                 "--tokens", cpu / "tokens.txt", "--artifacts", broken],
                        "corrupt-run", work, False)
        require("llama_decode failed" in result.stderr, "runtime failure did not propagate through llama.cpp")
        require(not (work / "corrupt-run" / "summary.json").exists(), "runtime failure published success")
        report.update(coral_checked=True, negative_cases=6, max_abs_error=max(errors),
                      atol=ATOL, rtol=RTOL, logit_values=len(actual), stats=candidate["stats"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("build/m2-results"))
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="runner-", dir=args.output_root)).resolve()
    print(f"Results: {work}", flush=True)
    report = run_suite(args.runner.resolve(), args.model.resolve(),
                       args.artifacts.resolve() if args.artifacts else None, work)
    (work / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
