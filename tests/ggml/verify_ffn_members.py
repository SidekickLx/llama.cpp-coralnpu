#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Prove standalone member placement on unmodified llama.cpp, without FFN fusion."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from verify_runner import ATOL, RTOL, invoke, read_run, require
from verify_unpatched import ROOT, require_clean

def copy_catalog(source, destination, name):
    lines = (source / name).read_text().splitlines()
    (destination / name).write_text("\n".join(lines) + "\n")
    for line in lines[1:]:
        if not line.strip() or line.startswith("#"): continue
        filename = line.split()[-1]
        require(Path(filename).name == filename, "unsafe catalog filename")
        shutil.copyfile(source / filename, destination / filename)

def compare(actual, reference):
    require(len(actual) == len(reference), "logit shape mismatch")
    errors = [abs(a - b) for a, b in zip(actual, reference)]
    require(all(e <= ATOL + RTOL * abs(b) for e, b in zip(errors, reference)),
            f"logit mismatch: {max(errors)}")
    return max(errors)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--matmuls", type=Path, required=True)
    parser.add_argument("--pointwise", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("build/m3-member-results"))
    args = parser.parse_args()
    revision = require_clean()
    runner, model = args.runner.resolve(strict=True), args.model.resolve(strict=True)
    cache = (runner.parent / "CMakeCache.txt").read_text()
    require(f"LLAMA_SOURCE_DIR:PATH={(ROOT / 'repos/llama.cpp').resolve()}" in cache,
            "runner build must use the locked clean llama.cpp source")
    args.output_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="members-", dir=args.output_root)).resolve()
    print(f"Evidence: {work}", flush=True)
    artifacts = work / "artifacts"
    artifacts.mkdir()
    copy_catalog(args.matmuls, artifacts, "profiles.tsv")
    copy_catalog(args.pointwise, artifacts, "pointwise.tsv")
    common = ["--model", model]
    invoke(runner, ["--mode", "cpu", *common, "--output", work / "cpu"], "cpu", work)
    baseline, reference = read_run(work / "cpu", "cpu")
    replay = ["--tokens", work / "cpu/tokens.txt"]
    runs = {}
    for observation in ("nodes", "backend"):
        output = work / observation
        invoke(runner, ["--mode", "coral", *common, *replay, "--artifacts", artifacts,
                        "--observation", observation, "--output", output], observation, work)
        summary, actual = read_run(output, "coral")
        require((output / "tokens.txt").read_bytes() == (work / "cpu/tokens.txt").read_bytes(),
                "fixed-token inputs changed")
        error = compare(actual, reference)
        require(summary["stats"]["pointwise_calls"] > 0, "no FFN member execution")
        runs[observation] = {"max_abs_error": error, "stats": summary["stats"]}
    require(runs["nodes"]["stats"]["ffn_preparation_plans"] == 0, "callback policy violated")
    require(runs["backend"]["stats"]["ffn_preparation_plans"] > 0, "no complete FFN admission")
    trace = [json.loads(line) for line in (work / "backend/trace.jsonl").read_text().splitlines()]
    plans = [item for item in trace if item["kind"] == "ffn_plan"]
    # Reservation is not execution. Require each step's actual GLU/down members
    # to share a graph_compute call and match an admitted complete plan.
    executed = []
    for step in range(baseline["logit_rows"]):
        nodes = [item for item in trace if item["kind"] == "node" and item["step"] == step]
        by_name = {node["name"]: node for node in nodes}
        require(len(by_name) == len(nodes), "ambiguous node names within a model step")
        glu = [node for node in nodes if node["op"] == "GLU"]
        require(glu, "no executed FFN activation")
        for activation in glu:
            matches = []
            for plan in plans:
                names = [member["name"] for member in plan["members"]]
                if activation["name"] not in names or not all(name in by_name for name in names):
                    continue
                group = [by_name[name] for name in names]
                if plan["m"] != activation["output"]["ne"][1]: continue
                require(len({node["compute_call"] for node in group}) == 1,
                        "complete FFN was split between compute calls")
                require([node["op"] for node in group] == [member["op"] for member in plan["members"]],
                        "plan differs from executed operations")
                matches.append(plan)
            require(matches, "executed activation has no matching same-split FFN plan")
            executed.append({"step": step, "activation": activation["name"],
                             "compute_call": activation["compute_call"],
                             "m": activation["output"]["ne"][1], "members": matches[0]["members"]})
    # Missing member artifacts must preserve correct CPU placement, never grant
    # a complete FFN plan. Other valid member kinds remain in the catalog.
    partial = work / "partial-artifacts"
    partial.mkdir()
    copy_catalog(artifacts, partial, "profiles.tsv")
    copy_catalog(artifacts, partial, "pointwise.tsv")
    lines = (partial / "pointwise.tsv").read_text().splitlines()
    (partial / "pointwise.tsv").write_text("\n".join(
        line for line in lines if not line.startswith("swiglu\t")) + "\n")
    invoke(runner, ["--mode", "coral", *common, *replay, "--artifacts", partial,
                    "--observation", "backend", "--output", work / "missing-member"],
           "missing-member", work)
    missing, actual = read_run(work / "missing-member", "coral")
    require(missing["stats"]["ffn_preparation_plans"] == 0, "missing artifact granted a plan")
    require(missing["stats"]["pointwise_calls"] == 0, "missing SwiGLU was offloaded")
    compare(actual, reference)
    # Registration may accept filenames, but runtime must reject corrupt bytes.
    bad = work / "bad-member-artifacts"
    bad.mkdir()
    copy_catalog(artifacts, bad, "profiles.tsv")
    copy_catalog(artifacts, bad, "pointwise.tsv")
    for line in (bad / "pointwise.tsv").read_text().splitlines()[1:]:
        if line.startswith("swiglu\t"):
            (bad / line.split()[-1]).write_bytes(b"invalid pointwise VMFB")
    invoke(runner, ["--mode", "coral", *common, *replay, "--artifacts", bad,
                    "--observation", "backend", "--output", work / "bad-member"],
           "bad-member", work, expect_success=False)
    require(not (work / "bad-member/summary.json").exists(), "failed member published success")
    require_clean()
    report = {"status": "passed", "llama_revision": revision, "llama_worktree_clean": True,
              "atol": ATOL, "rtol": RTOL,
              "runs": runs, "executed_ffns": executed, "complete_ffn_member_groups": len(executed),
              "missing_member_fallback": True, "corrupt_member_failure": True,
              "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
              "runner_sha256": hashlib.sha256(runner.read_bytes()).hexdigest()}
    (work / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "executed_ffns"}), flush=True)
if __name__ == "__main__":
    main()
