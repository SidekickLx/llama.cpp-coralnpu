#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Accept real-model FFN JIT with cold, persistent-hit and failure evidence."""
import argparse
import copy
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from verify_runner import read_run, require
from verify_ffn_regions import inspect_regions
from verify_ffn_members import compare, copy_catalog
from verify_unpatched import require_clean

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "compiler"))
from region_cache import file_digest, identity_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matmuls", type=Path, required=True)
    parser.add_argument("--pointwise", type=Path, required=True)
    parser.add_argument("--aot", type=Path, required=True)
    parser.add_argument("--m-buckets", type=int, nargs="+", default=[])
    parser.add_argument("--weight-layout", choices=("native", "transposed"), default="native")
    parser.add_argument("--model", type=Path, default=ROOT / "models/stories260K-f32.gguf")
    args = parser.parse_args()
    from region_service import select_row_capacity
    capacities = {m: select_row_capacity(m, args.m_buckets) for m in (1, 5)}
    profile_count = len(set(capacities.values()))
    revision = require_clean()
    work = Path(tempfile.mkdtemp(prefix="m5-jit-", dir=ROOT / "build"))
    print(f"Evidence: {work}", flush=True)
    runner = ROOT / "build/m1/coral-llama-runner"
    catalog = work / "members"
    catalog.mkdir()
    copy_catalog(args.matmuls, catalog, "profiles.tsv")
    copy_catalog(args.pointwise, catalog, "pointwise.tsv")
    config_path = work / "jit.json"
    checks = {}
    def command(label, argv, success=True, trace=False):
        argv = list(map(str, argv))
        if trace:
            argv = ["/usr/bin/strace", "-f", "-s", "4096", "-e", "trace=execve,execveat",
                    "-o", str(work / (label + ".exec.trace")), *argv]
        start = time.perf_counter()
        env = dict(os.environ)
        if label != "configure":
            env["PATH"] = "/nonexistent"
        result = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
        (work / (label + ".stdout.log")).write_text(result.stdout)
        (work / (label + ".stderr.log")).write_text(result.stderr)
        require((result.returncode == 0) == success, label + ": " + result.stderr[-4000:])
        checks[label] = {"exit": result.returncode, "wall_ms": (time.perf_counter() - start) * 1000}
        print(label + ": passed", flush=True)
        return result

    command("configure", [sys.executable, ROOT / "compiler/jit_worker.py", "configure",
                         "--cache", work / "cache", "--output", config_path,
                         "--weight-layout", args.weight_layout,
                         *(["--m-buckets", *args.m_buckets] if args.m_buckets else [])])
    command("cpu", [runner, "--mode", "cpu", "--model", args.model, "--output", work / "cpu"])
    _, reference = read_run(work / "cpu", "cpu")
    tokens = work / "cpu/tokens.txt"
    base = [runner, "--mode", "coral", "--model", args.model,
            "--artifacts", catalog, "--observation", "backend"]
    def inference(label, config=config_path, replay=tokens, success=True, trace=False):
        result = command(label, [*base, "--jit-config", config, "--tokens", replay,
                                 "--output", work / label], success, trace)
        if not success:
            require(not (work / label / "summary.json").exists(), "failure published completion")
            return result
        summary, values = read_run(work / label, "coral")
        inspect_regions(work / label, 20, {m: (6 if args.weight_layout == "native" and b == 1 else 3)
                                              for m, b in capacities.items()})
        require(summary["stats"]["matmuls"] == 84 and summary["stats"]["calls_completed"] == 104,
                "missing mixed matmul/FFN execution")
        require(summary["stats"]["ffn_module_cache_hits"] == 20 - profile_count and
                summary["stats"]["ffn_weight_preparations"] == 5 and
                summary["stats"]["ffn_weight_cache_hits"] == 15, "module/weight reuse failed")
        return summary["stats"], values

    cold, actual = inference("cold", trace=True)
    require(cold["jit_compilations"] == cold["jit_worker_launches"] == cold["jit_requests"] == profile_count,
            "each compiled capacity must compile once")
    require(cold["jit_disk_hits"] == 0, "cold run unexpectedly hit")
    error = compare(actual, reference)
    warm, reloaded = inference("warm", trace=True)
    require(warm["jit_compilations"] == warm["jit_worker_launches"] == 0 and
            warm["jit_disk_hits"] == warm["jit_requests"] == profile_count, "fresh-process cache hit failed")
    require(actual == reloaded, "persistent reload changed logits")
    for label, workers, compiles in (("cold", profile_count, profile_count), ("warm", 0, 0)):
        trace = (work / (label + ".exec.trace")).read_text()
        launches = [line for line in trace.splitlines() if "execve(" in line]
        require("execveat(" not in trace, "unexpected descriptor execution")
        require(sum('jit_worker.py", "compile"' in line for line in launches) == workers,
                "worker launch count differs from counters")
        require(sum(re.search(r'execve\("[^"]*/iree-compile"', line) is not None
                    for line in launches) == compiles, "actual compiler invocation count differs")
        if label == "warm":
            require(len(launches) == 1, "persistent hit launched a child process")

    # Keep shapes constant while changing the actual activation/token sequence.
    changed = tokens.read_text().splitlines()
    changed[3] = "404" if changed[3] != "404" else "405"
    new_tokens = work / "changed-tokens.txt"
    new_tokens.write_text("\n".join(changed) + "\n")
    command("changed-cpu", [runner, "--mode", "cpu", "--model", args.model,
                           "--tokens", new_tokens, "--output", work / "changed-cpu"])
    _, changed_reference = read_run(work / "changed-cpu", "cpu")
    changed_stats, changed_actual = inference("changed-activation", replay=new_tokens)
    changed_error = compare(changed_actual, changed_reference)
    require(changed_actual != actual and changed_stats["jit_compilations"] == 0 and
            changed_stats["jit_worker_launches"] == 0, "activation change triggered compilation/stale output")

    entries = list((work / "cache/entries").iterdir())
    require(len(entries) == profile_count, "unexpected canonical entries")
    for entry in entries:
        identity = json.loads((entry / "cache.json").read_text())
        shape = identity["profile"]
        original = args.aot / ("real-decode" if shape["m"] == 1 else "real-prefill")
        old = json.loads((original / "manifest.json").read_text())
        from region_service import Toolchain, optimize_region, specialize_rows
        from ffn_artifact import digest
        original_shape = old["contract"]["shape"]
        source_profile = tuple(original_shape[name] for name in ("m", "d", "f"))
        transformed = optimize_region((original / "region.stablehlo.mlir").read_bytes(),
                                      source_profile, args.weight_layout,
                                      toolchain=Toolchain(), evidence=work / ("aot-p1-" + str(shape["m"])))
        if args.m_buckets:
            transformed, _ = specialize_rows(transformed, source_profile, shape["m"], args.weight_layout,
                                             toolchain=Toolchain(), evidence=work / ("aot-p2-" + str(shape["m"])))
        require(identity["semantic_sha256"] == digest(transformed),
                "JIT import differs from accepted AOT semantics")
    victim = entries[0]
    (victim / "region.vmfb").write_bytes(b"deliberately corrupt cached program")
    repaired, values = inference("repair")
    require(repaired["jit_worker_launches"] == repaired["jit_compilations"] ==
            repaired["jit_recoveries"] == 1 and values == actual, "cache repair failed")
    index = next((work / "cache/requests").glob("*.json"))
    index.write_text('{"incomplete":true}')
    repaired_index, values = inference("index-repair")
    require(repaired_index["jit_worker_launches"] == 1 and
            repaired_index["jit_compilations"] == 0 and values == actual,
            "request-index repair should reuse canonical artifact")

    # Version mismatch is detected before a worker or stale package is used.
    config = json.loads(config_path.read_text())
    stale = copy.deepcopy(config)
    stale["environment"]["files"]["tool"] = "0" * 64
    stale["environment_sha256"] = identity_hash(stale["environment"])
    stale_path = work / "stale.json"
    stale_path.write_text(json.dumps(stale))
    failure = inference("stale-environment", config=stale_path, success=False)
    require("stale JIT configuration" in failure.stderr, "wrong stale configuration failure")

    # Failure injection uses a local worker copy; production tools stay intact.
    for label, body, timeout in (
        ("worker-failure", 'raise RuntimeError("intentional worker failure")\n', 600),
        ("worker-timeout", 'import subprocess, sys, time\n'
                           'p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])\n'
                           'print("child_pid=" + str(p.pid), flush=True)\n'
                           'time.sleep(60)\n', 1)):
        fake = work / (label + ".py")
        fake.write_text(body)
        broken = copy.deepcopy(config)
        broken["paths"]["jit_worker.py"] = str(fake)
        broken["environment"]["files"]["jit_worker.py"] = file_digest(fake)
        broken["environment_sha256"] = identity_hash(broken["environment"])
        broken["timeout_seconds"] = timeout
        path = work / (label + ".json")
        path.write_text(json.dumps(broken))
        failure = inference(label, config=path, success=False)
        require("compiler worker" in failure.stderr, "wrong worker failure")
        if label == "worker-timeout":
            require("timeout" in failure.stderr, "worker timeout missing")
            for log in (work / "cache/staging").glob("worker-*/worker.log"):
                text = log.read_text()
                if "child_pid=" in text:
                    pid = int(text.split("child_pid=")[1].splitlines()[0])
                    state = Path(f"/proc/{pid}/stat")
                    # A killed child can remain a zombie until WSL init reaps it.
                    require(not state.exists() or state.read_text().split()[2] == "Z",
                            "compiler descendant survived timeout")

    # The real scheduler exercises alternate FFN spellings, mutable weights,
    # multiple splits, graph reuse and arena destruction with JIT enabled.
    boundary_runner = ROOT / "build/m1/coral-ffn-region-test"
    for mode in ("mutable", "immutable", "external-consumer", "external-view",
                 "callbacks", "callback-subdivision", "stale-metadata", "disabled"):
        selected = config_path if mode in ("mutable", "immutable") else work / "must-not-open.json"
        result = command("boundary-" + mode, [boundary_runner, mode, catalog, selected])
        row = json.loads(result.stdout.splitlines()[-1])
        require(row["status"] == "passed" and row["jit_compilations"] == 0,
                "boundary test unexpectedly compiled")
        if mode not in ("mutable", "immutable"):
            require(row["jit_requests"] == 0, "ineligible region reached JIT")
    command("boundary-worker-failure", [boundary_runner, "jit-failure", catalog,
                                       work / "worker-failure.json"])
    require(len(list((work / "cache/entries").iterdir())) == profile_count,
            "equivalent GGML spellings created redundant canonical artifacts")

    require_clean()
    report = {"status": "passed", "llama_revision": revision,
              "cold": cold, "warm": warm, "changed_activation": changed_stats,
              "repaired": repaired, "index_repair": repaired_index,
              "max_abs_vs_cpu": error, "changed_max_abs_vs_cpu": changed_error,
              "checks": checks, "canonical_profiles": profile_count, "m_buckets": args.m_buckets,
              "weight_layout": args.weight_layout,
              "cold_worker_execs": profile_count, "cold_compiler_execs": profile_count,
              "warm_child_execs": 0, "runner_sha256": file_digest(runner)}
    (work / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "passed", "report": str(work / "report.json"),
                      "max_abs_vs_cpu": error, "cold": cold, "warm": warm}), flush=True)


if __name__ == "__main__":
    main()
