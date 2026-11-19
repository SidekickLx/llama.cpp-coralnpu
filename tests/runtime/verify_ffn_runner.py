#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Replay synthetic and captured model FFNs through three independent engines."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from ffn_compare import compare

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/ggml"))
from verify_unpatched import require_clean

def require(value, message):
    if not value:
        raise RuntimeError(message)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")

def shape(path):
    tensors = json.loads(path.read_text())["inputs"]
    return (tensors[0]["shape"][0], tensors[0]["shape"][1], tensors[1]["shape"][0])

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--real-fixtures", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True,
                        help="Passing six-profile AOT evidence; other shapes are prepared separately")
    parser.add_argument("--runner", type=Path, default=ROOT / "build/m1/coral-ffn-runner")
    parser.add_argument("--cpu-reference", type=Path, default=ROOT / "build/m1/coral-ffn-cpu-reference")
    parser.add_argument("--reference-runner", type=Path, default=ROOT / "build/m3/coral-stablehlo-reference")
    parser.add_argument("--tool", type=Path, default=ROOT / "build/m3/coral-stablehlo-tool")
    parser.add_argument("--output-root", type=Path, default=ROOT / "build/m4-runner-results")
    args = parser.parse_args()
    revision = require_clean()
    for name in ("fixtures", "real_fixtures", "artifacts", "runner", "cpu_reference", "reference_runner", "tool"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    args.output_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="three-way-", dir=args.output_root.resolve()))
    print(f"Evidence: {work}", flush=True)
    def invoke(command, label, timeout=600):
        start = time.perf_counter()
        with (work / (label + ".stdout.txt")).open("x") as out, \
             (work / (label + ".stderr.txt")).open("x") as err:
            result = subprocess.run(list(map(str, command)), stdout=out, stderr=err,
                                    cwd=ROOT, timeout=timeout, check=False)
        require(result.returncode == 0, f"{label}: failed; retained logs in {work}")
        return (time.perf_counter() - start) * 1000

    synthetic_report = json.loads((args.fixtures / "report.json").read_text())
    model_report = json.loads((args.real_fixtures / "report.json").read_text())
    lock = json.loads((ROOT / "dependencies.lock.json").read_text())
    require(synthetic_report["status"] == model_report["status"] == "passed" and
            synthetic_report["positive_cases"] == 14 and model_report["real_ffn_regions"] == 20 and
            model_report["model_sha256"] == lock["model"]["sha256"],
            "expected accepted M3 fixtures for the locked model")
    sources = [("synthetic-" + p.name, p) for p in sorted(args.fixtures.glob("ffn-region-*"))
               if (p / "imported.mlir").exists()]
    require(len(sources) == 14, "incomplete synthetic fixture set")
    for step in range(4):
        for layer in range(5):
            sources.append((f"model-step-{step}-layer-{layer}",
                            args.real_fixtures / f"capture/step-{step}-ffn-{layer}"))
    aot = json.loads((args.artifacts / "report.json").read_text())
    require(aot["status"] == "passed", "AOT preparation report is not passing")
    artifacts = {tuple(item["shape"]): args.artifacts / item["directory"] for item in aot["artifacts"]}
    (work / "cases").mkdir()
    (work / "artifacts").mkdir()
    groups = {}
    cases = []
    for label, source in sources:
        case = work / "cases" / label
        case.mkdir()
        for name in ("inputs.json", "ggml-output.json", "imported.mlir"):
            (case / name).write_bytes((source / name).read_bytes())
        if (source / "capture.json").exists():
            (case / "capture.json").write_bytes((source / "capture.json").read_bytes())
        dims = shape(case / "inputs.json")
        invoke([args.tool, "prepare-ffn", case / "imported.mlir", *dims,
                case / "program.mlir", case / "source-map.json"], "prepare-" + label)
        if dims not in artifacts:
            output = work / "artifacts" / ("m%d-d%d-f%d" % dims)
            invoke([sys.executable, ROOT / "compiler/compile_ffn.py", "--input", case / "imported.mlir",
                    "--m", dims[0], "--d", dims[1], "--f", dims[2], "--output", output],
                   "compile-" + label)
            artifacts[dims] = output
            print(f"Prepared additional AOT profile {dims}", flush=True)
        artifact = artifacts[dims]
        manifest = json.loads((artifact / "manifest.json").read_text())
        semantic = digest(case / "program.mlir")
        require(semantic == manifest["program"]["semantic_sha256"] ==
                digest(artifact / "region.stablehlo.mlir"),
                "captured computation differs from artifact program")
        record = {"case": label, "shape": list(dims), "directory": str(case.relative_to(work)),
                  "input_sha256": digest(case / "inputs.json"), "program_sha256": semantic,
                  "artifact_manifest_sha256": digest(artifact / "manifest.json")}
        cases.append(record)
        groups.setdefault(dims, []).append(record)
    print(f"Prepared {len(cases)} cases in {len(groups)} exact profiles", flush=True)

    # Measure Coral and GGML before launching the slow reference interpreter.
    # Each group reuses one loaded program for distinct inputs/weights.
    execution = []
    for dims, records in groups.items():
        label = "m%d-d%d-f%d" % dims
        group = work / label
        group.mkdir()
        arguments = []
        for record in records:
            arguments += ["--inputs", work / record["directory"] / "inputs.json"]
        cpu_file, coral_file, reload_file = [group / name for name in ("cpu.json", "coral.json", "reload.json")]
        invoke([args.cpu_reference, *arguments, "--repeat", 3, "--output", cpu_file], label + "-cpu")
        invoke([args.runner, "--artifact", artifacts[dims], *arguments, "--repeat", 3,
                "--recreate", 2, "--output", coral_file], label + "-coral")
        invoke([args.runner, "--artifact", artifacts[dims], *arguments, "--repeat", 1,
                "--recreate", 1, "--output", reload_file], label + "-reload")
        cpu, coral, reload = [json.loads(path.read_text()) for path in (cpu_file, coral_file, reload_file)]
        require(len(cpu["cases"]) == len(coral["cases"]) == len(reload["cases"]) == len(records),
                "incomplete batch output")
        require(coral["program_sha256"] == reload["program_sha256"] == records[0]["program_sha256"],
                "loaded program identity mismatch")
        for index, record in enumerate(records):
            case = work / record["directory"]
            current_cpu = cpu["cases"][index]
            current_coral = coral["cases"][index]
            fresh = reload["cases"][index]
            require(current_cpu["input_sha256"] == current_coral["input_sha256"] ==
                    fresh["input_sha256"] == record["input_sha256"], "runner used different input bytes")
            require(current_coral["outputs"] == fresh["outputs"], "fresh process reload changed output")
            write(case / "cpu.json", current_cpu)
            write(case / "coral.json", current_coral)
        for data in (coral, reload):
            for cycle in data["cycles"]:
                stats = cycle["stats"]
                require(stats["module_loads"] == 1 and stats["calls_completed"] == stats["calls_attempted"] and
                        stats["coral_dispatches"] == stats["instruction_samples"] and
                        stats["coral_dispatches"] > 0 and stats["kernel_instructions"] > 0 and
                        stats["host_executable_load_attempts"] == stats["failed_dispatches"] ==
                        stats["observer_errors"] == 0, "missing strict Coral evidence")
        cold, warm, wall = [], [], []
        for item in coral["cases"]:
            for call in item["calls"]:
                (cold if call["first_call_in_context"] else warm).append(call["invoke_ms"])
                wall.append(call["wall_ms"])
        execution.append({"shape": list(dims), "cases": len(records),
                          "coral_report": str(coral_file.relative_to(work)),
                          "cpu_report": str(cpu_file.relative_to(work)),
                          "reload_report": str(reload_file.relative_to(work)),
                          "counted_calls": sum(c["stats"]["calls_completed"] for data in (coral, reload) for c in data["cycles"]),
                          "coral_dispatches": sum(c["stats"]["coral_dispatches"] for data in (coral, reload) for c in data["cycles"]),
                          "kernel_instructions": sum(c["stats"]["kernel_instructions"] for data in (coral, reload) for c in data["cycles"]),
                          "cold_invoke_ms": cold, "warm_invoke_median_ms": statistics.median(warm),
                          "call_wall_median_ms": statistics.median(wall),
                          "module_constructor_median_ms": statistics.median(c["module_constructor_ms"] for c in coral["cycles"]),
                          "cpu_evaluation_median_ms": statistics.median(t for item in cpu["cases"] for t in item["evaluation_ms"])})
        print(f"Coral/CPU and fresh-process reload passed for {label}: {len(records)} cases", flush=True)

    for index, record in enumerate(cases):
        case = work / record["directory"]
        dims = tuple(record["shape"])
        reference = case / "stablehlo.json"
        reference_wall = invoke([args.reference_runner, artifacts[dims] / "region.stablehlo.mlir",
                                  "ffn", case / "inputs.json", reference], "reference-" + record["case"])
        cpu = json.loads((case / "cpu.json").read_text())["outputs"][0]
        coral = json.loads((case / "coral.json").read_text())["outputs"][0]
        hlo_report = json.loads(reference.read_text())
        hlo = hlo_report["outputs"][0]
        captured = json.loads((case / "ggml-output.json").read_text())["outputs"][0]
        comparisons = {"coral_vs_ggml": compare(coral, cpu),
                       "stablehlo_vs_ggml": compare(hlo, cpu),
                       "coral_vs_stablehlo": compare(coral, hlo),
                       "cpu_replay_vs_capture": compare(cpu, captured)}
        record["comparisons"] = comparisons
        record["reference_evaluation_ms"] = hlo_report["evaluation_ms"]
        record["reference_process_ms"] = reference_wall
        write(case / "comparison.json", comparisons)
        print(f"Three-way {index+1}/{len(cases)} {record['case']}: " +
              ("passed" if all(c["status"] == "passed" for c in comparisons.values()) else "FAILED"), flush=True)

    # Same-shaped role mistakes cannot be detected from tensor shapes alone.
    # All engines must agree on the changed program inputs, while comparison
    # with the original captured result must detect the mistake.
    source = work / "cases/model-step-0-layer-0"
    swapped = json.loads((source / "inputs.json").read_text())
    swapped["inputs"][1], swapped["inputs"][2] = swapped["inputs"][2], swapped["inputs"][1]
    swapped_file = work / "swapped-inputs.json"
    write(swapped_file, swapped)
    dims = shape(swapped_file)
    invoke([args.runner, "--artifact", artifacts[dims], "--inputs", swapped_file,
            "--repeat", 1, "--output", work / "swapped-coral.json"], "swapped-coral")
    invoke([args.cpu_reference, "--inputs", swapped_file, "--repeat", 1,
            "--output", work / "swapped-cpu.json"], "swapped-cpu")
    invoke([args.reference_runner, artifacts[dims] / "region.stablehlo.mlir", "ffn",
            swapped_file, work / "swapped-stablehlo.json"], "swapped-stablehlo")
    swapped_coral = json.loads((work / "swapped-coral.json").read_text())["cases"][0]["outputs"][0]
    swapped_cpu = json.loads((work / "swapped-cpu.json").read_text())["cases"][0]["outputs"][0]
    swapped_hlo = json.loads((work / "swapped-stablehlo.json").read_text())["outputs"][0]
    original = json.loads((source / "ggml-output.json").read_text())["outputs"][0]
    negative = {"coral_vs_cpu": compare(swapped_coral, swapped_cpu),
                "coral_vs_stablehlo": compare(swapped_coral, swapped_hlo),
                "against_original_capture": compare(swapped_coral, original)}
    require(negative["coral_vs_cpu"]["status"] == negative["coral_vs_stablehlo"]["status"] == "passed" and
            negative["against_original_capture"]["failed_count"] > 0, "role-swap negative control failed")
    require_clean()
    status = "passed" if all(v["status"] == "passed" for c in cases for v in c["comparisons"].values()) else "failed"
    result = {"status": status, "synthetic_cases": 14, "real_model_cases": 20,
              "profiles": len(groups), "fresh_process_reload": True, "atol": 1e-4, "rtol": 1e-4,
              "llama_revision": revision, "llama_worktree_clean": True,
              "model_sha256": model_report["model_sha256"],
              "model_capture_report_sha256": digest(args.real_fixtures / "report.json"),
              "synthetic_capture_report_sha256": digest(args.fixtures / "report.json"),
              "runner_sha256": digest(args.runner), "cpu_reference_sha256": digest(args.cpu_reference),
              "stablehlo_reference_sha256": digest(args.reference_runner),
              "execution": execution, "cases": cases, "role_swap_control": negative,
              "timing_scope": "Functional-test host wall times: MPACT simulator, not hardware performance"}
    write(work / "report.json", result)
    print(json.dumps({"status": status, "cases": len(cases), "profiles": len(groups),
                      "evidence": str(work)}), flush=True)
    return 0 if status == "passed" else 1

if __name__ == "__main__":
    sys.exit(main())
