#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""P1: imported-graph rewrite, official interpreter and real Coral execution."""
import argparse
import copy
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "compiler"))
sys.path.insert(0, str(ROOT / "tests/runtime"))
sys.path.insert(0, str(ROOT / "tests/ggml"))
from region_service import Toolchain, compile_region, optimize_region, invoke
from ffn_compare import compare
from verify_unpatched import require_clean
from verify_ffn_members import copy_catalog, compare as compare_logits
from verify_runner import read_run
from verify_ffn_regions import inspect_regions


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def transpose_inputs(document):
    result = copy.deepcopy(document)
    for tensor in result["inputs"][1:]:
        rows, cols = tensor["shape"]
        values = tensor["values"]
        tensor["shape"] = [cols, rows]
        tensor["values"] = [values[row * cols + col]
                            for col in range(cols) for row in range(rows)]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fixtures", "real-fixtures", "aot", "matmuls", "pointwise"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    for name in ("fixtures", "real_fixtures", "aot", "matmuls", "pointwise"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    revision = require_clean()
    work = Path(tempfile.mkdtemp(prefix="m5-p1-", dir=ROOT / "build"))
    print(f"Evidence: {work}", flush=True)
    toolchain = Toolchain()
    runner = ROOT / "build/m1/coral-ffn-runner"
    reference = ROOT / "build/m3/coral-stablehlo-reference"
    packages = {}
    records = []
    sources = [("tiny", args.fixtures / "ffn-region-split"),
               ("real-decode", args.real_fixtures / "capture/step-1-ffn-0"),
               ("real-prefill", args.real_fixtures / "capture/step-0-ffn-0")]
    for label, source in sources:
        inputs = read(source / "inputs.json")
        m, d = inputs["inputs"][0]["shape"]
        f = inputs["inputs"][1]["shape"][0]
        profile = (m, d, f)
        case = work / label
        case.mkdir()
        before = (source / "imported.mlir").read_bytes()
        after = optimize_region(before, profile, "transposed", toolchain=toolchain,
                                evidence=case / "pass")
        plan = read(case / "pass/plan.json")
        assert len(plan["weight_preparation"]) == 3
        assert plan["removed_identity_f32_converts"] == 0
        assert after.count(b"contracting_dims = [1] x [0]") == 3
        assert b"stablehlo.transpose" not in after and b"stablehlo.convert" not in after
        description = toolchain.describe(profile, "transposed")
        package = compile_region(after, profile, description["compatibility"],
                                 description["contract"]["numeric_policy"], output=case / "package",
                                 weight_layout="transposed")
        packages[label] = package
        write(case / "transposed-inputs.json", transpose_inputs(inputs))
        invoke([reference, case / "pass/before.mlir", "ffn", source / "inputs.json",
                case / "reference-before.json"], case, "reference-before")
        invoke([reference, case / "pass/after.mlir", "ffn", case / "transposed-inputs.json",
                case / "reference-after.json"], case, "reference-after")
        original = read(case / "reference-before.json")["outputs"][0]
        transformed = read(case / "reference-after.json")["outputs"][0]
        assert original == transformed, "P1 changed official reference output"
        invoke([ROOT / "build/m1/coral-ffn-cpu-reference", "--inputs", source / "inputs.json",
                "--output", case / "cpu.json"], case, "cpu")
        for layout, artifact in (("native", args.aot / label), ("transposed", package)):
            invoke([runner, "--artifact", artifact, "--weight-layout", layout, "--inputs",
                    source / "inputs.json", "--repeat", 10, "--recreate", 2,
                    "--output", case / (layout + ".json")], case, layout)
        native, optimized, cpu = [read(case / (name + ".json"))
                                  for name in ("native", "transposed", "cpu")]
        actual = optimized["cases"][0]["outputs"][0]
        comparisons = {"coral_vs_cpu": compare(actual, cpu["cases"][0]["outputs"][0]),
                       "coral_vs_reference": compare(actual, transformed),
                       "coral_vs_native": compare(actual, native["cases"][0]["outputs"][0])}
        assert all(item["status"] == "passed" for item in comparisons.values())
        measurements = {}
        for layout, report in (("native", native), ("transposed", optimized)):
            calls = report["cases"][0]["calls"]
            measurements[layout] = {
                "warm_wall_median_ms": statistics.median(
                    c["wall_ms"] for c in calls if not c["first_call_in_context"]),
                "warm_invoke_median_ms": statistics.median(
                    c["invoke_ms"] for c in calls if not c["first_call_in_context"]),
                "first_call_wall_ms": [c["wall_ms"] for c in calls if c["first_call_in_context"]],
                "weight_prepare_median_ms": statistics.median(report["cases"][0]["weight_preparations_ms"]),
                "dispatches_per_call": calls[0]["coral_dispatches"],
                "instructions_per_call": calls[0]["kernel_instructions"],
                "upload_bytes": report["cycles"][0]["stats"]["upload_bytes"],
                "download_bytes": report["cycles"][0]["stats"]["download_bytes"],
                "vmfb_bytes": (args.aot / label / "region.vmfb").stat().st_size
                    if layout == "native" else (package / "region.vmfb").stat().st_size}
        records.append({"case": label, "shape": profile, "official_reference_bit_exact": True,
                        "comparisons": comparisons, "measurements": measurements})
        print(label + ": three-way passed", flush=True)

    # A same-type convert is removable; a rounding conversion is not. These are
    # standard StableHLO operations, checked by the same verifier and admission.
    tiny_source = sources[0][1] / "imported.mlir"
    native = (args.aot / "tiny/region.stablehlo.mlir").read_text()
    position = native.index("    %")
    identity = native[:position] + "    %identity = stablehlo.convert %arg0 : tensor<2x3xf32>\n" + \
        native[position:].replace("%arg0", "%identity")
    converted = optimize_region(identity.encode(), (2, 3, 4), "transposed",
                                toolchain=toolchain, evidence=work / "identity-convert")
    assert read(work / "identity-convert/plan.json")["removed_identity_f32_converts"] == 1
    assert converted == (work / "tiny/pass/after.mlir").read_bytes()
    negatives = {
        "wrong-math": native.replace("stablehlo.exponential", "stablehlo.abs"),
        "rounding": native[:position] +
            "    %half = stablehlo.convert %arg0 : (tensor<2x3xf32>) -> tensor<2x3xf16>\n"
            "    %rounded = stablehlo.convert %half : (tensor<2x3xf16>) -> tensor<2x3xf32>\n" +
            native[position:].replace("%arg0", "%rounded"),
        "wrong-shape": native.replace("tensor<2x3xf32>", "tensor<3x3xf32>"),
        "already-transposed": converted.decode(),
    }
    for label, ir in negatives.items():
        try:
            optimize_region(ir.encode(), (2, 3, 4), "transposed", toolchain=toolchain,
                            evidence=work / label)
        except RuntimeError:
            assert not (work / label / "after.mlir").exists()
        else:
            raise AssertionError("accepted " + label)
    # A transposed package cannot be used under the old ABI, including square
    # cases where shape checks alone would not distinguish physical layout.
    result = subprocess.run([toolchain.checker, packages["tiny"], "2", "3", "4"],
                            capture_output=True, text=True)
    assert result.returncode != 0 and "contract" in result.stderr
    (work / "abi-mismatch.log").write_text(result.stderr)

    catalog = work / "catalog"
    catalog.mkdir()
    copy_catalog(args.matmuls, catalog, "profiles.tsv")
    copy_catalog(args.pointwise, catalog, "pointwise.tsv")
    # Catalog entries must be real local directories, not symlinks.
    import shutil
    for label in ("real-decode", "real-prefill"):
        shutil.copytree(packages[label], catalog / label)
    (catalog / "ffn.tsv").write_text(
        "coral-m4-ffn-v1\n1 64 172 real-decode transposed\n5 64 172 real-prefill transposed\n")
    for mode in ("mutable", "immutable", "weight-generation"):
        invoke([ROOT / "build/m1/coral-ffn-region-test", mode, catalog], work, mode)
    llama = ROOT / "build/m1/coral-llama-runner"
    model = ROOT / "models/stories260K-f32.gguf"
    invoke([llama, "--mode", "cpu", "--model", model, "--output", work / "cpu-model"], work, "cpu-model")
    invoke([llama, "--mode", "coral", "--model", model, "--artifacts", catalog,
            "--observation", "backend", "--tokens", work / "cpu-model/tokens.txt",
            "--output", work / "coral-model"], work, "coral-model")
    _, expected = read_run(work / "cpu-model", "cpu")
    summary, actual = read_run(work / "coral-model", "coral")
    inspect_regions(work / "coral-model", 20, {1: 3, 5: 3})
    error = compare_logits(actual, expected)
    assert summary["stats"]["ffn_weight_preparations"] == 5
    assert summary["stats"]["ffn_weight_cache_hits"] == 15
    report = {"status": "passed", "revision": revision, "profiles": records,
              "negative_cases": len(negatives) + 1, "identity_convert_eliminated": True,
              "weight_modes": ["mutable", "immutable", "weight-generation"],
              "model_max_abs_error": error, "model_stats": summary["stats"]}
    write(work / "report.json", report)
    print("P1 acceptance passed: " + str(work / "report.json"), flush=True)


if __name__ == "__main__":
    main()
