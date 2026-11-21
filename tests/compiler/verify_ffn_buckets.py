#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""P2 admission proof, official-reference equality, capacity reuse and boundaries."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "compiler"))
sys.path.insert(0, str(ROOT / "tests/runtime"))
sys.path.insert(0, str(ROOT / "tests/ggml"))
from region_service import Toolchain, optimize_region, specialize_rows, select_row_capacity, invoke
from region_cache import RegionCache
from ffn_compare import compare
from verify_ffn_layout import transpose_inputs, read, write
from verify_unpatched import require_clean


def tiny_inputs(rows):
    result = []
    for role, shape in enumerate(((rows, 3), (4, 3), (4, 3), (3, 4))):
        values = [((i * 7 + role * 5) % 17 - 8) / 16 for i in range(shape[0] * shape[1])]
        result.append({"dtype": "f32", "shape": shape, "values": values})
    return {"inputs": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-fixtures", type=Path, required=True)
    args = parser.parse_args()
    args.real_fixtures = args.real_fixtures.resolve(strict=True)
    revision = require_clean()
    work = Path(tempfile.mkdtemp(prefix="m5-p2-", dir=ROOT / "build"))
    print(f"Evidence: {work}", flush=True)
    tools = Toolchain()
    cache = RegionCache(work / "cache", tools)
    description = tools.describe((1, 3, 4))
    reference = ROOT / "build/m3/coral-stablehlo-reference"
    runner = ROOT / "build/m1/coral-ffn-runner"
    records, packages, cores = [], {}, {}
    for layout in ("native", "transposed"):
        profiles = [(m, 3, 4) for m in (1, 2, 3, 4, 5, 7, 8, 9)] + [(1, 64, 172), (5, 64, 172)]
        for profile in profiles:
            m, d, f = profile
            capacity = select_row_capacity(m, [1, 4, 8] if d == 3 else [8])
            label = f"{layout}-m{m}-d{d}-f{f}"
            case = work / label
            case.mkdir()
            if d == 3:
                inputs = tiny_inputs(m)
                write(case / "inputs.json", inputs)
                invoke([tools.tool, "emit-ffn", *profile, case / "imported.mlir"], case, "emit")
            else:
                step = 1 if m == 1 else 0
                source = args.real_fixtures / f"capture/step-{step}-ffn-0"
                inputs = read(source / "inputs.json")
                for name in ("inputs.json", "imported.mlir"):
                    (case / name).write_bytes((source / name).read_bytes())
            p1 = optimize_region((case / "imported.mlir").read_bytes(), profile, layout,
                                 toolchain=tools, evidence=case / "p1")
            core, compiled_profile = specialize_rows(p1, profile, capacity, layout,
                                                     toolchain=tools, evidence=case / "p2")
            plan = read(case / "p2/plan.json")
            assert plan["logical_m"] == m and plan["capacity_m"] == capacity
            assert plan["pad"]["high"] == [capacity - m, 0]
            assert b"stablehlo.pad" not in core and b"stablehlo.slice" not in core
            witness = (case / "p2/boundary.mlir").read_bytes()
            assert witness.count(b"stablehlo.pad") == witness.count(b"stablehlo.slice") == 1
            key = (layout, capacity, d, f)
            if key in cores:
                assert core == cores[key], "logical M fragmented canonical capacity code"
            else:
                cores[key] = core
            result = cache.resolve(core, compiled_profile, description["compatibility"],
                                   description["contract"]["numeric_policy"], weight_layout=layout)
            assert result["state"] == ("hit" if key in packages else "compiled")
            packages[key] = Path(result["artifact"])
            physical = transpose_inputs(inputs) if layout == "transposed" else inputs
            write(case / "physical-inputs.json", physical)
            invoke([reference, case / "p1/after.mlir", "ffn", case / "physical-inputs.json",
                    case / "before.json"], case, "reference-before")
            invoke([reference, case / "p2/boundary.mlir", "ffn", case / "physical-inputs.json",
                    case / "boundary.json"], case, "reference-boundary")
            expected = read(case / "before.json")["outputs"][0]
            assert expected == read(case / "boundary.json")["outputs"][0], "Pad/Core/Crop changed semantics"
            invoke([runner, "--artifact", result["artifact"], "--weight-layout", layout,
                    "--capacity-m", capacity, "--inputs", case / "inputs.json", "--repeat", 3,
                    "--output", case / "coral.json"], case, "coral")
            invoke([ROOT / "build/m1/coral-ffn-cpu-reference", "--inputs", case / "inputs.json",
                    "--repeat", 1, "--output", case / "cpu.json"], case, "cpu")
            coral, cpu = read(case / "coral.json"), read(case / "cpu.json")
            actual = coral["cases"][0]["outputs"][0]
            comparisons = {"coral_vs_reference": compare(actual, expected),
                           "coral_vs_cpu": compare(actual, cpu["cases"][0]["outputs"][0])}
            assert all(item["status"] == "passed" for item in comparisons.values())
            stats = coral["cycles"][0]["stats"]
            assert stats["ffn_padded_rows"] == stats["ffn_cropped_rows"] == (capacity - m) * 3
            assert stats["upload_bytes"] == 3 * d * f * 4 + 3 * capacity * d * 4
            assert stats["download_bytes"] == 3 * capacity * d * 4
            records.append({"case": label, "shape": profile, "capacity_m": capacity,
                            "cache_state": result["state"], "key": result["key"],
                            "official_reference_bit_exact": True, "comparisons": comparisons,
                            "stats": stats, "calls": coral["cases"][0]["calls"]})
            print(label + ": passed (" + result["state"] + ")", flush=True)

    # One live module and snapshot must survive both increasing/decreasing M.
    for layout in ("native", "transposed"):
        invoke([ROOT / "build/m1/coral-ffn-runtime-test", "bucket", packages[(layout, 8, 3, 4)],
                8, 3, 4, layout], work, "runtime-boundaries-" + layout)
    negatives = 0
    for policy in ([0], [-1], [4, 4], [8, 4], [True], [4.0], list(range(1, 18)), [1 << 30]):
        try:
            select_row_capacity(3, policy)
        except RuntimeError:
            negatives += 1
        else:
            raise AssertionError("invalid bucket policy accepted")
    assert select_row_capacity(3, []) == 3 and select_row_capacity(9, [1, 4, 8]) == 9
    source = (work / "native-m3-d3-f4/p1/after.mlir").read_bytes()
    # On this square X the first modified dot remains well typed while mixing
    # token rows. Full FFN admission, not shape checking alone, must reject it.
    unsafe = source.replace(b"contracting_dims = [1] x [1]",
                            b"contracting_dims = [0] x [1]", 1)
    (work / "row-mixing.mlir").write_bytes(unsafe)
    invoke([tools.tool, "verify", work / "row-mixing.mlir"], work, "verify-row-mixing")
    for label, ir, capacity in (("row-mixing", unsafe, 4), ("truncation", source, 2)):
        try:
            specialize_rows(ir, (3, 3, 4), capacity, "native", toolchain=tools,
                            evidence=work / ("reject-" + label))
        except RuntimeError:
            negatives += 1
        else:
            raise AssertionError("unsafe row specialization accepted")
    report = {"status": "passed", "revision": revision, "cases": records,
              "canonical_artifacts": len(packages), "requests": len(records),
              "negative_cases": negatives + 16,
              "exact_above_largest_bucket": True,
              "runtime_reuse": {layout: json.loads((work / ("runtime-boundaries-" + layout +
                                                   ".stdout.txt")).read_text().splitlines()[-1])
                               for layout in ("native", "transposed")}}
    write(work / "report.json", report)
    print("P2 acceptance passed: " + str(work / "report.json"), flush=True)


if __name__ == "__main__":
    main()
