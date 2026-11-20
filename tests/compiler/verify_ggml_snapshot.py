#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify the bounded metadata transport before the existing GGML importer."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
ROOT = Path(__file__).resolve().parents[2]


def main():
    source = Path(sys.argv[1])
    work = Path(tempfile.mkdtemp(prefix="m5-snapshot-", dir=ROOT / "build"))
    tool = ROOT / "build/m3/coral-stablehlo-tool"
    baseline = json.loads(source.read_text())
    checks = []
    def invoke(label, wire, success=False):
        path = work / (label + ".json")
        if isinstance(wire, dict):
            path.write_text(json.dumps(wire))
        else:
            path.write_text(wire)
        output = work / (label + ".mlir")
        result = subprocess.run([tool, "import-ggml-ffn", path, output],
                                capture_output=True, text=True, timeout=30)
        (work / (label + ".log")).write_text(result.stdout + result.stderr)
        assert (result.returncode == 0) == success, label + ": " + result.stderr
        assert output.exists() == success
        checks.append({"case": label, "accepted": success})
        return output
    valid = invoke("valid", baseline, True)
    # Shared gate/up storage still has four positional function arguments.
    shared = copy.deepcopy(baseline)
    removed = shared["inputs"][2]
    gate = shared["inputs"][1]
    shared["inputs"][2] = gate
    for node in shared["nodes"]:
        node["src"] = [gate if value == removed else value for value in node["src"]]
    shared["nodes"].pop(removed - 1)
    for node in shared["nodes"]:
        node["src"] = [value - int(value > removed) for value in node["src"]]
    for name in ("inputs", "operations"):
        shared[name] = [value - int(value > removed) for value in shared[name]]
    shared["output"] -= int(shared["output"] > removed)
    shared_ir = invoke("shared-weights", shared, True)
    for label, ir in (("valid", valid), ("shared", shared_ir)):
        subprocess.run([tool, "prepare-ffn", ir, *map(str, baseline["shape"]),
                        work / (label + ".canonical.mlir"), work / (label + ".metadata.json")], check=True)
    assert (work / "valid.canonical.mlir").read_bytes() == (work / "shared.canonical.mlir").read_bytes()
    for label in ("schema", "unknown-field", "float-shape", "wrong-profile", "dtype", "stride",
                  "op", "params", "cycle", "missing-member", "duplicate-member", "wrong-roles"):
        wire = copy.deepcopy(baseline)
        first = wire["operations"][0] - 1
        if label == "schema": wire["schema"] = "unknown"
        elif label == "unknown-field": wire["data"] = "forbidden"
        elif label == "float-shape": wire["shape"][0] = float(wire["shape"][0])
        elif label == "wrong-profile": wire["shape"][0] += 1
        elif label == "dtype": wire["nodes"][0]["type"] = 1
        elif label == "stride": wire["nodes"][0]["nb"][1] += 4
        elif label == "op": wire["nodes"][first]["op"] = 0
        elif label == "params": wire["nodes"][first]["op_params"][0] = 999
        elif label == "cycle": wire["nodes"][first]["src"][0] = wire["operations"][0]
        elif label == "missing-member": wire["operations"].pop()
        elif label == "duplicate-member": wire["operations"][1] = wire["operations"][0]
        else: wire["inputs"][1], wire["inputs"][2] = wire["inputs"][2], wire["inputs"][1]
        invoke(label, wire)
    invoke("duplicate-key", '{"schema":"bad",' + json.dumps(baseline)[1:])
    invoke("malformed", "{")
    invoke("oversize", " " * (64 * 1024 + 1))
    report = {"status": "passed", "positive": 2, "negative": len(checks) - 2,
              "shared_roles_preserved": True, "cases": checks}
    (work / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "passed", "report": str(work / "report.json"),
                      "positive": 2, "negative": len(checks) - 2}))


if __name__ == "__main__":
    main()
