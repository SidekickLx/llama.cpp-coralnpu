#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Validate exported metadata against real CPU and Coral GGUF execution."""
import argparse
import collections
import json
from pathlib import Path
import tempfile

from verify_runner import invoke, read_run, require, ATOL, RTOL


def snapshot(data):
    tensors = {t["id"]: t for t in data["tensors"]}
    require(len(tensors) == len(data["tensors"]), "duplicate tensor ID")
    require(len(data["nodes"]) == len(set(data["nodes"])), "duplicate graph node")
    require(set(data["nodes"]) <= tensors.keys(), "missing node metadata")
    expected = collections.defaultdict(list)
    for tensor in tensors.values():
        require(len(tensor["ne"]) == len(tensor["nb"]) == 4, "missing dimensions/strides")
        require(len(tensor["op_params"]) == 16, "missing op parameter words")
        for slot, src in enumerate(tensor["src"]):
            if src:
                require(src in tensors, "dangling source")
                expected[src].append((tensor["id"], slot))
        require(not tensor["view_src"] or tensor["view_src"] in tensors, "dangling view")
    for tensor in tensors.values():
        actual = [(c["id"], c["slot"]) for c in tensor["consumers"]]
        require(sorted(actual) == sorted(expected[tensor["id"]]), "incorrect consumers")
    return tensors


def inspect(path, mode):
    graph = json.loads(path.read_text())
    require(graph["schema"] == "coral-ggml-graph-v1", "wrong graph schema")
    require(graph["source_stage"] == "before_scheduler_split", "wrong source stage")
    require(graph["schedule_stage"] == "before_split_execution", "wrong schedule stage")
    tensors = snapshot(graph["source"])
    scheduled = {}
    rewritten = 0
    for i, split in enumerate(graph["splits"]):
        require(split["id"] == i, "noncontiguous split IDs")
        current = snapshot(split["graph"])
        for node in split["graph"]["nodes"]:
            require(node not in scheduled, "node in multiple splits")
            require(node in tensors, "optimized node requires a new schema mapping")
            scheduled[node] = split["backend"]
            if current[node]["src"] != tensors[node]["src"]:
                rewritten += 1
    require(set(scheduled) == set(graph["source"]["nodes"]), "incomplete graph coverage")
    require(any(t["view_src"] for t in tensors.values()), "fixture must exercise aliases")
    shapes = []
    for node in graph["source"]["nodes"]:
        glu = tensors[node]
        if glu["op"] != "GLU":
            continue
        require(glu["op_params"] == [2] + [0] * 15, "not canonical split SwiGLU")
        gate, up = [tensors[x] for x in glu["src"][:2]]
        require(gate["op"] == up["op"] == "MUL_MAT", "missing FFN projections")
        require(gate["src"][1] == up["src"][1], "gate/up activation mismatch")
        require(len(glu["consumers"]) == 1, "unexpected hidden consumer")
        consumer = glu["consumers"][0]
        down = tensors[consumer["id"]]
        require(down["op"] == "MUL_MAT" and consumer["slot"] == 1, "missing down projection")
        x = tensors[gate["src"][1]]
        wg = tensors[gate["src"][0]]
        wu = tensors[up["src"][0]]
        wd = tensors[down["src"][0]]
        require(x["ne"][0] == 64 and wg["ne"] == wu["ne"] == [64, 172, 1, 1],
                "unexpected model FFN dimensions")
        require(wd["ne"] == [172, 64, 1, 1], "wrong down orientation")
        require(all(t["dtype"] == "f32" for t in (x, wg, wu, wd, glu, down)),
                "non-F32 fixture")
        shapes.append([x["ne"][1], 64, 172])
        require(scheduled[node] == "CPU", "GLU unexpectedly offloaded")
        for projection in (gate, up, down):
            require(scheduled[projection["id"]] == ("CORAL" if mode == "coral" else "CPU"),
                    "wrong FFN projection placement")
    require(len(shapes) == 5, "expected five FFNs")
    if mode == "coral":
        require(len(graph["splits"]) > 1, "missing cross-backend splits")
        # Coral uses host-visible buffers. Distinct backends may share storage
        # without scheduler copy nodes; do not invent a copy requirement.
    else:
        require(rewritten == 0 and len(graph["splits"]) == 1, "unexpected CPU splits")
    return graph, {"step": graph["step"], "nodes": len(scheduled),
                   "tensors": len(tensors), "splits": len(graph["splits"]),
                   "rewritten_nodes": rewritten, "FFN_shapes": shapes,
                   "rebuilt": graph["graph_rebuilt"]}


def run(args, work):
    runner, model = args.runner.resolve(), args.model.resolve()
    baseline = work / "baseline"
    common = ["--model", model]
    invoke(runner, ["--mode", "cpu", *common, "--output", baseline], "baseline", work)
    _, reference = read_run(baseline, "cpu")
    replay = ["--tokens", baseline / "tokens.txt"]
    report = {}
    cpu_graphs = None
    for label, mode in (("cpu", "cpu"), ("repeat", "cpu"), ("coral", "coral")):
        output = work / label
        options = ["--mode", mode, *common, *replay, "--output", output, "--export-graph"]
        if mode == "coral":
            options.extend(["--artifacts", args.artifacts.resolve()])
        invoke(runner, options, label, work)
        summary, values = read_run(output, mode)
        require(summary["graph_exports"] == 4, "missing graph export count")
        require(len(values) == len(reference), "logit shape mismatch")
        error = max(abs(a - b) for a, b in zip(values, reference))
        if mode == "cpu":
            require((output / "logits.f32").read_bytes() == (baseline / "logits.f32").read_bytes(),
                    "observation changed CPU logits")
        else:
            require(all(abs(a - b) <= ATOL + RTOL * abs(b) for a, b in zip(values, reference)),
                    "Coral logits differ from CPU")
        graphs, evidence = [], []
        for step in range(4):
            graph, counts = inspect(output / f"graph-{step}.json", mode)
            graphs.append(graph)
            evidence.append(counts)
        require(graphs[0]["graph_rebuilt"] and graphs[1]["graph_rebuilt"], "missing rebuild")
        for graph in graphs[2:]:
            require(not graph["graph_rebuilt"], "decode should exercise graph reuse")
            require(graph["source"] == graphs[1]["source"], "reused source snapshot changed")
            require(graph["graph_generation"] == graphs[1]["graph_generation"], "wrong generation")
        if label == "cpu":
            cpu_graphs = graphs
        elif label == "repeat":
            require(graphs == cpu_graphs, "nondeterministic metadata/IDs across fresh runs")
        report[label] = {"max_logit_abs_error": error, "graphs": evidence,
                         "coral_dispatches": summary["stats"]["coral_dispatches"]}
    report["status"] = "passed"
    (work / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "passed", "evidence": str(work),
                      "coral_max_logit_abs_error": report["coral"]["max_logit_abs_error"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runner", "model", "artifacts", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="graph-", dir=args.output_root.resolve()))
    run(args, work)


if __name__ == "__main__":
    main()
