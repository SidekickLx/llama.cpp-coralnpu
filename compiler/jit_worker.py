#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Opt-in local FFN compiler worker; never runs numerical model computation."""
import argparse
import json
import os
from pathlib import Path
import sys
from region_cache import RegionCache, identity_hash, read_json
from region_service import Toolchain, optimize_region, specialize_rows, select_row_capacity, invoke, require
from ffn_artifact import write_json


def configure(args):
    tools = Toolchain(args.tool, args.checker, args.compiler)
    cache = RegionCache(args.cache.resolve(), tools)
    output = args.output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    require(1 <= args.timeout <= 600, "timeout must be in [1,600] seconds")
    select_row_capacity(1, args.m_buckets)
    write_json(output, {"schema": "coral.ffn.jit-config.v1",
                       "cache": str(cache.root),
                       "environment": cache.environment, "paths": cache.paths,
                       "environment_sha256": identity_hash(cache.environment),
                       "timeout_seconds": args.timeout,
                       "optimization": {"weight_layout": args.weight_layout, "m_buckets": args.m_buckets}})
    print(json.dumps({"status": "configured", "configuration": str(output),
                      "environment_sha256": identity_hash(cache.environment)}))


def compile_request(args):
    config = read_json(args.config)
    require(config["schema"] == "coral.ffn.jit-config.v1", "unknown JIT configuration")
    require(identity_hash(config["environment"]) == config["environment_sha256"],
            "JIT environment digest mismatch")
    optimization = config["optimization"]
    require(set(optimization) == {"weight_layout", "m_buckets"} and
            optimization["weight_layout"] in ("native", "transposed"), "invalid optimization policy")
    layout = optimization["weight_layout"]
    paths = config["paths"]
    tools = Toolchain(Path(paths["tool"]), Path(paths["checker"]), Path(paths["compiler"]))
    cache = RegionCache(Path(config["cache"]), tools)
    require(cache.environment == config["environment"],
            "stale JIT configuration: regenerate after changing the compiler/frontend/service")
    work = args.request.resolve(strict=True).parent
    require(work.is_relative_to(cache.root / "staging"), "request evidence must be in cache staging")
    request = read_json(args.request, 64 * 1024)
    require(set(request) == {"schema", "environment_sha256", "snapshot", "optimization"} and
            request["schema"] == "coral.ffn.jit-request.v1" and
            request["environment_sha256"] == config["environment_sha256"] and
            request["optimization"] == optimization,
            "invalid JIT request")
    request_key = identity_hash(request)
    snapshot = request["snapshot"]
    profile = snapshot["shape"]
    capacity = select_row_capacity(profile[0], optimization["m_buckets"])
    tools.describe(profile)
    write_json(work / "snapshot.json", snapshot)
    invoke([tools.tool, "import-ggml-ffn", work / "snapshot.json", work / "imported.mlir"],
           work, "import")
    canonical = optimize_region((work / "imported.mlir").read_bytes(), profile, layout,
                                toolchain=tools, evidence=work / "normalization")
    if optimization["m_buckets"]:
        canonical, profile = specialize_rows(canonical, profile, capacity, layout,
                                             toolchain=tools, evidence=work / "rows")
    description = tools.describe(profile, layout)
    result = cache.resolve(canonical, profile, description["compatibility"],
                           description["contract"]["numeric_policy"], weight_layout=layout)
    # This index is a memo of a verified GGML snapshot -> canonical artifact.
    # It lets a later process validate a hit without loading MLIR or a worker.
    index = {"schema": "coral.ffn.request-index.v1",
             "request_sha256": request_key, "key": result["key"]}
    pending = work / "request-index.pending.json"
    write_json(pending, index)
    with cache.writer():
        os.replace(pending, cache.root / "requests" / (request_key + ".json"))
    response = {"schema": "coral.ffn.jit-response.v1", "status": "passed",
                "request_sha256": request_key, "key": result["key"],
                "state": result["state"], "recovered": result["recovered"],
                "compile_ms": result["compile_ms"]}
    write_json(work / "response.json", response)
    print(json.dumps(response), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("configure")
    setup.add_argument("--cache", type=Path, required=True)
    setup.add_argument("--output", type=Path, required=True)
    setup.add_argument("--timeout", type=int, default=600)
    setup.add_argument("--weight-layout", choices=("native", "transposed"), default="native")
    setup.add_argument("--m-buckets", type=int, nargs="+", default=[])
    defaults = Toolchain()
    for name in ("tool", "checker", "compiler"):
        setup.add_argument("--" + name, type=Path, default=getattr(defaults, name))
    compile_parser = commands.add_parser("compile")
    compile_parser.add_argument("--config", type=Path, required=True)
    compile_parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "configure":
        configure(args)
    else:
        compile_request(args)


if __name__ == "__main__":
    main()
