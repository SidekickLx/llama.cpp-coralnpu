#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Exercise real compiler misses, validated hits and durable cache failures."""
import copy
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "compiler"))
from region_service import Toolchain, canonicalize_region
from region_cache import RegionCache, cache_identity, identity_hash


def main():
    work = Path(tempfile.mkdtemp(prefix="m5-cache-", dir=ROOT / "build"))
    print(f"Evidence: {work}", flush=True)
    tools = Toolchain()
    shape = (2, 3, 4)
    description = tools.describe(shape)
    target, policy = description["compatibility"], description["contract"]["numeric_policy"]
    ir = canonicalize_region(Path(sys.argv[1]).read_bytes(), shape,
                             toolchain=tools, evidence=work / "prepare")
    cache = RegionCache(work / "cache", tools)
    cold = cache.resolve(ir, shape, target, policy)
    assert cold["state"] == "compiled"
    hot = cache.resolve(ir, shape, target, policy)
    fresh = RegionCache(work / "cache").resolve(ir, shape, target, policy)
    assert hot["state"] == fresh["state"] == "hit" and cold["key"] == hot["key"] == fresh["key"]
    entry = Path(cold["artifact"])
    original_vmfb = (entry / "region.vmfb").read_bytes()
    repairs = []
    for damage in ("corrupt-vmfb", "incomplete-entry", "wrong-cache-identity"):
        if damage == "corrupt-vmfb":
            (entry / "region.vmfb").write_bytes(b"corrupt")
        elif damage == "incomplete-entry":
            (entry / "manifest.json").rename(entry / "incomplete-manifest.json")
        else:
            data = json.loads((entry / "cache.json").read_text())
            data["pipeline"] = "unknown"
            (entry / "cache.json").write_text(json.dumps(data))
        repaired = cache.resolve(ir, shape, target, policy)
        assert repaired["state"] == "compiled" and repaired["recovered"]
        repairs.append(damage)
    assert len(list((work / "cache/quarantine").iterdir())) == 3

    # An interrupted staging tree is never considered a completed cache entry.
    incomplete = work / "cache/staging/interrupted"
    incomplete.mkdir()
    (incomplete / "manifest.json").write_text('{"state":"building"}')
    assert cache.resolve(ir, shape, target, policy)["state"] == "hit"
    assert incomplete.exists()
    assert "shared:libIREECompiler.so" in cache.environment["files"]
    # A changed real frontend executable must produce a different key and
    # compile a new package, even with the same profile and semantic IR.
    frontend = work / "changed-frontend"
    shutil.copyfile(tools.tool.resolve(), frontend)
    with frontend.open("ab") as stream:
        stream.write(b"cache version invalidation control")
    frontend.chmod(0o755)
    changed_cache = RegionCache(work / "cache", Toolchain(frontend, tools.checker, tools.compiler))
    changed_tool = changed_cache.resolve(ir, shape, target, policy)
    assert changed_tool["state"] == "compiled" and changed_tool["key"] != cold["key"]
    identity = cache_identity(ir, shape, target, policy, cache.environment)
    keys = {}
    for field in ("compiler", "frontend", "pipeline", "shape", "dtype", "layout", "target", "policy"):
        changed = copy.deepcopy(identity)
        if field in ("compiler", "frontend"):
            changed["environment"]["files"]["compiler" if field == "compiler" else "tool"] = "0" * 64
        elif field == "pipeline": changed["pipeline"] += ".next"
        elif field == "shape": changed["profile"]["m"] += 1
        elif field == "dtype": changed["profile"]["dtype"] = "f16"
        elif field == "layout": changed["profile"]["layout"] = "transposed"
        elif field == "target": changed["target"]["target"]["features"] += ",+new"
        else: changed["numeric_policy"] = "fast"
        keys[field] = identity_hash(changed)
        assert keys[field] != cold["key"]
    assert len(set(keys.values())) == len(keys)
    invalid = ir.replace(b"stablehlo.exponential", b"stablehlo.abs")
    bad_key = identity_hash(cache_identity(invalid, shape, target, policy, cache.environment))
    failures = 0
    for attempt in range(2):
        try:
            cache.resolve(invalid, shape, target, policy)
        except RuntimeError:
            failures += 1
        else:
            raise AssertionError("invalid program was cached")
        assert not (work / "cache/entries" / bad_key).exists()
    assert len(list((work / "cache/staging").glob("*/failure.json"))) == failures

    # Reject concurrent writers explicitly; no incomplete entry is published.
    with (work / "cache/.writer.lock").open("r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            cache.resolve(invalid, shape, target, policy)
        except RuntimeError as error:
            assert "busy" in str(error)
        else:
            raise AssertionError("concurrent writer was admitted")
    assert (entry / "region.vmfb").read_bytes() == original_vmfb
    subprocess.run([ROOT / "build/m1/coral-ffn-runtime-test", "run", entry, "2", "3", "4"],
                   check=True, stdout=(work / "runtime.log").open("w"))
    report = {"status": "passed", "cold": cold, "warm": hot, "fresh_cache_instance": fresh,
              "repairs": repairs, "key_invalidations": list(keys), "compile_failures": failures,
              "interrupted_staging_ignored": True, "concurrent_writer_rejected": True,
              "runtime_calls": 14, "changed_frontend": changed_tool,
              "compiler_shared_library_hashed": True}
    (work / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
