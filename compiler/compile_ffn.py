#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compile a verified complete FFN and publish a new immutable AOT package."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
from ffn_artifact import digest, make_manifest, reflection, write_json

ROOT = Path(__file__).resolve().parents[1]

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def invoke(command, directory, label, timeout=600):
    with (directory / (label + ".stdout.txt")).open("x") as out, \
         (directory / (label + ".stderr.txt")).open("x") as err:
        result = subprocess.run(list(map(str, command)), cwd=directory,
                                stdout=out, stderr=err, timeout=timeout, check=False)
    require(result.returncode == 0, f"{label} failed; retained logs: {directory}")

def check_sources(description):
    lock_path = ROOT / "dependencies.lock.json"
    require(digest(lock_path.read_bytes()) == description["compatibility"]["dependency_lock_sha256"],
            "reconfigure the artifact checker after a dependency lock change")
    lock = json.loads(lock_path.read_text())
    sources = {
        "coralnpu-compiler": ROOT / "repos/coralnpu-compiler",
        "iree": ROOT / "repos/coralnpu-compiler/third_party/iree",
        "llvm-project": ROOT / "repos/coralnpu-compiler/third_party/llvm-project",
        "stablehlo": ROOT / "repos/coralnpu-compiler/third_party/iree/third_party/stablehlo",
    }
    for name, path in sources.items():
        actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        require(actual == lock["repositories"][name]["revision"], f"{name}: wrong source revision")
    for name, expected in description["compatibility"]["runtime_sources"].items():
        require(digest((sources["coralnpu-compiler"] / name).read_bytes()) == expected,
                "CRT/linker sources differ from runtime build")

def publish(directory, destination):
    # Linux renameat2 provides atomic publication without replacing an existing
    # directory, including a concurrent creator's empty directory.
    libc = ctypes.CDLL(None, use_errno=True)
    rename = libc.renameat2
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    result = rename(-100, os.fsencode(directory), -100, os.fsencode(destination), 1)
    if result:
        raise OSError(ctypes.get_errno(), "artifact publication failed", str(destination))

def compile_region(args):
    output = args.output.absolute()
    require(not output.exists() and not output.is_symlink(), "output must be a new directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=output.name + ".building-", dir=output.parent))
    print(f"Build evidence: {work}", flush=True)
    checker, tool, compiler = args.checker.resolve(strict=True), args.tool.resolve(strict=True), args.compiler.resolve(strict=True)
    dimensions = [str(args.m), str(args.d), str(args.f)]
    description = json.loads(subprocess.check_output([str(checker), "--describe", *dimensions], text=True))
    check_sources(description)
    # Use the pinned distribution's actual linker and archives explicitly.
    support_root = Path(str(compiler) + ".runfiles") / "_main"
    support = {
        "linker": support_root / "toolchain_rv32/bin/riscv32-unknown-elf-ld",
        "linker_script": support_root / "crt/coralnpu_tcm.ld",
        "iree_archive": support_root / "crt/libcoralnpu_iree.a",
        "crt_archive": support_root / "crt/libcoralnpu_crt.a",
    }
    support_hashes = {name: digest(path.read_bytes()) for name, path in support.items()}
    binary_hash = digest(compiler.read_bytes())
    source_bytes = args.input.read_bytes()
    (work / "source.mlir").write_bytes(source_bytes)
    start = time.perf_counter()
    invoke([tool, "prepare-ffn", work / "source.mlir", *dimensions,
            work / "region.stablehlo.mlir", work / "region.metadata.json"], work, "prepare")
    semantic = digest((work / "region.stablehlo.mlir").read_bytes())
    invoke([tool, "bind-ffn-reflection", work / "region.stablehlo.mlir",
            reflection(description, semantic), work / "compile.mlir"], work, "reflection")
    (work / "executables").mkdir()
    command = [compiler, "compile.mlir", *description["compiler_options"],
        "--coralnpu-embedded-linker-path=" + str(support["linker"]),
        "--coralnpu-linker-script-path=" + str(support["linker_script"]),
        "--coralnpu-dump-affinity-profile-format=json",
        "--coralnpu-dump-affinity-profile-file=affinity.json",
        "--iree-hal-dump-executable-sources-to=executables",
        "--iree-hal-dump-executable-binaries-to=executables",
        "-o", "region.vmfb"]
    invoke(command, work, "compile")
    affinities = json.loads((work / "affinity.json").read_text())["affinities"]
    count = 0
    for target, profile in affinities.items():
        require(profile["dynamic-dispatch-count"] == 0, "dynamic dispatch outside the static FFN contract")
        if target == "coralnpu":
            count += profile["dispatch-count"]
        else:
            require(profile["dispatch-count"] == 0, "numerical dispatch remained on " + target)
    require(count > 0, "no complete Coral dispatch placement")
    executables = []
    for source in sorted((work / "executables").glob("*.mlir")):
        text = source.read_text()
        targets = re.findall(r'target\(<"([^"]+)"', text)
        require(targets and all(target == "coralnpu" for target in targets),
                "executable has a missing or non-Coral target")
        # Executable source dumps retain the compiler's actual export names.
        for name in re.findall(r"hal\.executable\.export\s+(?:public\s+|private\s+)?@([A-Za-z0-9_.$-]+)", text):
            executables.append({"name": name, "target": "coralnpu"})
    # Multiple call sites may share one deduplicated executable. The affinity
    # report counts scheduled dispatches; source dumps enumerate unique exports.
    require(0 < len(executables) <= count, "incomplete executable placement evidence")
    require(len({item["name"] for item in executables}) == len(executables),
            "ambiguous duplicate executable export names")
    placement = {"numerical_target": "coralnpu", "host_dispatches": 0, "coral_dispatches": count}
    write_json(work / "placement.json", {"summary": placement, "executables": executables})
    require(digest(compiler.read_bytes()) == binary_hash, "compiler changed during build")
    require(all(digest(path.read_bytes()) == support_hashes[name] for name, path in support.items()),
            "compiler support files changed during build")
    manifest = make_manifest(work, description, binary_hash, support_hashes, placement)
    write_json(work / "manifest.json", manifest)
    invoke([checker, work, *dimensions], work, "artifact-check")
    # No paths or live tensor identities become semantic identity. Build logs
    # stay in the retained package for diagnostics and are not executable inputs.
    write_json(work / "build-report.json", {"status": "passed", "shape": [args.m, args.d, args.f],
        "compile_and_validation_ms": (time.perf_counter()-start)*1000,
        "input_sha256": digest(source_bytes), "semantic_sha256": semantic,
        "tool_sha256": digest(tool.read_bytes()), "placement": placement,
        "unique_executable_exports": len(executables)})
    publish(work, output)
    print(f"Artifact: {output}", flush=True)
    return output

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("m", "d", "f"):
        parser.add_argument("--" + name, type=int, required=True)
    parser.add_argument("--tool", type=Path, default=ROOT / "build/m3/coral-stablehlo-tool")
    parser.add_argument("--checker", type=Path, default=ROOT / "build/m1/coral-ffn-artifact-check")
    parser.add_argument("--compiler", type=Path, default=ROOT / "repos/coralnpu-compiler/bazel-bin/external/iree_core+/tools/iree-compile")
    compile_region(parser.parse_args())

if __name__ == "__main__":
    main()
