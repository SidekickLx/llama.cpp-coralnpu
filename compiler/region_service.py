#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Shared canonical StableHLO to complete FFN artifact compilation service."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
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

@dataclass(frozen=True)
class Toolchain:
    tool: Path = ROOT / "build/m3/coral-stablehlo-tool"
    checker: Path = ROOT / "build/m1/coral-ffn-artifact-check"
    compiler: Path = ROOT / "repos/coralnpu-compiler/bazel-bin/external/iree_core+/tools/iree-compile"

    def paths(self):
        return {name: getattr(self, name).resolve(strict=True)
                for name in ("tool", "checker", "compiler")}

    def describe(self, profile):
        require(isinstance(profile, (tuple, list)) and len(profile) == 3 and
                all(type(size) is int and size > 0 for size in profile), "invalid FFN profile")
        return json.loads(subprocess.check_output(
            [str(self.paths()["checker"]), "--describe", *map(str, profile)], text=True))

    def support(self):
        root = Path(str(self.paths()["compiler"]) + ".runfiles") / "_main"
        return {
            "linker": root / "toolchain_rv32/bin/riscv32-unknown-elf-ld",
            "linker_script": root / "crt/coralnpu_tcm.ld",
            "iree_archive": root / "crt/libcoralnpu_iree.a",
            "crt_archive": root / "crt/libcoralnpu_crt.a",
        }


def canonicalize_region(source, profile, *, toolchain, evidence):
    """Verify imported StableHLO and remove only nonsemantic metadata."""
    evidence = Path(evidence)
    evidence.mkdir(parents=True, exist_ok=False)
    toolchain.describe(profile)
    (evidence / "source.mlir").write_bytes(source)
    invoke([toolchain.paths()["tool"], "prepare-ffn", evidence / "source.mlir",
            *profile, evidence / "canonical.mlir", evidence / "metadata.json"],
           evidence, "prepare")
    return (evidence / "canonical.mlir").read_bytes()


def compile_region(canonical_ir, profile, target, numeric_policy, *,
                   output, toolchain=None):
    """The only FFN compilation pipeline, shared by AOT and JIT cache misses.

    canonical_ir is normalized standard StableHLO bytes, not a shape template.
    Runtime values, graph identities and output paths are not semantic inputs.
    The current implementation admits the locked F32 FFN ABI/target only.
    """
    toolchain = toolchain or Toolchain()
    description = toolchain.describe(profile)
    require(target == description["compatibility"], "unsupported target/runtime compatibility")
    require(numeric_policy == description["contract"]["numeric_policy"], "unsupported numeric policy")
    require(isinstance(canonical_ir, bytes) and 0 < len(canonical_ir) <= 16 * 1024 * 1024,
            "canonical StableHLO must be bounded nonempty bytes")
    output = Path(output).absolute()
    require(not output.exists() and not output.is_symlink(), "output must be a new directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=output.name + ".building-", dir=output.parent))
    paths = toolchain.paths()
    checker, tool, compiler = paths["checker"], paths["tool"], paths["compiler"]
    dimensions = list(map(str, profile))
    check_sources(description)
    support = toolchain.support()
    support_hashes = {name: digest(path.read_bytes()) for name, path in support.items()}
    binary_hash = digest(compiler.read_bytes())
    tool_hash = digest(tool.read_bytes())
    source_bytes = canonical_ir
    (work / "source.mlir").write_bytes(source_bytes)
    start = time.perf_counter()
    invoke([tool, "prepare-ffn", work / "source.mlir", *dimensions,
            work / "region.stablehlo.mlir", work / "region.metadata.json"], work, "prepare")
    require((work / "region.stablehlo.mlir").read_bytes() == canonical_ir,
            "compile_region requires canonical StableHLO; call canonicalize_region first")
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
    require(digest(tool.read_bytes()) == tool_hash, "frontend changed during build")
    manifest = make_manifest(work, description, binary_hash, support_hashes, placement)
    write_json(work / "manifest.json", manifest)
    invoke([checker, work, *dimensions], work, "artifact-check")
    # No paths or live tensor identities become semantic identity. Build logs
    # stay in the retained package for diagnostics and are not executable inputs.
    write_json(work / "build-report.json", {"status": "passed", "shape": list(profile),
        "compile_and_validation_ms": (time.perf_counter()-start)*1000,
        "input_sha256": digest(source_bytes), "semantic_sha256": semantic,
        "tool_sha256": digest(tool.read_bytes()), "placement": placement,
        "unique_executable_exports": len(executables)})
    publish(work, output)
    return output
