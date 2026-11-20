# SPDX-License-Identifier: MIT
"""Content-addressed FFN packages, with validation and atomic publication."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import time
from region_service import ROOT, Toolchain, compile_region, publish, require
from ffn_artifact import PIPELINE, digest, write_json


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode()


def identity_hash(value):
    return digest(canonical_json(value))


def file_digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            value.update(block)
    return value.hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON field")
        result[key] = value
    return result


def read_json(path, limit=1024 * 1024):
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        require(stat.S_ISREG(info.st_mode) and 0 < info.st_size <= limit,
                "cache metadata must be a bounded regular file")
        data = stream.read(limit + 1)
    require(len(data) <= limit, "cache metadata too large")
    return json.loads(data, object_pairs_hook=unique_object)


def compile_environment(toolchain):
    # Paths are configuration, never key material. Hash the tools actually used,
    # the service implementation and relevant support libraries independently.
    paths = toolchain.paths()
    paths.update(toolchain.support())
    paths["python"] = Path(sys.executable).resolve(strict=True)
    for name in ("region_service.py", "region_cache.py", "ffn_artifact.py", "jit_worker.py"):
        path = ROOT / "compiler" / name
        if path.exists():
            paths[name] = path
    # iree-compile is a small launcher; the implementation lives in
    # libIREECompiler.so. Include the resolved dynamic dependency closure.
    libraries = {}
    for role in ("compiler", "tool", "checker", "linker", "python"):
        linked = subprocess.run(["/usr/bin/ldd", str(paths[role])], capture_output=True,
                                text=True, timeout=30, check=False)
        require(linked.returncode == 0 or "not a dynamic executable" in linked.stderr +
                linked.stdout or "statically linked" in linked.stdout, "cannot inspect tool dependencies")
        require("not found" not in linked.stdout, "unresolved compiler/tool library")
        for line in linked.stdout.splitlines():
            match = re.search(r"(?:=>\s+)?(/[^\s]+)\s+\(", line)
            if not match:
                continue
            path = Path(match.group(1)).resolve(strict=True)
            name = "shared:" + Path(match.group(1)).name
            require(name not in libraries or libraries[name] == path,
                    "conflicting compiler/tool library resolutions")
            libraries[name] = path
    paths.update(libraries)
    description = toolchain.describe((1, 1, 1))
    identity = {"schema": "coral.compile-environment.v1",
                "compatibility": description["compatibility"],
                "compiler_options": description["compiler_options"],
                "python_version": sys.version,
                "files": {name: file_digest(path) for name, path in paths.items()}}
    return identity, {name: str(path) for name, path in paths.items()}


def cache_identity(canonical_ir, profile, target, numeric_policy, environment):
    return {"schema": "coral.ffn.cache.v1", "pipeline": PIPELINE,
            "semantic_sha256": digest(canonical_ir),
            "profile": {"m": profile[0], "d": profile[1], "f": profile[2],
                        "dtype": "f32", "layout": "row-major", "abi": "coral.ffn.f32.v1"},
            "target": target, "numeric_policy": numeric_policy,
            "environment": environment}


class RegionCache:
    def __init__(self, root, toolchain=None):
        self.toolchain = toolchain or Toolchain()
        self.environment, self.paths = compile_environment(self.toolchain)
        self.root = Path(root).absolute()
        self.root.mkdir(parents=True, exist_ok=True)
        require(not self.root.is_symlink(), "cache root must not be a symlink")
        for name in ("entries", "staging", "quarantine", "requests"):
            directory = self.root / name
            directory.mkdir(exist_ok=True)
            require(directory.is_dir() and not directory.is_symlink(), "invalid cache directory")

    @contextmanager
    def writer(self):
        fd = os.open(self.root / ".writer.lock",
                     os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
        try:
            require(stat.S_ISREG(os.fstat(fd).st_mode), "invalid cache lock")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("cache writer busy; concurrent compile deduplication is not supported") from error
            yield
        finally:
            os.close(fd)

    def validate(self, entry, identity, evidence):
        require(entry.is_dir() and not entry.is_symlink(), "invalid cache entry directory")
        require(read_json(entry / "cache.json") == identity, "cache identity mismatch")
        manifest = read_json(entry / "manifest.json")
        require(manifest["program"]["semantic_sha256"] == identity["semantic_sha256"],
                "cached program identity mismatch")
        files = identity["environment"]["files"]
        require(manifest["compiler"]["binary_sha256"] == files["compiler"],
                "cached compiler identity mismatch")
        require(manifest["compiler"]["support_sha256"] ==
                {name: files[name] for name in ("linker", "linker_script", "iree_archive", "crt_archive")},
                "cached compiler support identity mismatch")
        profile = identity["profile"]
        with evidence.open("w") as log:
            result = subprocess.run([str(self.toolchain.checker.resolve(strict=True)), str(entry),
                                     *map(str, (profile["m"], profile["d"], profile["f"]))],
                                    stdout=log, stderr=subprocess.STDOUT, timeout=30, check=False)
        require(result.returncode == 0, "cached artifact rejected; see " + str(evidence))

    def resolve(self, canonical_ir, profile, target, numeric_policy):
        start = time.perf_counter()
        description = self.toolchain.describe(profile)
        require(target == description["compatibility"], "unsupported cache target")
        require(numeric_policy == description["contract"]["numeric_policy"], "unsupported cache policy")
        identity = cache_identity(canonical_ir, profile, target, numeric_policy, self.environment)
        key = identity_hash(identity)
        entry = self.root / "entries" / key
        work = Path(tempfile.mkdtemp(prefix="lookup-", dir=self.root / "staging"))
        recovered = False
        if entry.exists() or entry.is_symlink():
            try:
                self.validate(entry, identity, work / "hit-check.log")
                return self.result(entry, key, "hit", False, start, 0)
            except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
                (work / "invalid-entry.txt").write_text(str(error) + "\n")
        with self.writer():
            if entry.exists() or entry.is_symlink():
                try:
                    self.validate(entry, identity, work / "locked-check.log")
                    return self.result(entry, key, "hit", False, start, 0)
                except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
                    # Preserve the exact failed entry; never delete build evidence.
                    quarantine = self.root / "quarantine" / (key + "-" + work.name)
                    publish(entry, quarantine)
                    (work / "quarantine.txt").write_text(str(error) + "\n")
                    recovered = True
            compile_start = time.perf_counter()
            try:
                package = compile_region(canonical_ir, profile, target, numeric_policy,
                                         output=work / "package", toolchain=self.toolchain)
                # Recheck every pinned executable/service/support file before
                # publication. A changed environment cannot populate an old key.
                current, _ = compile_environment(self.toolchain)
                require(current == self.environment, "compile environment changed during build")
                write_json(package / "cache.json", identity)
                self.validate(package, identity, work / "publish-check.log")
                publish(package, entry)
            except Exception as error:
                write_json(work / "failure.json", {"status": "failed", "key": key,
                                                  "error": str(error)})
                raise
        return self.result(entry, key, "compiled", recovered, start,
                           (time.perf_counter() - compile_start) * 1000)

    @staticmethod
    def result(entry, key, state, recovered, start, compile_ms):
        return {"key": key, "artifact": str(entry), "state": state, "recovered": recovered,
                "elapsed_ms": (time.perf_counter() - start) * 1000, "compile_ms": compile_ms}
