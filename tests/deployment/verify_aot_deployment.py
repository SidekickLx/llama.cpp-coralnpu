#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Prepare a compiler-free root, then verify it with Linux namespaces and chroot."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/ggml"))
sys.path.insert(0, str(ROOT / "tests/runtime"))
from verify_runner import read_run, require
from verify_ffn_members import compare as compare_logits, copy_catalog
from verify_ffn_regions import inspect_regions
from verify_unpatched import require_clean
from ffn_compare import compare as compare_tensors

PACKAGE_FILES = ("manifest.json", "region.vmfb", "region.stablehlo.mlir",
                 "region.metadata.json", "placement.json")
PROGRAMS = ("coral-llama-runner", "coral-ffn-runner",
            "coral-matmul-runner", "coral-aot-environment-probe")

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter_read(stream):
            value.update(block)
    return value.hexdigest()

def iter_read(stream):
    while True:
        block = stream.read(1024 * 1024)
        if not block:
            return
        yield block

def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")

def copy_file(source, destination, executable=False):
    require(source.is_file(), f"missing input: {source}")
    require(not destination.exists(), f"refusing to overwrite: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    destination.chmod(0o555 if executable else 0o444)

def copy_package(source, destination):
    for name in PACKAGE_FILES:
        copy_file(source / name, destination / name)

def inventory(root):
    result = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), f"symlink in deployed root: {path}")
        relative = path.relative_to(root)
        if relative.parts[0] in ("output", "tmp"):
            continue
        if path.is_file():
            result[str(relative)] = {"bytes": path.stat().st_size, "sha256": digest(path),
                                     "mode": stat.S_IMODE(path.stat().st_mode)}
        else:
            require(path.is_dir(), f"special file in deployed root: {path}")
    return result

def prepare(args):
    require(os.geteuid() != 0, "prepare as the ordinary workspace owner, not root")
    revision = require_clean()
    parent = args.output_root.resolve()
    require(parent.is_relative_to(ROOT / "build"), "evidence must stay under this project's build/")
    parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="deployment-", dir=parent))
    print(f"Evidence: {work}", flush=True)
    root = work / "rootfs"
    root.mkdir()
    build = args.build_dir.resolve(strict=True)
    model_evidence = args.model_evidence.resolve(strict=True)
    ffn_evidence = args.ffn_evidence.resolve(strict=True)
    aot_evidence = args.aot_evidence.resolve(strict=True)
    for source in (model_evidence, ffn_evidence, aot_evidence):
        require(json.loads((source / "report.json").read_text())["status"] == "passed",
                f"source evidence has not passed: {source}")
    dependencies = {}
    for name in PROGRAMS:
        program = build / name
        copy_file(program, root / "bin" / name, executable=True)
        result = subprocess.run(["ldd", str(program)], capture_output=True, text=True, check=True)
        (work / f"{name}.ldd.txt").write_text(result.stdout)
        require("not found" not in result.stdout, "unresolved runtime library")
        for line in result.stdout.splitlines():
            # ldd lists the transitive closure, including the ELF interpreter.
            match = re.search(r"(?:=>\s+)?(/[^\s]+)\s+\(", line)
            if not match:
                continue
            source = Path(match.group(1)).resolve(strict=True)
            if "ld-linux" in source.name:
                destination = root / "lib64/ld-linux-x86-64.so.2"
            else:
                destination = root / "lib" / Path(match.group(1)).name
            if destination.exists():
                require(digest(source) == digest(destination), "conflicting library names")
            else:
                copy_file(source, destination, executable=True)
            dependencies[str(destination.relative_to(root))] = str(source)
    catalog = root / "data/catalog"
    catalog.mkdir(parents=True)
    for index in ("profiles.tsv", "pointwise.tsv"):
        copy_catalog(model_evidence / "artifacts", catalog, index)
    copy_file(model_evidence / "artifacts/ffn.tsv", catalog / "ffn.tsv")
    for line in (catalog / "ffn.tsv").read_text().splitlines()[1:]:
        if line.strip() and not line.startswith("#"):
            name = line.split()[-1]
            require(Path(name).name == name and name not in (".", ".."), "invalid package name")
            copy_package(model_evidence / "artifacts" / name, catalog / name)
    # Two independent deployment controls: no FFN profile and corrupt package.
    for name in ("members-only", "corrupt"):
        destination = root / "data" / name
        shutil.copytree(catalog, destination)
    (root / "data/members-only/ffn.tsv").unlink()
    bad_vmfb = root / "data/corrupt/real-prefill/region.vmfb"
    bad_vmfb.chmod(0o644)
    bad_vmfb.write_bytes(b"deliberately corrupt deployment control")
    bad_vmfb.chmod(0o444)
    model = args.model.resolve(strict=True)
    copy_file(model, root / "data/model.gguf")
    copy_file(model_evidence / "cpu/tokens.txt", root / "data/tokens.txt")
    expected = work / "expected"
    expected.mkdir()
    for name in ("cpu", "regions-0"):
        shutil.copytree(model_evidence / name, expected / name)

    packages = {}
    for parent in (aot_evidence, ffn_evidence / "artifacts"):
        for manifest_path in sorted(parent.glob("*/manifest.json")):
            manifest = json.loads(manifest_path.read_text())
            shape = manifest["contract"]["shape"]
            key = (shape["m"], shape["d"], shape["f"])
            packages[key] = manifest_path.parent
    selected = {}
    for path in sorted((ffn_evidence / "cases").glob("*/inputs.json")):
        tensors = json.loads(path.read_text())["inputs"]
        m, d = tensors[0]["shape"]
        f, weight_d = tensors[1]["shape"]
        require(d == weight_d, "incorrect retained FFN input shape")
        selected.setdefault((m, d, f), path)
    require(len(selected) == 9, "expected all nine accepted FFN profiles")
    profiles = []
    for shape, source in sorted(selected.items()):
        m, d, f = shape
        name = f"m{m}-d{d}-f{f}"
        copy_package(packages[shape], root / "data/ffn" / name)
        copy_file(source, root / "data/inputs" / (name + ".json"))
        reference = {}
        for engine in ("cpu", "coral"):
            reference[engine] = json.loads((source.parent / (engine + ".json")).read_text())["outputs"][0]
        write_json(expected / (name + ".json"), reference)
        profiles.append({"name": name, "shape": list(shape), "source_case": source.parent.name})

    optimized_profiles = []
    optimized_model = False
    if args.optimized_model_evidence:
        source = args.optimized_model_evidence.resolve(strict=True)
        require(json.loads((source / "report.json").read_text())["status"] == "passed",
                "optimized model evidence has not passed")
        destination = root / "data/optimized-catalog"
        destination.mkdir()
        for index in ("profiles.tsv", "pointwise.tsv"):
            copy_catalog(source / "artifacts", destination, index)
        copy_file(source / "artifacts/ffn.tsv", destination / "ffn.tsv")
        names = {line.split()[3] for line in (destination / "ffn.tsv").read_text().splitlines()[1:]
                 if line.strip() and not line.startswith("#")}
        for name in names:
            require(Path(name).name == name and name not in (".", ".."), "invalid optimized package")
            copy_package(source / "artifacts" / name, destination / name)
        copy_file(source / "cpu/tokens.txt", root / "data/optimized-tokens.txt")
        shutil.copytree(source / "regions-0", expected / "optimized-model")
        optimized_model = True
    if args.optimized_ffn_evidence:
        source = args.optimized_ffn_evidence.resolve(strict=True)
        data = json.loads((source / "report.json").read_text())
        require(data["status"] == "passed", "optimized FFN evidence has not passed")
        # Both ABIs, both a pure P1 exact core and P2 padded real-model cases.
        for case in data["cases"]:
            m, d, f = case["shape"]
            if (d == 3 and m != 1) or d not in (3, 64):
                continue
            name = "optimized-" + case["case"]
            layout = "transposed" if case["case"].startswith("transposed-") else "native"
            copy_package(source / "cache/entries" / case["key"], root / "data/ffn" / name)
            inputs = source / case["case"] / "inputs.json"
            copy_file(inputs, root / "data/inputs" / (name + ".json"))
            reference = {}
            for engine in ("cpu", "coral"):
                reference[engine] = json.loads((inputs.parent / (engine + ".json")).read_text())["cases"][0]["outputs"][0]
            write_json(expected / (name + ".json"), reference)
            optimized_profiles.append({"name": name, "shape": [m, d, f],
                                       "capacity_m": case["capacity_m"], "layout": layout})

    compiler = ROOT / "repos/coralnpu-compiler/bazel-bin/external/iree_core+/tools/iree-compile"
    host_compilers = [compiler, Path("/usr/bin/clang-19")]
    snapshots = []
    for path in host_compilers:
        require(path.is_file(), f"host compiler control is absent: {path}")
        snapshots.append({"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size})
    absent = {"/bin/sh", "/usr/bin/python3", "/bin/iree-compile", "/usr/bin/iree-compile",
              "/usr/local/bin/iree-compile", "/usr/bin/clang", "/usr/bin/clang-19",
              "/usr/bin/gcc", "/usr/bin/cc", "/usr/bin/bazel",
              str(ROOT / "compiler/compile_ffn.py"), str(ROOT / "compiler/jit_worker.py"),
              str(ROOT / "build/m3/coral-stablehlo-tool")}
    for path in host_compilers:
        absent.add(str(path))
        absent.add(str(path.resolve()))
    (root / "control").mkdir()
    (root / "control/absent-paths.txt").write_text("\n".join(sorted(absent)) + "\n")
    (root / "output").mkdir()
    (root / "tmp").mkdir()
    # Package inputs are read-only. Only two explicit scratch/output dirs are
    # writable in the deployed environment.
    for path in root.rglob("*"):
        path.chmod(0o555 if path.is_dir() or path.stat().st_mode & 0o111 else 0o444)
    manifest = inventory(root)
    write_json(work / "inventory.json", manifest)
    config = {"schema": "coral.aot.deployment.v1", "llama_revision": revision,
              "profiles": profiles, "optimized_profiles": optimized_profiles,
              "optimized_model": optimized_model, "dependencies": dependencies, "host_compilers": snapshots,
              "inventory_sha256": digest(work / "inventory.json"),
              "model_sha256": digest(model), "prepare_uid": os.getuid(), "prepare_gid": os.getgid()}
    write_json(work / "deployment.json", config)
    print(json.dumps({"status": "prepared", "deployment": str(work),
                      "files": len(manifest), "bytes": sum(row["bytes"] for row in manifest.values()),
                      "ffn_profiles": len(profiles)}), flush=True)

def isolated(work, program, arguments, label, success=True, probe=False):
    root = work / "rootfs"
    wrappers = ["/usr/bin/unshare", "/usr/bin/setpriv", "/usr/sbin/chroot"]
    command = ["/usr/bin/strace", "-f", "-s", "4096", "-e", "trace=execve,execveat",
               "-o", str(work / (label + ".exec.trace")),
               wrappers[0], "--mount", "--net", "--pid", "--fork", "--kill-child=SIGKILL",
               wrappers[1], "--no-new-privs",
               wrappers[2], "--userspec=65534:65534", "--groups=65534", str(root), program, *arguments]
    env = {"PATH": "/bin", "LD_LIBRARY_PATH": "/lib", "LANG": "C", "LC_ALL": "C",
           "HOME": "/nonexistent", "TMPDIR": "/tmp"}
    result = subprocess.run(command, env=env, cwd="/", close_fds=True,
                            capture_output=True, text=True, timeout=300, check=False)
    for name, content in (("stdout", result.stdout), ("stderr", result.stderr)):
        (work / f"{label}.{name}.log").write_text(content)
    require((result.returncode == 0) == success,
            f"{label}: exit {result.returncode}; {result.stderr[-2500:]}")
    trace = (work / (label + ".exec.trace")).read_text()
    require("execveat(" not in trace, "unexpected descriptor-based executable launch")
    launches = [line for line in trace.splitlines() if "execve(" in line]
    completed = []
    rejected = []
    for line in launches:
        match = re.search(r'execve\("([^"]+)"', line)
        require(match is not None, "unrecognized executable trace")
        if line.endswith("= 0"):
            completed.append(match.group(1))
        else:
            require(probe and "= -1 ENOENT " in line, "unexpected executable attempt")
            rejected.append(match.group(1))
    require(completed == wrappers + [program], f"unexpected child executable: {completed}")
    return result, {"runtime_execve_count": 1, "wrapper_execve_count": 3,
                    "rejected_execve_count": len(rejected), "exit_code": result.returncode}

def verify(args):
    require(os.geteuid() == 0, "verify needs root only to create namespaces/chroot and drop to UID 65534")
    work = args.deployment.resolve(strict=True)
    require(work.is_relative_to(ROOT / "build") and work.name.startswith("deployment-"),
            "only a prepared deployment under this project's build/ is accepted")
    require(not (work / "report.json").exists(), "completed evidence must not be overwritten")
    require(not (work / "verification-started.json").exists(),
            "this deployment was already attempted; prepare a fresh directory to preserve its evidence")
    config = json.loads((work / "deployment.json").read_text())
    require(config["schema"] == "coral.aot.deployment.v1", "invalid deployment schema")
    require(digest(work / "inventory.json") == config["inventory_sha256"], "inventory changed")
    root = work / "rootfs"
    require(root.is_dir() and not root.is_symlink(), "invalid root directory")
    expected_inventory = json.loads((work / "inventory.json").read_text())
    require(inventory(root) == expected_inventory, "runtime files differ from the prepared inventory")
    require(sorted(path.name for path in (root / "bin").iterdir()) == sorted(PROGRAMS),
            "unexpected executable entry point")
    write_json(work / "verification-started.json",
               {"schema": config["schema"], "inventory_sha256": config["inventory_sha256"]})
    # Chown ONLY the validated staged root. The project and host tools are not
    # modified. Runners never retain root or host capability access.
    os.chown(root, 0, 0)
    for path in root.rglob("*"):
        require(not path.is_symlink(), "symlink appeared in staged root")
        os.chown(path, 0, 0)
    for name in ("output", "tmp"):
        path = root / name
        os.chown(path, 65534, 65534)
        path.chmod(0o755)
    evidence = {}
    result, trace = isolated(work, "/bin/coral-aot-environment-probe",
                             ["/control/absent-paths.txt"], "environment", probe=True)
    environment = json.loads(result.stdout.splitlines()[-1])
    require(environment["status"] == "passed" and
            trace["rejected_execve_count"] == environment["absent_executables"],
            "missing compiler absence controls")
    evidence["environment"] = trace
    model_results = {}
    cpu_reference, reference = read_run(work / "expected/cpu", "cpu")
    _, old_coral = read_run(work / "expected/regions-0", "coral")
    tokens = (root / "data/tokens.txt").read_bytes()
    for mode in ("cpu", "members", "regions", "regions-reload", "greedy", "missing-ffn", "corrupt"):
        numerical = mode != "cpu"
        catalog = "members-only" if mode == "missing-ffn" else "corrupt" if mode == "corrupt" else "catalog"
        options = ["--mode", "coral" if numerical else "cpu", "--model", "/data/model.gguf",
                   "--output", "/output/" + mode]
        if mode != "greedy":
            options.extend(["--tokens", "/data/tokens.txt"])
        if numerical:
            options.extend(["--artifacts", "/data/" + catalog, "--observation", "backend",
                            "--ffn", "off" if mode == "members" else "auto"])
        result, trace = isolated(work, "/bin/coral-llama-runner", options, "model-" + mode,
                                 success=mode != "corrupt")
        evidence["model-" + mode] = trace
        output = root / "output" / mode
        if mode == "corrupt":
            require("region.vmfb: SHA-256 mismatch" in result.stderr,
                    "corrupt deployment failed for an unexpected reason")
            require(not (output / "summary.json").exists(), "corrupt deployment reported success")
            model_results[mode] = {"rejected": True}
            continue
        summary, actual = read_run(output, "coral" if numerical else "cpu")
        require((output / "tokens.txt").read_bytes() == tokens, "model tokens changed")
        error = compare_logits(actual, reference)
        require(actual == (old_coral if numerical else reference), "deployment changed model logits")
        fused = mode in ("regions", "regions-reload", "greedy")
        inspect_regions(output, 20 if fused else 0)
        if numerical:
            require(summary["stats"]["matmuls"] == (84 if fused else 144), "missing matmul execution")
            require(summary["stats"]["calls_completed"] == (104 if fused else 164), "wrong VM count")
        model_results[mode] = {"max_abs_error_vs_cpu": error, "stats": summary["stats"]}
    ffn_results = []
    for profile in config["profiles"] + config.get("optimized_profiles", []):
        name = profile["name"]
        expected = json.loads((work / "expected" / (name + ".json")).read_text())
        for reload in (False, True):
            label = name + ("-reload" if reload else "")
            result, trace = isolated(work, "/bin/coral-ffn-runner",
                ["--artifact", "/data/ffn/" + name, "--inputs", "/data/inputs/" + name + ".json",
                 "--repeat", "1" if reload else "2", "--recreate", "1" if reload else "2",
                 "--output", "/output/" + label + ".json",
                 "--weight-layout", profile.get("layout", "native"),
                 "--capacity-m", str(profile.get("capacity_m", profile["shape"][0]))], "ffn-" + label)
            evidence["ffn-" + label] = trace
            report = json.loads((root / "output" / (label + ".json")).read_text())
            require(report["status"] == "passed" and report["shape"] == profile["shape"], "FFN failed")
            case = report["cases"][0]
            require(len(case["calls"]) == (1 if reload else 4), "incomplete repeated FFN execution")
            for output in case["outputs"]:
                require(output == expected["coral"], "deployment changed standalone FFN output")
                comparison = compare_tensors(output, expected["cpu"])
                require(comparison["status"] == "passed", "deployed FFN differs from CPU")
            for cycle in report["cycles"]:
                stats = cycle["stats"]
                require(stats["calls_completed"] == stats["calls_attempted"] == (1 if reload else 2),
                        "FFN VM call mismatch")
                for field in ("host_executable_load_attempts", "failed_dispatches", "observer_errors"):
                    require(stats[field] == 0, "FFN host/failure counter")
                require(stats["coral_dispatches"] == stats["instruction_samples"] > 0, "missing FFN samples")
            ffn_results.append({"profile": name, "reload": reload, "comparison": comparison,
                                "cycles": report["cycles"]})
    optimized_model_runs = []
    if config.get("optimized_model", False):
        _, expected_logits = read_run(work / "expected/optimized-model", "coral")
        for repeat in range(2):
            label = "optimized-model-" + str(repeat)
            result, trace = isolated(work, "/bin/coral-llama-runner",
                ["--mode", "coral", "--model", "/data/model.gguf",
                 "--artifacts", "/data/optimized-catalog", "--observation", "backend",
                 "--tokens", "/data/optimized-tokens.txt", "--output", "/output/" + label], label)
            evidence[label] = trace
            summary, actual = read_run(root / "output" / label, "coral")
            require(actual == expected_logits, "optimized deployment changed logits")
            error = compare_logits(actual, reference)
            inspect_regions(root / "output" / label, 20, {1: 3, 5: 3})
            require(summary["stats"]["ffn_module_cache_hits"] == 19 and
                    summary["stats"]["ffn_padded_rows"] == 124 and
                    summary["stats"]["jit_requests"] == 0, "optimized deployment did not reuse capacity")
            optimized_model_runs.append({"max_abs_error_vs_cpu": error, "stats": summary["stats"]})

    result, trace = isolated(work, "/bin/coral-matmul-runner",
        ["--module", "/data/catalog/m1_k64_n172.vmfb", "--m", "1", "--k", "64", "--n", "172",
         "--repeat", "3", "--recreate", "2", "--self-test"], "matmul")
    evidence["matmul"] = trace
    matmul = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    checks = [row for row in matmul if "case" in row]
    cycles = [row for row in matmul if "cycle" in row]
    require(len(checks) == 16 and all(row["passed"] and row["mismatches"] == 0 for row in checks),
            "standalone matmul numerical checks failed")
    require(len(cycles) == 2, "standalone matmul context recreation missing")
    for cycle in cycles:
        require(cycle["calls_attempted"] == cycle["calls_completed"] == 8, "matmul VM call mismatch")
        require(cycle["coral_dispatches"] == cycle["instruction_samples"] > 0,
                "matmul instruction evidence missing")
        for field in ("host_executable_load_attempts", "failed_dispatches", "observer_errors"):
            require(cycle[field] == 0, "matmul host/failure counter")
    require(inventory(root) == expected_inventory, "runtime modified an immutable deployment input")
    for item in config["host_compilers"]:
        path = Path(item["path"])
        require(path.is_file() and path.stat().st_size == item["bytes"] and digest(path) == item["sha256"],
                "host compiler changed; caches must be preserved")
    # Git belongs to the ordinary workspace owner; do not alter safe.directory
    # settings merely because namespace setup required root.
    git_args = {"text": True, "user": config["prepare_uid"], "group": config["prepare_gid"],
                "extra_groups": []}
    source = ROOT / "repos/llama.cpp"
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], **git_args).strip()
    status = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], **git_args).strip()
    require(revision == config["llama_revision"] and not status, "llama.cpp source changed")
    report = {"schema": "coral.aot.deployment.acceptance.v1", "status": "passed",
              "compiler_physically_absent_from_runtime_root": True,
              "host_compilers_preserved": True, "llama_revision": config["llama_revision"],
              "model_sha256": config["model_sha256"], "environment": environment,
              "deployed_files": len(expected_inventory),
              "deployed_bytes": sum(row["bytes"] for row in expected_inventory.values()),
              "model_runs": model_results, "ffn_profiles": len(config["profiles"]),
              "ffn_runs": ffn_results, "optimized_model_runs": optimized_model_runs,
              "optimized_ffn_profiles": len(config.get("optimized_profiles", [])),
              "matmul": matmul, "process_evidence": evidence}
    write_json(work / "report.json", report)
    print(json.dumps({"status": "passed", "report": str(work / "report.json"),
                      "environment": environment, "ffn_profiles": len(config["profiles"]),
                      "isolated_processes": len(evidence)}), flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="run as the workspace owner")
    prep.add_argument("--build-dir", type=Path, default=Path("build/m1"))
    prep.add_argument("--model", type=Path, default=Path("models/stories260K-f32.gguf"))
    prep.add_argument("--model-evidence", type=Path, required=True)
    prep.add_argument("--ffn-evidence", type=Path, required=True)
    prep.add_argument("--aot-evidence", type=Path, required=True)
    prep.add_argument("--optimized-model-evidence", type=Path)
    prep.add_argument("--optimized-ffn-evidence", type=Path)
    prep.add_argument("--output-root", type=Path, default=Path("build/m4-deployment-results"))
    check = commands.add_parser("verify", help="root only for namespace/chroot setup; runners use UID 65534")
    check.add_argument("--deployment", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        verify(args)

if __name__ == "__main__":
    main()
