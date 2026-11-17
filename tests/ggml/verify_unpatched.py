#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify source identity and M2/importer acceptance without a llama.cpp patch."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def require_clean():
    lock = json.loads((ROOT / "dependencies.lock.json").read_text())
    source = ROOT / "repos/llama.cpp"
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"], text=True).strip()
    if revision != lock["repositories"]["llama.cpp"]["revision"] or status:
        raise RuntimeError("acceptance requires the clean, locked llama.cpp checkout")
    return revision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("build-dir", "check-library", "model", "artifacts", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    revision = require_clean()
    build = args.build_dir.resolve()
    cache = (build / "CMakeCache.txt").read_text()
    source = (ROOT / "repos/llama.cpp").resolve()
    if f"LLAMA_SOURCE_DIR:PATH={source}" not in cache:
        raise RuntimeError("build directory must use the clean, locked llama.cpp checkout")
    runner = build / "coral-llama-runner"
    help_text = subprocess.check_output([str(runner), "--help"], text=True)
    if "--export-graph" in help_text:
        raise RuntimeError("runner still advertises the removed graph observer")
    args.output_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="unpatched-", dir=args.output_root.resolve()))
    suites = [
        ("m2", ROOT / "tests/ggml/verify_runner.py",
         ["--runner", runner, "--model", args.model.resolve(),
          "--artifacts", args.artifacts.resolve(), "--output-root", work / "m2"]),
        ("importer", ROOT / "tests/compiler/verify_importer.py",
         ["--runner", build / "coral-ggml-import-test",
          "--check-library", args.check_library.resolve(), "--output-root", work / "importer"]),
        ("model-importer", ROOT / "tests/compiler/verify_model_import.py",
         ["--runner", build / "coral-model-import-test",
          "--check-library", args.check_library.resolve(), "--model", args.model.resolve(),
          "--output-root", work / "model-importer"]),
    ]
    for label, script, options in suites:
        result = subprocess.run(
            [sys.executable, str(script), *map(str, options)],
            capture_output=True, text=True, timeout=900, check=False)
        (work / (label + ".stdout.txt")).write_text(result.stdout)
        (work / (label + ".stderr.txt")).write_text(result.stderr)
        if result.returncode:
            raise RuntimeError(f"{label} failed; retained evidence: {work}")
        print(f"{label}: passed", flush=True)
    require_clean()
    report = {"status": "passed", "llama_revision": revision,
              "llama_worktree_clean": True, "suites": [item[0] for item in suites]}
    (work / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({**report, "evidence": str(work)}))


if __name__ == "__main__":
    main()
