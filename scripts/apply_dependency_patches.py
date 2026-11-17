#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Apply pinned, hash-checked project patches without resetting local changes."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def git(repo, *args, check=True):
    return subprocess.run(["git", "-C", str(repo), *args], check=check,
                          capture_output=True, text=True)

def main():
    lock = json.loads((ROOT / "dependencies.lock.json").read_text())
    for name, patches in lock["project_patches"].items():
        repo = ROOT / "repos" / name
        revision = git(repo, "rev-parse", "HEAD").stdout.strip()
        if revision != lock["repositories"][name]["revision"]:
            raise SystemExit(f"{name}: checkout differs from the lock file")
        for entry in patches:
            patch = ROOT / entry["path"]
            if hashlib.sha256(patch.read_bytes()).hexdigest() != entry["sha256"]:
                raise SystemExit(f"patch checksum mismatch: {entry['path']}")
            if git(repo, "apply", "--reverse", "--check", str(patch), check=False).returncode == 0:
                print(f"already applied: {entry['path']}")
                continue
            result = git(repo, "apply", "--check", str(patch), check=False)
            if result.returncode:
                raise SystemExit(result.stderr)
            git(repo, "apply", str(patch))
            print(f"applied: {entry['path']}")

if __name__ == "__main__":
    main()
