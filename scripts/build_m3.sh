#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
compiler_root="$project_root/repos/coralnpu-compiler"
python3 - "$project_root" <<'PY'
import json
from pathlib import Path
import subprocess
import sys
root = Path(sys.argv[1])
lock = json.loads((root / "dependencies.lock.json").read_text())
checkouts = {
    "coralnpu-compiler": root / "repos/coralnpu-compiler",
    "iree": root / "repos/coralnpu-compiler/third_party/iree",
    "llvm-project": root / "repos/coralnpu-compiler/third_party/llvm-project",
    "stablehlo": root / "repos/coralnpu-compiler/third_party/iree/third_party/stablehlo",
}
for name, path in checkouts.items():
    actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if actual != lock["repositories"][name]["revision"]:
        raise SystemExit(f"{name}: revision differs from dependencies.lock.json")
PY
cd "$compiler_root"
bazel build --config=dev --jobs="${M3_BUILD_JOBS:-4}" \
    --inject_repository="coral_project=$project_root" \
    @coral_project//compiler/ir:coral-stablehlo-tool
binary="$compiler_root/bazel-bin/external/+local_repository+coral_project/compiler/ir/coral-stablehlo-tool"
mkdir -p "$project_root/build/m3"
ln -sfn "$binary" "$project_root/build/m3/coral-stablehlo-tool"
printf 'StableHLO tool: %s\n' "$project_root/build/m3/coral-stablehlo-tool"
