# llama.cpp-coralnpu

An experimental **CoralNPU backend for llama.cpp / GGML**, using the
**IREE/MLIR compiler stack** and the **MPACT simulator**. The project explores
how model operations become device programs, how a persistent runtime executes
them, and how to prove where the computation happened.

The long-term goal is to run llama.cpp on a simulated SoC with a CoralNPU
accelerator. Today, llama.cpp runs on the Linux host and offloads supported
matrix multiplications to MPACT through IREE.

**Status: unreleased research prototype.** M1 (standalone runtime) and M2
(GGML integration and a small model) have passed their current functional
acceptance tests. These are development milestones. The first `v0.1.0` release
is reserved for SoC integration; completing M2 does not constitute that release.

[Quick start](#quick-start) · [Tests](#tests) ·
[Repository layout](#repository-layout) · [Development](#development) ·
[Project plan](docs/project-plan.md) · [M3 contracts](docs/m3-contracts.md)

## Features

- **GGML integration:** an out-of-tree Coral backend for supported, contiguous
  two-dimensional F32 `MUL_MAT` operations. The scheduler keeps unsupported
  operations on the CPU.
- **Ahead-of-time compilation:** static shapes are exported to StableHLO and
  compiled to IREE VM bytecode modules (`.vmfb`). Weights remain runtime inputs.
- **Persistent C++ runtime:** VM/device lifetime management, tensor ABI checks,
  module reuse, and an opt-in immutable model-weight cache.
- **A real model path:** llama.cpp loads GGUF, tokenizes the prompt, builds
  the graph, and performs short prefill and decode runs.
- **Execution evidence:** per-node placement, successful Coral dispatches,
  simulator kernel instruction counts, and module/weight-cache statistics.
- **M3 frontend foundations:** original DAG and actual split snapshots, official
  StableHLO typed ops/verifier, and the upstream CPU reference interpreter.
- **Numerical and failure tests:** fixed-token CPU/Coral comparison, adapter
  capability boundaries, mixed graphs, mutable weights, and error propagation.

Current limits: Linux x86_64, single-threaded diagnostic inference, synchronous
execution, and a validated tiny F32 model. Attention and other unsupported
operations still run on the CPU. FFN subgraph compilation, JIT, W8A8/W4A8,
concurrent requests, and SoC/Linux-driver integration remain future work.
There is no standalone installable backend package or general chat CLI yet.

M3 steps 1–3 provide the [FFN contracts](docs/m3-contracts.md),
[real GGML graph export](docs/m3-graph-export.md), and
[typed StableHLO construction, verification and reference evaluation](docs/m3-stablehlo.md).
Standalone matmul/FFN fixtures pass numerical tests. Importing real GGML regions,
region planning, and fused execution remain future work.

## How it works

~~~text
Preparation:
  CPU run -> observed matmul shapes -> StableHLO .mlir -> Coral compiler -> .vmfb

Inference:
  GGUF -> llama.cpp -> GGML scheduler
                        |-- unsupported operations -> CPU
                        +-- supported F32 matmul -> Coral adapter
                                                 -> IREE VM / HAL
                                                 -> MPACT CoralNPU simulation
~~~

GGUF contains model metadata and tensors. It is not translated wholesale into
StableHLO here. The exporter emits one matmul program per static shape, with
the contract `Y[M,N] = X[M,K] × transpose(W[N,K])`. The compiler runs during
preparation; inference only loads existing VMFBs.

A missing shape is ineligible for Coral and stays on the CPU. Once an operation
is accepted for Coral execution, an invalid artifact or runtime failure is
reported as an error; it is not silently rerun on the CPU.

## Requirements

The validated environment is **Ubuntu 22.04, x86_64, on WSL2**. The build
requires Linux x86_64; the package commands below target Ubuntu 22.04.

| Tool | Version used |
| --- | --- |
| Clang / LLD | 19.1.7 / major version 19 |
| Bazel | 9.1.0 |
| Python | 3.13.18 |
| CMake | 3.31.10; project minimum 3.26 |
| Ninja | 1.13.2 |
| uv, for local Python setup | 0.13.8 |

The first compiler build can take hours. As a practical starting point, provide
16 GiB of memory to Linux/WSL, use four build jobs, and allow about 80 GiB of
free SSD space. These are planning recommendations, not measured minimums;
the retained development workspace is already about 50 GiB. Keep WSL sources
and caches in its Linux filesystem, such as `~/projects`.

[dependencies.lock.json](dependencies.lock.json) pins source revisions, the
model, and target settings. This walkthrough needs the compiler and llama.cpp
checkouts. IREE, LLVM, CoralNPU, and MPACT come through the compiler's submodules
and Bazel dependencies. Separate Renode and MPACT checkouts are not needed.

## Quick start

Follow these steps in order for a **fresh checkout**, in the same Bash terminal.
If you already have a working environment, use
[Incremental builds](#incremental-builds) instead of recreating it.

### 1. Open Linux and clone the project

On Windows, install Ubuntu if needed from administrator PowerShell:

~~~powershell
wsl --install -d Ubuntu-22.04
~~~

Complete the distribution setup and restart if requested, then open it:

~~~powershell
wsl -d Ubuntu-22.04
~~~

See Microsoft's [WSL installation guide](https://learn.microsoft.com/en-us/windows/wsl/install)
for prerequisites. **All remaining commands run in Bash inside Linux/WSL.**

~~~bash
sudo apt update
sudo apt install -y git
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/SidekickLx/llama.cpp-coralnpu.git
cd llama.cpp-coralnpu

set -euo pipefail
export PROJECT_ROOT="$PWD"
mkdir -p downloads tools/bin repos models build
~~~

The SSH clone URL is `git@github.com:SidekickLx/llama.cpp-coralnpu.git`.

### 2. Install system prerequisites

The pinned upstream toolchain expects Clang 19 in `/usr/bin`. Ubuntu 22.04's
default CMake and Python packages alone are insufficient for this workflow.

~~~bash
sudo apt update
sudo apt install -y \
    build-essential git curl wget ca-certificates gnupg \
    lsb-release software-properties-common python3 python3-venv \
    pkg-config unzip zip zlib1g-dev

curl -fL https://apt.llvm.org/llvm.sh -o downloads/llvm.sh
sudo bash downloads/llvm.sh 19

/usr/bin/clang-19 --version
/usr/bin/ld.lld-19 --version
~~~

The installer from the [LLVM package repository](https://apt.llvm.org/) adds
the appropriate package source and installs the requested toolchain.

### 3. Install local Python and build tools

Use [uv](https://docs.astral.sh/uv/guides/install-python/) to install Python
inside the workspace, then install CMake and Ninja in a virtual environment.
The exporter and validation scripts use Python's standard library; PyTorch,
TensorFlow, JAX, and NumPy are not required for this walkthrough.

~~~bash
python3 -m venv tools/uv-bootstrap
tools/uv-bootstrap/bin/python -m pip install 'uv==0.13.8'

export UV_PYTHON_INSTALL_DIR="$PROJECT_ROOT/tools/python"
tools/uv-bootstrap/bin/uv python install 3.13.18
tools/uv-bootstrap/bin/uv venv --python 3.13.18 --seed .venv
source .venv/bin/activate
python -m pip install 'cmake==3.31.10' 'ninja==1.13.2'

curl -fL \
    https://github.com/bazelbuild/bazel/releases/download/9.1.0/bazel-9.1.0-linux-x86_64 \
    -o downloads/bazel-9.1.0-linux-x86_64
(
    cd downloads
    echo 'a667454f3f4f8878df8199136b82c199f6ada8477b337fae3b1ef854f01e4e2f  bazel-9.1.0-linux-x86_64' \
        | sha256sum -c -
)
install -m 755 downloads/bazel-9.1.0-linux-x86_64 tools/bin/bazel-9.1.0
~~~

The binary/checksum come from the official
[Bazel 9.1.0 release](https://github.com/bazelbuild/bazel/releases/tag/9.1.0).
Create local configuration so later shells reuse the same tools and cache:

~~~bash
test ! -e env.sh
test ! -e bazel.local.rc
test ! -e tools/bin/bazel

cat > env.sh <<'EOF'
export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CORAL_WORKSPACE="$PROJECT_ROOT"
export UV_PYTHON_INSTALL_DIR="$PROJECT_ROOT/tools/python"
source "$PROJECT_ROOT/.venv/bin/activate"
export PATH="$PROJECT_ROOT/tools/bin:$PATH"
export CC=/usr/bin/clang-19
export CXX=/usr/bin/clang++-19
export CORALNPU_SIMULATOR=mpact
export CMAKE_BUILD_PARALLEL_LEVEL=4
EOF

cat > bazel.local.rc <<EOF
startup --output_user_root=$PROJECT_ROOT/.cache/bazel
build --jobs=4
build --local_resources=cpu=4
build --local_resources=memory=10000
test --test_output=errors
EOF

cat > tools/bin/bazel <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "$workspace/tools/bin/bazel-9.1.0" \
    "--bazelrc=$workspace/bazel.local.rc" "$@"
EOF
chmod +x tools/bin/bazel

source ./env.sh
python --version
cmake --version
ninja --version
bazel --version
~~~

These configuration files are ignored by Git. Use a checkout path without
spaces for the generated Bazel configuration. If memory is limited, lower
`--jobs` and the memory budget in `bazel.local.rc` before the first build.

### 4. Fetch the pinned dependencies

This creates two checkouts at the exact commits in the lock file. It stops if
either destination already exists.

~~~bash
python - <<'PY'
import json
from pathlib import Path
import subprocess

lock = json.loads(Path("dependencies.lock.json").read_text())
for name in ("coralnpu-compiler", "llama.cpp"):
    target = Path("repos") / name
    if target.exists():
        raise SystemExit(f"Existing checkout: {target}; inspect it before continuing.")
    dependency = lock["repositories"][name]
    subprocess.run(["git", "init", str(target)], check=True)
    subprocess.run(
        ["git", "-C", str(target), "remote", "add", "origin", dependency["url"]],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(target), "fetch", "--depth=1", "origin", dependency["revision"]],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(target), "checkout", "--detach", "FETCH_HEAD"],
        check=True,
    )
PY

export CORAL_SOURCE="$PROJECT_ROOT/repos/coralnpu-compiler"
(
    cd "$CORAL_SOURCE"
    git submodule update --init --depth=1 -j 4 -- .
    git -C third_party/iree submodule update --init --depth=1 -j 4 \
        -- . ':(exclude)third_party/llvm-project'
    ./scripts/patch-third_party.sh all
)
~~~

The official patches are necessary. Review their output: a fresh checkout
should apply them successfully. The script reports both already-applied patches
and conflicts as `SKIPPED`; investigate unexpected skips before building.
Patched submodules appearing modified is expected. See the
[pinned upstream instructions](https://github.com/google-coral/coralnpu-compiler/blob/770dc9332dffed8ff91d8cface7e9d176227d912/README.md)
for dependency details.

Apply this project's hash-checked scheduler observation patch after fetching
the pinned checkouts. This command is safe to repeat and never resets local changes.

~~~bash
python scripts/apply_dependency_patches.py
~~~

### 5. Build the compiler, simulator, and runners

First build the compiler and MPACT shared library. This is the expensive step;
subsequent builds reuse its cache.

~~~bash
(
    cd "$CORAL_SOURCE"
    bazel build --config=dev \
        @iree_core//tools:iree-compile \
        @coralnpu_mpact//sim/hw_sim:libcoralnpu_simulator_mpact.so
)
MPACT_LIB="$(
    cd "$CORAL_SOURCE"
    bazel info --config=dev bazel-bin
)/external/coralnpu_mpact+/sim/hw_sim/libcoralnpu_simulator_mpact.so"
test -f "$MPACT_LIB"

cmake -S . -B build/m1 -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_C_COMPILER=clang-19 \
    -DCMAKE_CXX_COMPILER=clang++-19 \
    -DCORALNPU_MPACT_SIMULATOR_LIB="$MPACT_LIB" \
    -DCORAL_BUILD_M2=ON \
    -DBUILD_TESTING=ON

cmake --build build/m1 --target \
    coral-llama-runner coral-matmul-runner coral-adapter-test -j 4
~~~

The `build/m1` name comes from the standalone runtime milestone; it also contains
M2. CMake imports the IREE runtime and llama.cpp sources. It requires an existing
MPACT library and does not launch a simulator build during configuration.

### 6. Download the first model and run the CPU baseline

**stories260K-f32** is a tiny Llama-architecture model with five layers, hidden
size 64, and a 512-token vocabulary. Its GGUF is about 1.13 MiB. It is useful
for integration tests, rather than general conversation.

Download the pinned revision and verify its checksum:

~~~bash
python - <<'PY'
import hashlib
import json
from pathlib import Path
import urllib.request

model = json.loads(Path("dependencies.lock.json").read_text())["model"]
target = Path("models") / model["file"]
target.parent.mkdir(exist_ok=True)

def verified(path):
    payload = path.read_bytes()
    return len(payload) == model["bytes"] and hashlib.sha256(payload).hexdigest() == model["sha256"]

if target.exists():
    if not verified(target):
        raise SystemExit(f"Checksum mismatch: {target}; preserve and inspect this file.")
else:
    url = f'https://huggingface.co/{model["repo"]}/resolve/{model["revision"]}/{model["file"]}'
    partial = target.with_suffix(target.suffix + ".partial")
    if partial.exists():
        raise SystemExit(f"Existing partial download: {partial}; inspect it before retrying.")
    with urllib.request.urlopen(url) as response, partial.open("xb") as output:
        output.write(response.read())
    if not verified(partial):
        raise SystemExit(f"Downloaded model failed checksum verification: {partial}")
    partial.rename(target)
print(f"Verified {target}: {model['sha256']}")
PY

export DEMO_DIR="$(mktemp -d "$PROJECT_ROOT/build/demo-XXXXXX")"
printf 'Demo directory: %s\n' "$DEMO_DIR"

./build/m1/coral-llama-runner \
    --mode cpu \
    --model models/stories260K-f32.gguf \
    --prompt "Once upon a time" \
    --decode 3 \
    --output "$DEMO_DIR/cpu"

python -m json.tool "$DEMO_DIR/cpu/summary.json"
~~~

Expect `"status": "completed"`, `"mode": "cpu"`, and
`"coral_registered": false`. This diagnostic runner writes tokens, logits, and
logs rather than printing a generated story. It performs one prefill and three
decode steps, continuing for the requested step count even if it selects EOS.

The CPU run records compatible matrix shapes in `profiles.tsv`. There are nine
for this model and prompt. They still need to be compiled for the Coral run.

### 7. Compile the observed shapes to VMFB

Define a shell helper for strict F32 compilation, also used by the tests below.
The default target allows Zvt and uses the pinned 8 KiB ITCM / 32 KiB DTCM profile.

~~~bash
compile_matmul() {
    local m="$1" k="$2" n="$3" stem="$4"

    python "$PROJECT_ROOT/compiler/export_matmul.py" \
        --m "$m" --k "$k" --n "$n" --output "$stem.mlir"
    (
        cd "$CORAL_SOURCE"
        bazel run --config=dev @iree_core//tools:iree-compile -- \
            "$stem.mlir" \
            --iree-input-type=stablehlo \
            --iree-execution-model=async-internal \
            --iree-hal-target-device=local \
            --iree-hal-local-target-device-backends=llvm-cpu \
            --iree-llvmcpu-target-cpu=host \
            --iree-hal-target-device=coralnpu \
            --coralnpu-fast-math=false \
            --coralnpu-roofline-speedup-threshold=0 \
            --coralnpu-dump-affinity-profile-format=pretty \
            -o "$stem.vmfb"
    ) > "$stem.compile.log" 2>&1
}

export ARTIFACT_DIR="$DEMO_DIR/artifacts"
mkdir "$ARTIFACT_DIR"

while read -r m k n filename; do
    compile_matmul "$m" "$k" "$n" "$ARTIFACT_DIR/${filename%.vmfb}"
done < <(tail -n +2 "$DEMO_DIR/cpu/profiles.tsv")

# Publish the index only after every compilation succeeds.
cp "$DEMO_DIR/cpu/profiles.tsv" "$ARTIFACT_DIR/profiles.tsv"
~~~

Each shape produces `.mlir`, `.vmfb`, and a `.compile.log` with compilation and
placement information. Inspect the log if compilation fails. The local target
supports VM/HAL orchestration; the runtime rejects host executable loading to
catch unintended compute placement. Actual Coral execution is checked at runtime.

Weights are passed to these programs by the adapter. There is no GGUF-to-MLIR
translation or compiler invocation inside inference.

### 8. Run the same token sequence through Coral

~~~bash
./build/m1/coral-llama-runner \
    --mode coral \
    --model models/stories260K-f32.gguf \
    --tokens "$DEMO_DIR/cpu/tokens.txt" \
    --artifacts "$ARTIFACT_DIR" \
    --output "$DEMO_DIR/coral"

python -m json.tool "$DEMO_DIR/coral/summary.json"
~~~

`--tokens` replays the exact CPU prefill and decode inputs. It cannot be
combined with `--prompt` or `--decode`. Comparisons therefore use the same
sequence even if an argmax decision differs.

The recorded successful run for the pinned setup has:

| Evidence | Recorded value |
| --- | ---: |
| Logits | 4 rows × 512 vocabulary entries |
| Completed Coral matmuls | 144 |
| Coral dispatches | 256 |
| Kernel instructions | 1,447,460 |
| IREE host executable load attempts | 0 |
| Failed dispatches / observer errors | 0 |
| Maximum absolute CPU/Coral logit error | `7.152557373046875e-6` |

The runner requires actual Coral dispatches at every inference step. Unsupported
operations still execute on the CPU; zero IREE host loads does not mean the
whole model ran on Coral. See the [M2 report](docs/m2-test-report.md) for the
node-level evidence.

### 9. Check correctness automatically

~~~bash
python tests/ggml/verify_runner.py \
    --runner build/m1/coral-llama-runner \
    --model models/stories260K-f32.gguf \
    --artifacts "$ARTIFACT_DIR"
~~~

This launches fresh CPU, CPU-replay, and Coral processes, compares all 2,048
logits, checks dispatch/cache evidence, and exercises six failure cases.
The fixed tolerance is `abs(actual-reference) <= 1e-4 + 1e-4*abs(reference)`.
Inference subprocesses have no compiler or shell in `PATH`.

The script prints its new `build/m2-results/runner-*` directory and writes
`verification.json` on success. Keep that report with the logs.

## Outputs and reuse

| File | Purpose |
| --- | --- |
| `summary.json` | Completion marker, devices, dimensions, and cumulative statistics |
| `tokens.txt` | Exact prefill and decode inputs for replay |
| `logits.f32` | Little-endian F32 logits, one row per inference step |
| `steps.jsonl` | Per-step argmax, timing, placement counts, and runtime statistics |
| `trace.jsonl` | Tensor metadata, node placement, dispatches, and instructions |
| `profiles.tsv` | Static matmul shapes and expected VMFB filenames |

Output directories must be new. The example uses `mktemp` to preserve earlier
runs. Keep `DEMO_DIR` and `ARTIFACT_DIR` for subsequent tests. In another terminal,
source `env.sh` and set those variables to your earlier run's actual directories.

A different prompt or model may introduce new shapes. Repeat the CPU-profile
and AOT steps in a new directory before the Coral run. Current acceptance
evidence covers the pinned stories260K model and short sequence only.

## Tests

### Model integration

Quick start step 9 is the model regression suite. Omitting `--artifacts` runs
only CPU and basic error-path checks; that does not validate Coral execution.
See the [runner guide](docs/m2-runner.md) for the full contract.

### Runtime and adapter boundaries

In the same terminal as Quick start, reuse `compile_matmul` to prepare three
small runtime fixtures and register them with CTest:

~~~bash
export TEST_ARTIFACTS="$(mktemp -d "$PROJECT_ROOT/build/test-artifacts-XXXXXX")"
for m in 1 16 32; do
    compile_matmul "$m" 16 8 "$TEST_ARTIFACTS/m${m}_zvt"
done

cmake -S . -B build/m1 \
    -DM1_VMFB_M1="$TEST_ARTIFACTS/m1_zvt.vmfb" \
    -DM1_VMFB_M16="$TEST_ARTIFACTS/m16_zvt.vmfb" \
    -DM1_VMFB_M32="$TEST_ARTIFACTS/m32_zvt.vmfb"
ctest --test-dir build/m1 --output-on-failure
~~~

Expect four passing tests: three M1 shapes and `m2_adapter_boundaries`.
The adapter suite runs 23 isolated cases with 95 checks covering capability
queries, CPU routing, mixed graphs, mutable/immutable weights, cache reuse,
and failure propagation. CTest registers artifact-dependent tests only after
you supply the VMFB paths.

To run the adapter suite directly:

~~~bash
python tests/ggml/verify_adapter.py \
    --runner build/m1/coral-adapter-test \
    --m1-vmfb "$TEST_ARTIFACTS/m1_zvt.vmfb" \
    --m16-vmfb "$TEST_ARTIFACTS/m16_zvt.vmfb"
~~~

See the [adapter acceptance report](docs/m2-adapter-validation.md) for coverage.
The broader M1 validation also compares Zvt/RVV and checks file I/O and
rejection paths; the [M1 guide](docs/m1-runtime.md) describes its extra fixtures.

### Performance measurements

After correctness passes, measure repeated CPU/Coral runs on the same inputs:

~~~bash
python tests/ggml/benchmark_runner.py \
    --runner build/m1/coral-llama-runner \
    --model models/stories260K-f32.gguf \
    --tokens "$DEMO_DIR/cpu/tokens.txt" \
    --artifacts "$ARTIFACT_DIR" \
    --warmup-pairs 2 \
    --repeats 20
~~~

Results go into a new `build/m2-benchmarks/paired-*` directory. The
[recorded report](docs/m2-test-report.md) measured median warm decode times of
0.782 ms/token on CPU and 25.972 ms/token through MPACT. These are timings of a
traced software simulator on one host, **not hardware NPU throughput**.
Logical upload/download counts also do not measure all internal DDR/TCM
traffic or peak device memory.

## Repository layout

| Path | Responsibility |
| --- | --- |
| [`src/runtime/`](src/runtime/) | M1 C++ runtime, VM/HAL and simulator ownership, tensor ABI, standalone runner |
| [`src/ggml/`](src/ggml/) | Backend capability/routing logic, artifact and weight-cache integration, model runner |
| [`src/support/`](src/support/) | Shared JSON output helpers |
| [`src/contracts/`](src/contracts/) | M3 FFN ABI, shape, numerical policy, and tensor-role metadata |
| [`src/planner/`](src/planner/) | M3 candidate/plan/rejection data contracts; planner implementation follows |
| [`compiler/export_matmul.py`](compiler/export_matmul.py) | Export static F32 matmul shapes to StableHLO text |
| [`tests/runtime/`](tests/runtime/) | Standalone runtime validation across compiled profiles |
| [`tests/ggml/`](tests/ggml/) | Adapter boundaries, model comparisons, and performance measurements |
| [`docs/`](docs/) | Plan, environment baseline, run guides, and validation reports |
| [`dependencies.lock.json`](dependencies.lock.json) | Dependency revisions, patch hashes, model checksum, and target profile |
| [`CMakeLists.txt`](CMakeLists.txt) | Runtime build, optional M2 integration, and CTest registration |

Local-only paths are excluded from Git:

| Path | Local contents |
| --- | --- |
| `repos/` | Downloaded upstream source checkouts |
| `tools/`, `.venv/` | Installed tools and Python environments |
| `downloads/`, `models/` | Downloaded installers and GGUF assets |
| `.cache/` | Dependency build cache |
| `build/` | Runners, VMFBs, compiler logs, inference outputs, and test reports |
| `.local/`, `env.sh`, `bazel.local.rc` | Machine-specific records and configuration |

## Development

### Where to start reading

1. Read [the project plan](docs/project-plan.md) and the
   [M3 FFN contract](docs/m3-contracts.md) for architecture, interfaces, and
   the ordered implementation/acceptance steps.
2. Follow [`compiler/export_matmul.py`](compiler/export_matmul.py) to see the
   tensor ABI in a small StableHLO module.
3. Read [`src/runtime/matmul_runtime.h`](src/runtime/matmul_runtime.h) and
   [its implementation](src/runtime/matmul_runtime.cc) for loading, binding,
   invocation, and dispatch observation.
4. Read [`src/ggml/ggml_coral.cc`](src/ggml/ggml_coral.cc) for operation support
   and backend integration, then
   [`src/ggml/llama_runner.cc`](src/ggml/llama_runner.cc) for the model entry point.
5. Extend the relevant test in `tests/runtime/` or `tests/ggml/` alongside
   a behavior change.

### Incremental builds

From an existing checkout with a configured `build/m1` directory:

~~~bash
source ./env.sh
cmake --build build/m1 --target \
    coral-llama-runner coral-matmul-runner coral-adapter-test -j 4
~~~

Keep build outputs and dependency caches. Reuse the same toolchain and Bazel
configuration between shells. Project C++ changes normally need only the build
above; exporter, target-flag, or compiler changes require matching VMFBs.

### Troubleshooting

| Symptom | What to check |
| --- | --- |
| Clang 19 is missing | Step 2 must provide `/usr/bin/clang-19` and LLD 19 |
| CMake cannot find the simulator | Build MPACT and pass its existing `.so` with `CORALNPU_MPACT_SIMULATOR_LIB` |
| Compiler build is slow or runs out of memory | Preserve the cache, lower job/resource limits, and resume the same build |
| Compilation fails | Read the shape's `.compile.log`; confirm pinned sources and patches |
| Runner refuses an output directory | Select a new directory; existing results are preserved |
| No Coral offload occurs | Check dtype/layout/precision and whether shapes have compiled entries in `profiles.tsv` |
| VMFB ABI or load failure | Use artifacts matching these shapes and pinned dependencies; inspect compiler/runtime logs |
| CTest finds fewer than four tests | Enable M2/testing, build the test executable, and configure all three `M1_VMFB_M*` paths |
| Another terminal cannot find tools | Source `env.sh` before building or running preparation steps |

### Contributing and roadmap

See [CONTRIBUTING.md](CONTRIBUTING.md). Include numerical results and actual
execution evidence for backend changes. Keep machine-specific paths,
credentials, models, upstream checkouts, and generated binaries out of commits.

Future work includes GGML-to-StableHLO FFN import and compilation, standard
IR verification/reference evaluation, AOT/JIT artifact management,
W8A8 and packed W4A8 device execution, and SoC/Linux-driver integration.
The [project plan](docs/project-plan.md) records the scope. Release numbers
will be assigned separately from development milestones.

## License

Original work is licensed under the [MIT License](LICENSE). Referenced and
downloaded upstream projects and models retain their respective licenses;
this repository does not relicense third-party materials.

M3 development: [interface and semantic contract](docs/m3-contracts.md) and
[real GGML graph export](docs/m3-graph-export.md), plus
[StableHLO construction, verification and reference evaluation](docs/m3-stablehlo.md).

[GGML importer interfaces and tests](docs/m3-importer.md).
