# Llamacpp-coralNPU

An experimental CoralNPU backend for llama.cpp / GGML. The project uses the IREE/MLIR compiler stack and the MPACT simulator to explore matrix multiplication, feed-forward networks (FFNs), compiled artifact management, and device execution.

## Current status

The **M1 standalone C++ runtime** has been built and validated on MPACT. All six configurations across three matrix shapes and the Zvt/RVV profiles passed; see the [M1 validation report](docs/m1-validation.md). Device-internal staging measurements and complete allocation statistics remain to be implemented.

**M2** adds a GGML Coral adapter and a model runner. Short prefill and decode runs with the stories260K F32 model passed CPU/Coral numerical comparisons. See the [M2 runner guide](docs/m2-runner.md) and [M2 test and performance report](docs/m2-test-report.md). Dedicated synthetic graph tests for adapter edge cases are still pending. FFN subgraph compilation, JIT, W4A8, and the Linux driver remain future work. The [project plan](docs/project-plan.md) defines the full scope and acceptance criteria.

The initial environment validation covered CPU model inference, the upstream matmul example, strict F32 execution on MPACT, and Renode system examples. Results are summarized in the [environment baseline](docs/environment-baseline.md). Simulator wall-clock measurements do not represent hardware NPU performance.

Local build caches, executables, and test artifacts are retained and excluded from Git. Reuse the existing build directories for incremental builds; do not clean or delete the caches.

## Repository layout

- [docs/project-plan.md](docs/project-plan.md): Architecture, milestones, and acceptance criteria.
- [docs/environment-baseline.md](docs/environment-baseline.md): Tool versions and historical validation results.
- [dependencies.lock.json](dependencies.lock.json): Public upstream URLs, pinned revisions, patches, and model checksums.
- [CONTRIBUTING.md](CONTRIBUTING.md): Contribution guidelines.
- [src/runtime/](src/runtime/): M1 C++ runtime adapter and standalone matmul runner.
- [src/ggml/](src/ggml/): M2 GGML Coral adapter and model runner.
- [docs/m2-runner.md](docs/m2-runner.md): Model execution, profile compilation, and CPU/Coral comparison procedures.
- [docs/m2-test-report.md](docs/m2-test-report.md): Model tests, execution evidence, and repeated performance measurements.
- [tests/ggml/verify_runner.py](tests/ggml/verify_runner.py): Fixed-token replay, model numerical checks, and failure-path validation.
- [compiler/export_matmul.py](compiler/export_matmul.py): Static-shape F32 matmul export to StableHLO.
- [docs/m1-runtime.md](docs/m1-runtime.md): Tensor ABI, resource lifetimes, metric definitions, and build/validation instructions.
- [docs/m1-validation.md](docs/m1-validation.md): M1 validation results and limitations.
- [tests/runtime/verify_m1.py](tests/runtime/verify_m1.py): Validation of six precompiled configurations, file I/O, and rejection cases.
- [CMakeLists.txt](CMakeLists.txt): Runtime build entry point and optional CTest registration.

Local upstream checkouts live in `repos/`; local tools and Python environments live in `tools/` and `.venv/`. These directories are excluded by `.gitignore` and are not part of the published repository.

## Getting started

1. Set up a Linux x86_64 environment; WSL2 is supported on Windows. Use the tool versions listed in the environment baseline.
2. Clone and check out the revisions pinned in `dependencies.lock.json`. Use `coralnpu-compiler` as the dependency root, initialize its matching submodules, and apply the official patches shipped with that revision.
3. Follow the [pinned upstream build instructions](https://github.com/google-coral/coralnpu-compiler/blob/770dc9332dffed8ff91d8cface7e9d176227d912/README.md) to build the compiler, runtime, and matmul examples. Use `--config=dev` and choose build parallelism appropriate for the host.
4. Validate the upstream Zvt/RVV matmul examples first. Explicitly set `--coralnpu-fast-math=false` for strict F32 checks, then continue with the backend milestones in the project plan.

See the [M1 runner guide](docs/m1-runtime.md) for build instructions, execution commands, and remaining work. This repository does not yet provide a standalone installable GGML backend package.

## License

Original work in this project is licensed under the [MIT License](LICENSE). Referenced or locally downloaded upstream projects and models retain their own licenses; this repository does not relicense those third-party materials.
