# M4 final gate: AOT deployment without compiler entry points

Date: 2026-11-18. **Passed.**
Implementation commit: `76096f5` on `codex/m4-ffn-aot`.

AOT programs load and execute in a separate Linux filesystem containing no
compiler executable, compiler frontend, shell, Python interpreter or build tool.
This completes the planned M4 AOT software acceptance scope together with the
[artifact/runtime](m4-step1-3-report.md), [standalone three-way](m4-runner-test-report.md)
and [GGML region](m4-ggml-region-test-report.md) results. It is not a versioned
release or acceptance of SoC, quantized or production-scale model support.

The development compiler and all build caches remain on the host. The runtime
root contains actual copies of the required deployment files and cannot reach
those host files. Compiler absence is therefore a filesystem property of the
tested runtime environment, independently of executable search paths.

## Evidence

The final retained directory is:

`build/m4-deployment-results/deployment-n99yeemm/`

It contains:

- `report.json`: structured acceptance, all counters and numerical comparisons.
- `inventory.json`: every immutable deployed file, size, mode and SHA-256.
- `deployment.json`: profile sources, dependency closure and host compiler hashes.
- `environment.stdout.log`: privilege, network and compiler absence checks.
- `environment.exec.trace`: actual failed compiler/tool launch attempts.
- `model-regions.exec.trace` and other `*.exec.trace`: every executable launch.
- `rootfs/output/`: actual model traces, logits, tokens and standalone FFN JSON.
- `expected/`: retained CPU and previous Coral results used for comparison.

Source inputs are the accepted model evidence `model-69qio4fj`, standalone
evidence `three-way-p3m7_t6e` and AOT evidence `aot-cvoknmn_` under their
respective `build/m4-*` directories. The first complete passing attempt,
`deployment-vi50_0vi`, is also preserved. The final run adds explicit rejection
reason and standalone matmul counter assertions.

## Runtime environment

Preparation runs as the normal workspace owner. It copies three existing
runners, the small test probe, the ELF dependency closure, the model, AOT
packages and fixture inputs. The tested root has **150 immutable files,
94,684,149 bytes (90.30 MiB)**. This includes duplicate control packages,
nine FFN profiles and unstripped executables; it is not a minimal shipping size.

Only `bin/coral-llama-runner`, `bin/coral-ffn-runner`,
`bin/coral-matmul-runner` and `bin/coral-aot-environment-probe` are installed
as application entry points. Runtime libraries include MPACT and the standard
C/C++ loader/libraries. `LD_LIBRARY_PATH=/lib` selects the copied libraries;
the binaries' original build RUNPATHs are absent in this filesystem.
Normalized StableHLO files are retained as integrity-checked artifact metadata;
there is no compiler available to translate them at inference time.

Each invocation starts in fresh mount, network and PID namespaces and enters
the staged root with `chroot`. The outer launcher requires root for that setup.
The actual program runs as **UID/GID 65534**, with `no_new_privs` enabled and
zero effective, permitted and inheritable capabilities. The probe confirms:

~~~json
{"status":"passed","uid":65534,"pid":1,"capabilities":0,
 "no_new_privs":true,"active_network_interfaces":0,
 "absent_executables":15}
~~~

There are no host directory bindings, no `/proc`, `/sys`, `/home`, `/mnt`
or `/root` paths, and no active network interface. Open file descriptors are
closed by the launcher except standard input/output/error. Only dedicated
`/output` and `/tmp` directories are writable by the runtime user.
The immutable file inventory is verified before and after execution.

## Compiler absence controls

For every control path, the probe performs both `stat` and a real `execv`
attempt. **All 15 return ENOENT (2)**. Controls include:

- The pinned `iree-compile` path and its resolved Bazel output path.
- `/bin/iree-compile`, `/usr/bin/iree-compile` and `/usr/local/bin/iree-compile`.
- Clang, its resolved LLVM path, GCC, CC and Bazel system entry points.
- The project's FFN compilation frontend and StableHLO compiler tool.
- A shell and Python interpreter.

Example actual probe record:

~~~json
{"kind":"absent_executable","path":"/usr/bin/clang-19",
 "stat_errno":2,"exec_errno":2}
~~~

Each inference trace contains three **host setup** launches (`unshare`,
`setpriv` and `chroot`), followed by exactly one runtime program launch.
There is no further `execve` or `execveat` from inference. Failed launch
attempts occur only in the dedicated environment probe.
The namespace setup tools, Python orchestrator and strace live outside the
runtime root and are not reachable by the deployed program.

The original host IREE compiler and Clang still exist with exactly the same
sizes and SHA-256 hashes after verification. No compiler or build cache was
deleted, renamed or rebuilt for this gate.

## Whole-model results

The model remains `stories260K-f32.gguf`: five layers, D=64, F=172,
512-token vocabulary. Runs cover five prefill tokens and three decode steps.
The locked llama.cpp revision remains clean:
`9fba51d2c9a1b5cb5f0c47986c031334e2d7663a`.

| Isolated run | Independent matmul calls | Whole FFN calls | Total VM calls | Coral dispatches |
| --- | ---: | ---: | ---: | ---: |
| CPU baseline | 0 | 0 | 0 | 0 |
| Member execution | 144 | 0 | 164 | 276 |
| Complete regions | 84 | 20 | 104 | 256 |
| Complete regions, new process | 84 | 20 | 104 | 256 |
| Complete regions, greedy token selection | 84 | 20 | 104 | 256 |
| No FFN catalog, member fallback | 144 | 0 | 164 | 276 |

Every successful Coral run is **bit-identical to the corresponding previous
development-environment Coral logits**. The isolated CPU baseline is
bit-identical to the previous CPU baseline. All Coral runs have maximum absolute
error **8.58306884765625e-6** against CPU, within `atol=1e-4, rtol=1e-4`.

The short greedy run does not read a token replay file. It selects its own
decode inputs, reproduces the CPU token sequence and produces identical logits
to the fixed-token Coral runs. This is a short generation control, not broad
language-quality or long-context acceptance.

Each complete model region run records 80 bypassed original member kernels,
256 instruction samples, 1,504,628 kernel instructions, five FFN weight-set
preparations and 15 cache hits. Host executable loads, failed dispatches and
observer errors are all zero.

A deliberately corrupt complete-region VMFB fails with
`FFN artifact: region.vmfb: SHA-256 mismatch` and produces no success summary.
No compiler is started to repair or replace it. Removing the FFN catalog
instead gives correct member execution using the pre-existing member AOTs.

## Independent runtime results

All nine accepted static FFN profiles load inside the compiler-free root:

| M | D | F |
| ---: | ---: | ---: |
| 1 | 3 | 4 |
| 2 | 3 | 4 |
| 5 | 3 | 4 |
| 3 | 7 | 11 |
| 1 | 64 | 172 |
| 5 | 64 | 172 |
| 1 | 128 | 256 |
| 16 | 128 | 256 |
| 32 | 128 | 256 |

Each profile performs two calls in each of two recreated contexts, followed
by one load/call in another process: **45 calls, 180 Coral dispatches,
1,870,260 kernel instructions** in total.

All standalone outputs are bit-identical to their prior Coral results.
Maximum absolute error versus the saved independent CPU outputs is
`0.000152587890625`; the maximum error/tolerance ratio is `0.0782490534`,
so every element meets the absolute-plus-relative tolerance. All dispatches
have instruction samples, with zero host loads, dispatch failures or observer
errors.

The separate matmul runner tests M=1, K=64, N=172 with changed activations,
alternate/restored weights, zero inputs and rejected invalid bindings.
Across two device contexts, **16 calls and 32 dispatches** pass with zero
numerical error on these fixtures and no host numerical execution.

The full gate consists of **27 isolated process invocations**: one environment
probe, seven model invocations (including the expected failure), eighteen
standalone FFN invocations and one matmul invocation.

## Reproduce

Prerequisites: the existing Linux x86_64 build and accepted artifact/evidence
directories; `ldd`, `strace`, `unshare`, `setpriv` and `chroot`.
The helper uses the Python standard library. No container image or download
is needed. Build the small test-only probe before preparing the runtime root:

~~~bash
source env.sh
cmake --build build/m1 --target \
  coral-aot-environment-probe coral-llama-runner coral-ffn-runner coral-matmul-runner -j4

python tests/deployment/verify_aot_deployment.py prepare \
  --model-evidence build/m4-region-model-results/model-69qio4fj \
  --ffn-evidence build/m4-runner-results/three-way-p3m7_t6e \
  --aot-evidence build/m4-aot-results/aot-cvoknmn_
~~~

Use the **new directory printed by prepare**, then run:

~~~bash
sudo /usr/bin/python3 tests/deployment/verify_aot_deployment.py verify \
  --deployment build/m4-deployment-results/deployment-NEW_DIRECTORY
~~~

Root is used only by the outside orchestrator to establish the namespace/chroot
and assign staged ownership; every probe/runner drops to UID 65534.
On this WSL host the same verifier was launched through `wsl.exe -u root`.
The script only accepts prepared roots under this project's `build/` tree,
verifies the inventory, and rejects repeated verification of the same directory.
Prepare again for a new attempt; old evidence and outputs remain intact.

## Scope and timing

This gate proves that the deployed programs require **runtime libraries and
AOT artifacts, not compiler entry points**. It validates fresh-process loading,
device recreation, model routing, numerical results and failure behavior.
The root is a test deployment fixture, not a general installer or a hardened
sandbox for arbitrary untrusted programs.

Execution here is traced, so its durations are not used as a performance
benchmark. The previous [paired model measurements](m4-ggml-region-test-report.md)
remain the timing evidence: fewer outer calls were established, but no latency
improvement was measured. MPACT time/instruction counts do not represent
physical Coral silicon or a complete SoC.
