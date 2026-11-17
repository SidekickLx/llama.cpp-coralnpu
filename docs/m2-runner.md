# M2 模型 runner

入口是 `src/ggml/llama_runner.cc`。它通过 llama.cpp 加载 GGUF、创建推理上下文，
再由 GGML scheduler 调用 `ggml-coral`。runner 不手动拆分 Transformer，不在推理时调用编译器。
本阶段用于检查数值和执行位置，不是性能 benchmark。

adapter 已单独提交为 `fa601d7`。runner、验证脚本和本说明是后续改动。

## 两种运行模式

- `--mode cpu`：进程仅注册 CPU device，不注册或初始化 Coral 后端。
- `--mode coral`：在创建模型和 context 前注册 Coral ACCEL device，
  读取已有的 `profiles.tsv` 和 VMFB；不支持的算子由 GGML 留在 CPU。

runner 检查实际注册的 device，拒绝额外的 GPU/ACCEL，避免把其他后端的执行误记为 CPU。
它不调用插件扫描入口。CPU 模式仍使用同一个可执行文件，文件中包含 adapter 的链接代码。

两种模式都使用单线程、关闭 Flash Attention 和权重 repacking、F32 KV cache。
模型只读并保持到 context 释放之后，因此可显式开启 adapter 的不可变权重缓存。
当前不提供 LoRA、权重更新、并发请求或会话复用接口。

## 增量构建

以下在项目根目录的 WSL shell 中执行，前提是已按 [M1 说明](m1-runtime.md)
配置 `build/m1` 并生成 MPACT 动态库：

~~~bash
source ./env.sh
cmake -S . -B build/m1 -DCORAL_BUILD_M2=ON
cmake --build build/m1 --target coral-llama-runner -j 4
~~~

后续只修改 C++ 时直接运行最后一条命令。不要删除构建目录或执行 clean。

## CPU 基线与 profile 导出

~~~bash
./build/m1/coral-llama-runner \
    --mode cpu \
    --model models/stories260K-f32.gguf \
    --prompt "Once upon a time" \
    --decode 3 \
    --output build/m2-results/cpu-demo
~~~

输出目录必须不存在。prompt 包括 special tokens 在内限 1–32 个 token，
一次 prefill 后执行 2–4 次单 token decode，默认 3 次。
每次保存最后一个位置的完整词表 logits，并采用 argmax 选择下一次输入。
这是固定步数诊断，即使选到 EOS 也继续到指定步数。

`profiles.tsv` 从实际执行的 GGML 节点中收集兼容矩阵尺寸，不代表这些尺寸已有 VMFB。
首行为 `coral-m2-v1`，后续各列是 M、K、N、VMFB 文件名，例如：

~~~text
coral-m2-v1
1    64    32    m1_k64_n32.vmfb
5    64    32    m5_k64_n32.vmfb
~~~

数学语义为 `Y[M,N] = X[M,K] × transpose(W[N,K])`。
相同 shape 的多个模型权重共享代码模块，但权重快照按各自的存储身份缓存。

## 提前编译实际 profile

沿用 M1 导出器和 strict F32 参数。下面使用新产物目录，保留全部旧产物。
Bazel 复用已有编译器；此步骤在推理进程之外进行。

~~~bash
set -euo pipefail
source ./env.sh
PROJECT_ROOT="$PWD"
PROFILE_FILE="$PROJECT_ROOT/build/m2-results/cpu-demo/profiles.tsv"
ARTIFACT_DIR="$PROJECT_ROOT/build/m2-artifacts/model-demo"
CORAL_SOURCE="$PROJECT_ROOT/repos/coralnpu-compiler"

mkdir -p "$(dirname "$ARTIFACT_DIR")"
mkdir "$ARTIFACT_DIR"

while read -r m k n filename; do
    mlir="$ARTIFACT_DIR/${filename%.vmfb}.mlir"
    python compiler/export_matmul.py --m "$m" --k "$k" --n "$n" --output "$mlir"
    (
        cd "$CORAL_SOURCE"
        bazel run --config=dev @iree_core//tools:iree-compile -- \
            "$mlir" \
            --iree-input-type=stablehlo \
            --iree-execution-model=async-internal \
            --iree-hal-target-device=local \
            --iree-hal-local-target-device-backends=llvm-cpu \
            --iree-llvmcpu-target-cpu=host \
            --iree-hal-target-device=coralnpu \
            --coralnpu-fast-math=false \
            --coralnpu-roofline-speedup-threshold=0 \
            --coralnpu-dump-affinity-profile-format=pretty \
            -o "$ARTIFACT_DIR/$filename"
    ) > "$ARTIFACT_DIR/${filename%.vmfb}.compile.log" 2>&1
done < <(tail -n +2 "$PROFILE_FILE")

# Publish the index only after every compilation succeeds.
cp "$PROFILE_FILE" "$ARTIFACT_DIR/profiles.tsv"
~~~

默认使用允许 Zvt 的 target 和工具链默认内存配置。
编译日志保存静态 placement；运行时另行验证实际 dispatch。
生成产物应和固定依赖版本配套，当前 manifest 尚不是完整的跨工具链兼容性协议。

## 固定 token 的 Coral 对照

~~~bash
./build/m1/coral-llama-runner \
    --mode coral \
    --model models/stories260K-f32.gguf \
    --tokens build/m2-results/cpu-demo/tokens.txt \
    --artifacts build/m2-artifacts/model-demo \
    --output build/m2-results/coral-demo
~~~

`--tokens` 同时提供 prefill 和后续 decode 的输入，不能与 `--prompt` 或 `--decode` 同用。
即使两条路径的 argmax 不同，重放也始终使用 CPU 保存的输入，保证 logits 比较的是同一序列。
token 文件不绑定模型身份，对照时必须使用相同 GGUF；固定模型的 SHA256 见
`dependencies.lock.json`。

Coral 模式要求每次 prefill/decode 都有实际 NPU dispatch，否则返回非零状态。
已有 VMFB 损坏、ABI 不匹配、runtime 或观察回调失败也会终止，不会静默重跑 CPU。

## 输出文件与统计口径

| 文件 | 内容 |
| --- | --- |
| `summary.json` | 成功完成标记、模式、设备、词表和 logits 尺寸、累计统计；失败运行不会写成功标记 |
| `tokens.txt` | 实际送入模型的 token：版本行、prefill/decode 数量、所有输入 ID |
| `logits.f32` | little-endian F32，形状 `[1 + decode_steps, vocab_size]`，第一行来自 prefill |
| `steps.jsonl` | 每步的位置、argmax、CPU/Coral 节点数和 runtime 统计增量 |
| `trace.jsonl` | 节点类型、shape、stride、精度参数、buffer ID、执行位置及 Coral dispatch/指令数 |
| `profiles.tsv` | 实际观察到的兼容矩阵尺寸集合，用于下一轮 AOT 准备 |

Coral 节点必须匹配 adapter 成功调用后返回的 dispatch 事件。
其余数值节点记为 CPU，依据是当前进程已确认只有 CPU 和可选的 CORAL 两种 device。
view/reshape 等无计算节点不计入数值执行日志。

逐节点 eval callback 会限制 scheduler 合批，日志中的时间只用于功能诊断。
buffer ID/大小是本次运行观察到的 GGML buffer，不是完整分配调用记录或峰值内存。
logical upload/download 仍沿用 M1 的 host↔HAL 统计，不等于模拟器内部 DDR/TCM 流量；
权重快照复用也不等于权重常驻 NPU。

## 验证

~~~bash
python tests/ggml/verify_runner.py \
    --runner build/m1/coral-llama-runner \
    --model models/stories260K-f32.gguf \
    --artifacts build/m2-artifacts/model-demo
~~~

不提供 `--artifacts` 时仅验证 CPU 重放和基础错误路径。验证脚本：

- 在独立进程中运行 CPU 和 Coral，推理时将 PATH 设为空工具目录；
- 核对固定 token CPU 重放的 logits 完全一致；
- 对全部 CPU/Coral logits 使用预先固定的容差：
  `abs(actual-reference) <= 1e-4 + 1e-4*abs(reference)`；
- 对账逐节点 trace 与每步 dispatch/指令数；检查 CPU 保留算子；
- 检查第二、第三次 decode 不重新加载模块，且有权重缓存命中；
- 检查已有结果目录、损坏 token 文件、词表外 token、缺失产物、完全无 offload 和损坏 VMFB 的失败路径。

每轮验证在 `build/m2-results/runner-*` 新建目录，保存 stdout/stderr 和验证报告。
这覆盖模型 runner 的集成行为；adapter 对各类 dtype、stride、广播、精度参数及
可变权重的专门合成图回归仍是 M2 后续工作。

## 本次集成验证（2026-11-16）

固定 stories260K-f32 GGUF 的 SHA256 已与依赖锁核对。默认 prompt 产生 5 个 token，
导出 9 种 shape；严格 F32、允许 Zvt、默认内存配置下，9 个编译 placement 报告的
数值 dispatch 均落在 coralnpu。没有修改上游源码。

| 阶段 | Coral matmul | Coral dispatch | kernel 指令数 | 模块加载 | 权重缓存命中 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prefill，5 tokens | 36 | 40 | 573551 | 7 | 0 |
| Decode 1 | 36 | 72 | 291303 | 2 | 36 |
| Decode 2 | 36 | 72 | 291303 | 0 | 36 |
| Decode 3 | 36 | 72 | 291303 | 0 | 36 |

合计 144 次矩阵乘、256 次 dispatch，全部有模拟器指令计数。
每步另有 80 个数值节点在 CPU，其中 10 个是未被 adapter 接收的矩阵乘。
初次 prefill 准备 36 份模型权重，后续 108 次调用均命中权重缓存。

CPU 固定 token 重放的 logits 完全一致；Coral 与 CPU 的 2048 个 logits
最大绝对误差为 7.152557373046875e-6，满足预设容差。
四步 argmax 均一致，依次为 432、383、286、261。
正例的 host executable 加载尝试、失败 dispatch 和 observer 错误均为 0。
推理验证进程的 PATH 中没有编译器或 shell。

这些结果说明小模型 runner 的调用链可用，不代表已覆盖 M2 的全部 shape/layout/precision
拒绝组合，也不代表硬件性能或完整设备内存统计已验收。
