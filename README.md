# Llamacpp-coralNPU

面向 llama.cpp / GGML 的 CoralNPU 后端实验项目，使用 IREE/MLIR 编译链与 MPACT 模拟器研究矩阵乘、FFN、编译产物管理及设备执行。

## 当前状态

M1 独立 C++ runtime 已完成编译和 MPACT 运行验证：三种 shape 的 Zvt/RVV 共六组配置通过，详见 [M1 验证记录](docs/m1-validation.md)。设备内部 staging 与完整分配统计仍待补充。GGML Coral backend、完整模型 NPU 推理、FFN/JIT/W4A8 和 Linux driver 尚待实现。详细范围见[工程实践计划](docs/project-plan.md)。

已完成一次 CPU 模型、官方 matmul、严格 F32/MPACT 和 Renode 系统示例验证，摘要见[环境基线](docs/environment-baseline.md)。当前 M1 构建缓存、可执行文件及本机测试产物保留并由 Git 忽略；后续使用同一构建目录增量编译，不清理缓存。

## 目录

- `docs/project-plan.md`：架构、阶段和验收标准。
- `docs/environment-baseline.md`：工具版本与历史验证摘要。
- `dependencies.lock.json`：公开上游地址、固定 revision、补丁及模型校验值。
- `CONTRIBUTING.md`：贡献约定。
- src/runtime/：M1 C++ runtime 适配层与命令行 runner。
- compiler/export_matmul.py：导出固定 shape 的 F32 StableHLO matmul。
- docs/m1-runtime.md：tensor ABI、生命周期、统计口径与构建验证步骤。
- docs/m1-validation.md：本次 M1 实际验证结果和限制。
- tests/runtime/verify_m1.py：运行六组已编译产物、文件 I/O 与拒绝用例。
- CMakeLists.txt：runtime 构建入口和可选 CTest 注册。

`repos/` 用于本机上游源码副本，`tools/` 和 `.venv/` 用于本机工具环境；这些目录由 `.gitignore` 排除，不属于本仓库的发布内容。

## 开发起点

1. 准备 Linux x86_64 环境；Windows 可使用 WSL2。工具版本参考环境基线。
2. 根据 `dependencies.lock.json` 克隆并检出固定版本。以 `coralnpu-compiler` 为依赖根，初始化其配套子模块并应用该版本自带的官方补丁。
3. 按[固定版本的官方构建说明](https://github.com/google-coral/coralnpu-compiler/blob/770dc9332dffed8ff91d8cface7e9d176227d912/README.md)构建编译器、运行时及 matmul 示例。使用 `--config=dev`，按本机资源选择并行度。
4. 先验证官方 Zvt/RVV matmul；严格 F32 检查显式设置 `--coralnpu-fast-math=false`。之后按工程计划实现后端。

M1 的构建、运行方式及剩余任务见 [M1 runner 说明](docs/m1-runtime.md)。此仓库目前不提供独立可安装的 GGML 后端。

## 许可证

本项目原创内容采用 [MIT](LICENSE) 许可证。引用或在本机下载的上游项目和模型保留各自的许可证；本仓库不重新授权这些第三方内容。
