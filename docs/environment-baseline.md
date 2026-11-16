# 环境验证基线

验证日期：2026-11-15。此文件保存脱敏后的历史结果摘要，不包含主机路径、用户名、硬件标识或原始日志。

## 工具版本

Linux x86_64 / Ubuntu 22.04（WSL2）；Clang/LLVM/LLD 19.1.7；Bazel 9.1.0；Python 3.13.18；CMake 3.31.10；Ninja 1.13.2；JAX/jaxlib 0.11.1；NumPy 2.4.6；ml-dtypes 0.5.4；Renode 1.17.0。

源码 revision 和模型 SHA256 见 [`dependencies.lock.json`](../dependencies.lock.json)。IREE、LLVM 和 StableHLO 使用 compiler 配套的官方补丁。

## 已验证的范围

- llama.cpp CPU 版本使用 `stories260K-f32.gguf`，提示词 `Once upon a time`，完成 16 token 生成。
- 官方 16×16 matmul 的 Zvt 与 RVV-only 配置均与 NumPy 参考结果一致。
- 8 组动态输入 F32 检查显式关闭 fast math，使用默认 8 KiB ITCM / 32 KiB DTCM。每组均有 Coral placement、成功的 MPACT dispatch 和非零设备 kernel 指令数。
- Renode 官方 ARM Linux 启动、Coral 应用执行、ramdisk 复制加载及 intrinsic 示例：4/4 通过。

| 用例（M、K、N） | 最大绝对误差 | kernel 指令数 | host 数值 dispatch |
| --- | ---: | ---: | ---: |
| `m1_k16_n8_zvt` | 0 | 117 | 0 |
| `m1_k16_n8_rvv` | 0 | 117 | 0 |
| `m16_k16_n8_zvt` | 0 | 228 | 0 |
| `m16_k16_n8_rvv` | 0 | 1060 | 0 |
| `m32_k16_n8_zvt` | 0 | 353 | 0 |
| `m32_k16_n8_rvv` | 0 | 1984 | 0 |
| `m3_k16_n8_zero_zvt` | 0 | 208 | 0 |
| `m3_k16_n8_zero_rvv` | 0 | 315 | 0 |

严格 F32 检查使用 NumPy 参考、`rtol=1e-5` 和 `atol=1e-6`。混合目标 VMFB 的运行器同时注册 local-task 与 coralnpu 设备，并检查 host 数值 dispatch 为 0。

这些结果验证工具链与小规模运算，不代表 llama.cpp 的 Coral backend 已实现，也不代表真实硬件性能。原始构建和验证产物已清理；未来代码变更需重新执行对应测试。
