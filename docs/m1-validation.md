# M1 运行验证记录

验证日期：2026-11-16。C++ runtime 源码提交：06ea5a2。
环境为 Ubuntu 22.04 / WSL2 / Linux x86_64，使用 MPACT；依赖 revision 见
[dependencies.lock.json](../dependencies.lock.json)。工具版本见[环境基线](environment-baseline.md)。

## 结果

CMake 配置、C++ 编译和链接成功；StableHLO 经固定版本 Coral/IREE 编译器生成 VMFB，
由本项目独立 runner 加载并执行。没有修改上游依赖源码来通过本次验证。

[verify_m1.py](../tests/runtime/verify_m1.py) 的最终运行通过：

- 六组配置，每组 19 次数值调用，共 114 次、152 次 Coral dispatch。
- 每组包含两轮 context 创建/释放的自测（每轮 8 次调用），以及 3 次随机 F32 文件输入输出验算。
- 所有 dispatch 成功，全部有有效指令计数，总计 86,906 条 kernel 指令。
- 全部正例的 host executable 加载尝试、dispatch 失败和 observer 错误均为 0。
- 11 组拒绝用例通过；整个套件启动 23 个 runner 进程。
- CTest 的 M=1、16、32 三个 Zvt 配置入口均通过。

固定 K=16、N=8，数学语义为 Y = X × transpose(W)。关闭 Coral fast math。
下表只统计六组正例；每次调用的指令数为该调用全部 Coral kernel 的合计。

| M | 配置 | 数值调用 | Coral dispatch | 指令数/调用 | 最大绝对误差 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Zvt 允许 | 19 | 38 | 283 | 2.38418579e-7 |
| 16 | Zvt 允许 | 19 | 19 | 228 | 3.57627869e-7 |
| 32 | Zvt 允许 | 19 | 19 | 353 | 4.76837158e-7 |
| 1 | RVV-only | 19 | 38 | 283 | 2.38418579e-7 |
| 16 | RVV-only | 19 | 19 | 1187 | 2.38418579e-7 |
| 32 | RVV-only | 19 | 19 | 2240 | 4.76837158e-7 |

确定性自测的最大误差均为 0。随机文件输入为含正负值的 F32，runner 使用 double
累加后转 F32 的参考；验证脚本另外使用 Python math.fsum 独立核对原始输出文件。
全部满足 abs(error) <= 1e-6 + 1e-5*abs(reference)，没有 NaN/Inf。
Zvt 允许不等于每个 shape 都使用矩阵指令；M=1 的两套配置具有相同指令计数。

## 行为与拒绝检查

自测确认重复调用、修改 activation、同 shape 换权重、恢复旧权重和零 activation。
修改权重源 vector 不会改变 PreparedWeight 的快照。错误 stride 和元素数在设备执行前
被拒绝，拒绝后仍可继续合法调用。每轮自测只显式分配两个权重 buffer 和一个
activation buffer；后续 activation 更新复用已有 buffer。逻辑上传和回读字节数与
输入规模、调用数一致。

独立反例覆盖错误 shape、缺失导出函数、损坏/缺失 VMFB、零维度、未配对的输入文件、
错误文件大小、NaN 输入、已有输出文件、缺失 ABI 标记和 host-only 数值模块。
已有输出文件的内容保持不变。

host-only VMFB 在 VM context 初始化时触发拒绝型 loader：
host_executable_load_attempts=1，calls_attempted=0，calls_completed=0。
因此该反例确认 host 数值 executable 确实被拒绝，而不只是正例没有触发 host 路径。

## 复现与产物保留

完整编译、产物生成与运行命令见 [M1 runner 说明](m1-runtime.md)。
生成 host-only 反例需要显式指定已安装的 ld.lld-19：IREE 的自动查找不识别该版本后缀。
这不需要重建编译器，也不影响已经生成的 Coral VMFB。

本机保留以下内容，全部由 Git 忽略：

- build/m1/：CMake/Ninja 构建树、runner、运行时库和 CTest 结果。
- build/m1-artifacts/：三个正常 StableHLO 输入、缺失标记的输入和八个 VMFB。
- build/m1-results/verification-*/：每轮验证的输入输出、stdout/stderr、result.json。
- build/m1-logs/：依赖构建、VMFB 编译、runner 构建和验证日志。
- .cache/bazel/ 及依赖目录中的 Bazel 输出：编译器和 MPACT 缓存。

正常重复构建 runner 已确认输出 ninja: no work to do。
后续 VMFB 编译时，Bazel 也确认 iree-compile 为 up-to-date。
不要删除这些目录或执行 clean；修改源码后在同一目录增量构建。

## 边界与剩余工作

本次证明独立 runtime 的小规模严格 F32 运算闭环，尚未接入 GGML 或完整模型。
指令计数来自模拟器，墙钟时间不是硬件性能测量。
当前 upload_bytes/download_bytes 只表示 host 与 HAL buffer 的逻辑复制，
binding_allocations 只统计适配层显式分配。MPACT 内部 DDR/TCM staging、完整 allocator
观测仍待实现，因此 M1 计划中的全部统计任务尚未完成。
