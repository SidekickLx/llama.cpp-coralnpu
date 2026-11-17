# M2 adapter 独立验收记录

日期：2026-11-16。范围：固定版本 GGML + Coral adapter + MPACT，Linux x86_64 / WSL2。

**结果：23 个独立测试进程、95 项检查全部通过；与已有模型闭环测试合起来，满足当前 M2 / v0.1 的验收范围。**
生产 adapter/runtime 无需修改。runner、模型测试和原性能报告已提交为 `34aecf0`；
本轮新增独立回归程序、CTest 注册与验收证据。

## 测试方法

- [C++ 测试程序](../tests/ggml/adapter_test.cc) 直接构建小型 GGML 图，注册 CPU 和 CORAL。
  合法图通过真实 scheduler 自动分配，不手动指定节点后端。
- [Python 执行入口](../tests/ggml/verify_adapter.py) 为每个用例启动独立进程，
  保留 stdout/stderr，并在全部通过后写入 `verification.json`。
- 使用已有的 M1 strict F32 VMFB：M=1/16、K=16、N=8，允许 Zvt。
  没有重新编译 MLIR/VMFB，未清理已有产物。测试进程 PATH 为 `/nonexistent`。
- 逐元素参考由独立标量循环计算，使用 double 累加，并按 tensor stride 和 batch 广播读取输入。
  容差为 `abs(actual-reference) <= 1e-6 + 1e-5*abs(reference)`。
- 小图输入使用有正有负、可被 F32/F16/BF16 精确表示的二进制分数，
  便于隔离布局与路由错误。本轮最大绝对误差为 0；这不替代真实模型的非零数值误差测试。

## 覆盖矩阵

| 范围 | 验证内容 | 结果 |
| --- | --- | --- |
| 能力查询 | 61 项检查：默认/F32 精度；各输入及输出的 F16/BF16/Q4_0；row/inner stride；batch 轴；extra；零/负/超大维度；shape 不匹配；空指针；其他操作 | 全部符合支持/拒绝约定 |
| 参数保守性 | BF16/F16/Q8/Q4 参数、Hadamard hint、16 个 op_params 槽位中的未知值 | Coral 拒绝 |
| buffer 类型 | CPU host buffer、非 host 类型、空类型 | 接受 host，拒绝其余类型 |
| 查询无执行副作用 | 注册后，将私有测试目录的索引及无效 VMFB 改名保留，再重复查询 | 无需原文件；无模块加载、数值调用或失败 |
| 合法 Coral 图 | M=1、M=16、显式 F32 accumulator/source precision | 自动分配 CORAL，数值正确，真实 dispatch 非零 |
| 合法 CPU 图 | 权重/activation 行间 padding；F16/BF16 权重；batch/broadcast；4 类非 F32 precision；缺少 AOT shape | 11 类均自动分配 CPU，数值正确，Coral 模块加载和 dispatch 为 0 |
| 混合图 | 同图一个受支持 matmul 和一个 BF16 precision matmul | 分别分配 CORAL/CPU，两个输出均正确 |
| 默认可变权重 | WEIGHTS buffer，但未承诺不可变；原地址连续更新权重 | 3 次输出随新值改变，每次重新准备，无缓存命中 |
| opt-in 下的普通输入 | 开启只读模型缓存，但该权重位于普通 buffer | 3 次原地址更新均生效，无缓存命中 |
| 可变视图 | 默认缓存选项；连续 view 的数据原地址更新 | 3 次更新均生效，无缓存命中 |
| 只读视图 | 开启缓存且底层权重保持不变，只改变 activation | 3 次结果正确，view 不进入 leaf-weight 缓存 |
| 不可变权重 | 两份同 shape 不同值权重，3 次调用中改变 activation | 两份初始快照互不串数据；后续 4 次命中，模块不重载 |
| 图预检查失败 | 同一 split 先放合法节点，再放不受支持节点，直接调用 backend | 返回 FAILED；所有 sentinel 输出保持不变；无模块加载或 dispatch |
| VMFB ABI 不匹配 | 索引请求 M=1，文件实际为 M=16 | 返回 FAILED，输出保持不变，无 dispatch |
| 失败后的复用 | 两类失败后再次调用原 backend | 继续 FAILED，不恢复为成功 |

非法元数据与未知参数只做能力查询，不送进可能断言的上游 CPU kernel。
Q4_0 的范围是拒绝能力查询；本轮没有测试量化 kernel 数值，也不扩大 v0.1 的 F32 范围。
三维/broadcast 的 CPU 执行和二维的 row stride 是实际数值用例；
其他被拒绝的 metadata 组合由能力查询覆盖。

## 执行证据

最终 CTest 运行：`build/m2-results/adapter-ypwxcfb0`。

~~~text
PASS capabilities: 61 checks, 0 dispatches
PASS weight_stride: 1 checks, 0 dispatches
PASS broadcast: 1 checks, 0 dispatches
PASS mixed_graph: 1 checks, 2 dispatches
PASS cache_default_mutable: 3 checks, 6 dispatches
PASS cache_optin_mutable: 3 checks, 6 dispatches
PASS cache_view_readonly: 3 checks, 6 dispatches
PASS cache_immutable: 3 checks, 12 dispatches
PASS preflight_failure: 2 checks, 0 dispatches
PASS abi_failure: 2 checks, 0 dispatches
~~~

| 统计 | 结果 |
| --- | ---: |
| 独立进程 | 23 |
| 命名检查项 | 95 |
| 成功 Coral matmul | 22 |
| 实际 Coral dispatch / 有效指令样本 | 43 / 43 |
| Kernel 退休指令数 | 6,171 |
| 标量参考最大绝对误差 | 0 |
| 有意触发并验证的 FAILED 返回 | 4 |
| CTest：M1 三项 + M2 独立验证 | 4/4 通过 |

所有成功 Coral 调用均检查非零指令数，并对账事件和 runtime 计数。
四次 FAILED 来自两个负例及各自重试；没有伪装为 CPU 成功，也没有发布被拒绝的输出。

原始 [verification.json](../build/m2-results/adapter-ypwxcfb0/verification.json) 和逐进程日志留在本机，
不随 Git 发布。[公开证据](m2-adapter-evidence.json) 保存完整检查名称、路由、dispatch、
缓存计数，以及测试源码、生产源码、可执行文件和 VMFB 的 SHA256，不含本机绝对路径。
[CTest 日志](../build/m1/Testing/Temporary/LastTest.log) 为最近一次测试记录；本次记录的副本保存在
[独立结果目录](../build/m2-results/adapter-ypwxcfb0/ctest.log)。

首次调试的 `adapter-41d_lw7f` 在 view 用例中遇到测试图构造问题：
单纯的 metadata alias 没有向 scheduler 提供 VIEW 依赖。
测试改为显式 `ggml_view_2d` 后通过；无需改动生产 adapter。
中间结果 `adapter-a5ra1qxp` 也保留，正式依据为最后的 23 用例结果。

## 复现

在 WSL 项目根目录，保留现有构建目录：

~~~bash
source ./env.sh
cmake -S . -B build/m1 -DCORAL_BUILD_M2=ON
cmake --build build/m1 --target coral-adapter-test -j 4
ctest --test-dir build/m1 -R m2_adapter_boundaries --output-on-failure
~~~

CTest 仅在配置已有 `M1_VMFB_M1` 和 `M1_VMFB_M16` 时注册该测试。
也可直接运行并看到每个用例的结果：

~~~bash
python tests/ggml/verify_adapter.py \
    --runner build/m1/coral-adapter-test \
    --m1-vmfb build/m1-artifacts/m1_zvt.vmfb \
    --m16-vmfb build/m1-artifacts/m16_zvt.vmfb
~~~

每轮创建新的 `adapter-*` 结果目录，不覆盖旧证据。

## 验收结论与边界

已有 [模型集成与性能报告](m2-test-report.md) 验证真实 GGUF 的短 prefill、三次 decode、
固定 token logits、真实 NPU 模拟执行与 runner 失败传播；
本轮补齐原计划要求的独立 layout/dtype/precision 拒绝测试，以及权重缓存边界。

M2 / v0.1 的功能验收证据现已齐备。测试范围仍是固定工具链、单线程、同步执行、F32 Coral matmul。
长期并发、其他模型、完整 attention/FFN offload、量化执行和真实硬件性能不在本次验收中。
本轮没有重跑性能 benchmark；原报告中的测量值继续对应原模型 runner 和原始测量记录。
