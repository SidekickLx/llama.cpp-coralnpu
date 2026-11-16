# M1：独立 F32 matmul runner

## 状态与阅读顺序

本分支提供 M1 runner、C++ 适配层、StableHLO 导出器和构建描述。
**本轮仅实现源码并做静态审查，尚未配置构建、编译或运行新代码。**
下文命令是后续验证步骤，不代表已经执行成功。M1 运行验收尚未完成。

本实现针对 dependencies.lock.json 中的固定版本，仅支持 Linux x86_64
（包括 WSL2）与 MPACT。复用 IREE/Coral C API，不修改上游，不实现新 kernel。

推荐阅读：

1. [matmul_runtime.h](../src/runtime/matmul_runtime.h)：主要对象与外部接口。
2. [matmul_runner.cc](../src/runtime/matmul_runner.cc)：命令行、CPU 参考和用例。
3. [matmul_runtime.cc](../src/runtime/matmul_runtime.cc)：资源、设备和模块调用。
4. [export_matmul.py](../compiler/export_matmul.py)：生成 StableHLO。
5. [CMakeLists.txt](../CMakeLists.txt)：未来构建入口及可选 CTest 注册。

## Tensor ABI

```text
Y[m,n] = sum_k X[m,k] * W[n,k]
```

| 参数 | 普通 row-major shape | 对应 GGML ne |
| --- | --- | --- |
| 第 0 个参数 X | [M,K]，F32 | [K,M,1,1] |
| 第 1 个参数 W | [N,K]，F32 | [K,N,1,1] |
| 唯一返回值 Y | [M,N]，F32 | [N,M,1,1] |

矩阵必须连续、无广播，元素步长为 4 字节、行步长为列数乘 4。
维度大于零，每个张量不得超过 2^31-1 字节；这是 RV32 的保守地址边界，
不代表设备拥有这么多 TCM。默认 8 KiB ITCM / 32 KiB DTCM 下的实际可编译范围
仍需验证。首批 (M,K,N) 为 (1,16,8)、(16,16,8)、(32,16,8)。

参数顺序固定为 X、W；未来 GGML adapter 将 src[1] 映射到 X、src[0] 映射到 W。
权重数值作为运行时参数，不固化到 MLIR/VMFB。

导出函数为 m1.matmul；CLI 的 --function 接受不带模块名前缀的导出名。
函数的 iree.reflection 包含 coral.m1.abi，例如：

```text
v1;f32;row-major;X=1x16;W=8x16;Y=1x8
```

runner 在创建 VM context 前核对标记和同步 VM 调用约定 0rr_r。
缺少标记、shape 不一致或 ABI 错误时拒绝加载。
官方 JAX 方阵示例没有该标记，不能直接作为本 runner 的输入。
该标记仅约束 tensor ABI；完整 target/profile、工具链兼容性 manifest 属于 M4。

## 所有权与执行

| 对象 | 职责 |
| --- | --- |
| DeviceContext | VM instance、host/Coral 设备、设备组、HAL module 和计数 |
| TensorBinding | 借用调用者数据，检查 shape、元素数和行步长 |
| PreparedWeight | 拥有独立的不可变权重快照，没有按 shape 共享权重的缓存 |
| LoadedModule | VMFB 字节、模块、VM context、列表和可复用 activation buffer |

模块和权重共享持有底层设备；公开的 DeviceContext 对象销毁后，
既有模块和权重仍持有有效设备。移动后的模块/权重不能调用。
所有对象约定在单个 host 线程使用，不支持并发调用或并发销毁。

```cpp
coral::runtime::DeviceContext device;
coral::runtime::MatmulShape shape{1, 16, 8};
coral::runtime::LoadedModule module(device, vmfb_path, shape);
coral::runtime::PreparedWeight weight(
    device, coral::runtime::TensorBinding::Dense(w, shape.n, shape.k));
auto y = module.Run(
    coral::runtime::TensorBinding::Dense(x, shape.m, shape.k), weight);
```

Dense() 不延长 vector 生命周期，数据必须活到同步操作返回。
PreparedWeight 构造完成后已复制数据，修改原始 vector 不改变快照；
换权重需新建 PreparedWeight，同 shape 的不同权重可共用模块代码。

每次 Run 更新 activation、复用 VM context 和权重，返回独立 host vector。
只有执行、输出 ABI 检查、真实 dispatch 证据和回读都成功后才发布结果。
调用结束清理 VM 输入输出引用，避免保留上一轮输出。

## 严格设备执行

- 明确选用 MPACT，不读取环境变量切换 FPGA/其他后端。
- 同时提供 local-sync 和 coralnpu，满足混合目标模块。
- host 仅配置拒绝型 loader：允许 embedded-ELF 格式协商，
  但加载任何 host 数值 executable 都会报错，不能执行 CPU 数值 kernel。
- 每次调用必须有非零 Coral dispatch；全部 dispatch 指令计数有效、合计非零，
  没有执行或观测失败。
- CPU 参考只用于验算，不用于替代失败的设备输出。
- 参数错误在执行前拒绝。调用阶段失败后，共享 context 不再复用，
  需要新建 DeviceContext。
- 不承诺进程内强制中断失控模拟器；批量测试由外部超时监督，
  CTest 对每个已注册测试设置 180 秒超时。

上游 MPACT 适配器缓存进程内模拟器，因此 --recreate 验证本项目的
设备/模块/buffer 生命周期，不表示每次都重新启动模拟器进程。

## 统计口径

stdout 输出数值用例和每个 context 的 JSON 行：

- module_loads、calls_attempted、calls_completed。
- coral_dispatches、failed_dispatches、instruction_samples、kernel_instructions。
- observer_errors、host_executable_load_attempts。
- binding_allocations：本适配层显式创建的 activation/weight buffer 次数，
  不包含 IREE 内部和输出分配。
- upload_bytes、download_bytes：逻辑 host 与 HAL buffer 之间的复制量，
  不等于 MPACT 内部 DDR/TCM staging。
- module_load_ms、invoke_ms：主机墙钟时间，不代表真实硬件性能。

host_numerical_execution=disabled 表示执行策略，不是伪造的实测 dispatch 计数。
权重快照复用也不等于已证明权重常驻真实 NPU。
设备内部 staging 和完整 allocator 统计仍待深化。

## 后续构建步骤：尚未执行

在项目根目录记录路径，然后按固定版本官方说明构建依赖。
之前构建产物已清理，此步骤可能重新下载 Bazel 依赖并长时间编译。

```bash
PROJECT_ROOT="$PWD"
CORAL_SOURCE="$PROJECT_ROOT/repos/coralnpu-compiler"
cd "$CORAL_SOURCE"
bazel build --config=dev \
    @iree_core//tools:iree-compile \
    @coralnpu_mpact//sim/hw_sim:libcoralnpu_simulator_mpact.so
cd "$PROJECT_ROOT"
```

将 MPACT_LIB 设为上述目标实际产出的 .so 绝对路径，再配置本项目：

```bash
cmake -S "$PROJECT_ROOT" -B "$PROJECT_ROOT/build/m1" -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_C_COMPILER=clang-19 \
    -DCMAKE_CXX_COMPILER=clang++-19 \
    -DCORALNPU_MPACT_SIMULATOR_LIB="$MPACT_LIB"
cmake --build "$PROJECT_ROOT/build/m1" --target coral-matmul-runner -j 4
```

入口仅引入 IREE runtime，不调用 Coral 顶层 CMake 的 compiler/CRT 构建。
必须提供已有的 MPACT .so；否则配置立即报错，避免上游插件在配置阶段自动启动 Bazel。
此接入尚未经配置或链接验证。

## 后续产物生成与运行：尚未执行

导出器直接写 StableHLO，不依赖 JAX/PyTorch，不编译或执行模型。
输出文件必须不存在：

```bash
mkdir -p "$PROJECT_ROOT/build/m1/artifacts"
python3 "$PROJECT_ROOT/compiler/export_matmul.py" \
    --m 1 --k 16 --n 8 \
    --output "$PROJECT_ROOT/build/m1/artifacts/m1.mlir"
cd "$CORAL_SOURCE"
bazel run --config=dev @iree_core//tools:iree-compile -- \
    "$PROJECT_ROOT/build/m1/artifacts/m1.mlir" \
    --iree-input-type=stablehlo \
    --iree-execution-model=async-internal \
    --iree-hal-target-device=local \
    --iree-hal-local-target-device-backends=llvm-cpu \
    --iree-llvmcpu-target-cpu=host \
    --iree-hal-target-device=coralnpu \
    --coralnpu-fast-math=false \
    --coralnpu-roofline-speedup-threshold=0 \
    --coralnpu-dump-affinity-profile-format=pretty \
    -o "$PROJECT_ROOT/build/m1/artifacts/m1_zvt.vmfb"
cd "$PROJECT_ROOT"
```

这是允许 Zvt 的配置；RVV-only 使用不同输出文件名，并增加：

```text
--coralnpu-target-cpu-features=+m,+f,+zbb,+zvl128b,+zve32f,+zfbfmin,+zvfbfmin,+zvfbfwma
```

async-internal 保留完整 HAL 调度，导出的函数接口仍同步接收两个 tensor 并返回一个。

```bash
"$PROJECT_ROOT/build/m1/coral-matmul-runner" \
    --module "$PROJECT_ROOT/build/m1/artifacts/m1_zvt.vmfb" \
    --m 1 --k 16 --n 8 --repeat 3 --recreate 2 --self-test
```

不传文件时使用含正负值的确定性数据。
--input 与 --weights 必须成对给出，格式为无头 little-endian 连续 F32，
大小精确为 M*K*4、N*K*4 字节；不是 .npy/GGUF。
--output 保存 baseline 的 Y，全部用例通过后才创建，拒绝覆盖已有文件。

--self-test 覆盖重复输入、修改 activation、同 shape 换权重、恢复旧权重、
零 activation、源 vector 修改后的权重快照隔离、错误 stride/元素数拒绝，
以及拒绝后继续调用。CPU 参考以 double 累加后转 F32，
容差为 abs(error) <= 1e-6 + 1e-5*abs(reference)，拒绝 NaN/Inf。

退出码：0 为当前请求检查全部通过；1 为运行/数值/文件失败；2 为参数错误。
--help 在创建设备前返回。

为 M=16、32 分别生成产物后，配置 M1_VMFB_M1、M1_VMFB_M16、M1_VMFB_M32
即可注册相应 CTest。必须提供已有 VMFB，CMake 不会自动生成。
Zvt/RVV 两套产物分别验收。

## 待实际验收

- [ ] CMake 配置与 C++ 编译、链接。
- [ ] 三种 M 的 Zvt/RVV 严格 F32 数值和真实执行证据。
- [ ] 重复调用、换权重、快照隔离与反复创建释放。
- [ ] 错误 ABI、shape、stride、损坏 VMFB 明确失败。
- [ ] 人为放在 host 的数值 kernel 被拒绝。
- [ ] 补充设备内部 staging/分配观测，完成 M1 全部统计。
