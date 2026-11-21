// SPDX-License-Identifier: MIT
#pragma once
#include "contracts/pointwise_contract.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace coral::runtime {

// The first ABI supports only Y[M,N] = X[M,K] * transpose(W[N,K]), F32.
struct MatmulShape {
    std::size_t m = 1;
    std::size_t k = 16;
    std::size_t n = 8;

    void Validate() const;
    std::size_t InputElements() const;
    std::size_t WeightElements() const;
    std::size_t OutputElements() const;
    std::string AbiTag() const;
};

// Borrowed storage. It must remain valid until the synchronous operation
// returns. Dense() does not take ownership or extend a vector's lifetime.
struct TensorBinding {
    const float *data = nullptr;
    std::size_t element_count = 0;
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::size_t row_stride_bytes = 0;

    static TensorBinding Dense(const std::vector<float> &values, std::size_t rows,
                               std::size_t cols);
    void Validate(std::size_t expected_rows, std::size_t expected_cols) const;
};

struct RuntimeStats {
    std::uint64_t module_loads = 0;
    std::uint64_t calls_attempted = 0;
    std::uint64_t calls_completed = 0;
    std::uint64_t coral_dispatches = 0;
    std::uint64_t failed_dispatches = 0;
    std::uint64_t dispatches_with_instruction_count = 0;
    std::uint64_t kernel_instructions = 0;
    std::uint64_t observer_errors = 0;
    std::uint64_t host_executable_load_attempts = 0;
    // Only explicit input/weight buffer allocations made by this adapter.
    std::uint64_t binding_allocations = 0;
    // Logical host <-> HAL copies, NOT MPACT DDR/TCM traffic.
    std::uint64_t upload_bytes = 0;
    std::uint64_t download_bytes = 0;
    // Host FFN row-boundary work, separate from VM invocation and simulator
    // traffic. Count only successfully completed padded invocations.
    std::uint64_t ffn_padded_rows = 0, ffn_cropped_rows = 0;
    double ffn_boundary_ms = 0.0;
    double module_load_ms = 0.0;
    double invoke_ms = 0.0;
};

namespace detail {
struct DeviceState;
struct WeightState;
struct ModuleState;
} // namespace detail

// Shared ownership keeps devices alive while a module or weight uses them.
// All objects associated with a context are confined to one host thread.
// Uses MPACT explicitly; there is no environment-selected hardware backend.
class DeviceContext {
  public:
    DeviceContext();
    ~DeviceContext();
    DeviceContext(const DeviceContext &) = delete;
    DeviceContext &operator=(const DeviceContext &) = delete;

    RuntimeStats stats() const;
    // Create a fresh healthy device. Existing modules/snapshots retain their
    // original owner and must not be mixed with this new context.
    void Reset();

  private:
    std::shared_ptr<detail::DeviceState> state_;
    friend class PreparedWeight;
    friend class LoadedModule;
    friend class PointwiseModule;
    friend class FFNModule;
};

// An immutable, owning snapshot of W[N,K]. No shape-only weight cache.
// Preparing two equal-shaped weights always makes two independent snapshots.
class PreparedWeight {
  public:
    PreparedWeight(const DeviceContext &device, TensorBinding weights);
    ~PreparedWeight();
    PreparedWeight(PreparedWeight &&) noexcept;
    PreparedWeight &operator=(PreparedWeight &&) noexcept;
    PreparedWeight(const PreparedWeight &) = delete;
    PreparedWeight &operator=(const PreparedWeight &) = delete;

  private:
    std::unique_ptr<detail::WeightState> state_;
    friend class LoadedModule;
    friend class FFNModule;
};

// One VM context and loaded module, reused across all Run() calls.
// Argument validation throws std::invalid_argument. IREE/device/ABI failures
// throw std::runtime_error. An execution failure poisons the shared context;
// construct a new DeviceContext before trying again.
class LoadedModule {
  public:
    LoadedModule(const DeviceContext &device, const std::string &vmfb_path, MatmulShape shape,
                 const std::string &function = "matmul");
    ~LoadedModule();
    LoadedModule(LoadedModule &&) noexcept;
    LoadedModule &operator=(LoadedModule &&) noexcept;
    LoadedModule(const LoadedModule &) = delete;
    LoadedModule &operator=(const LoadedModule &) = delete;

    // Returns an owned output only after invocation, shape/type checks,
    // real Coral dispatch evidence and readback have all succeeded.
    std::vector<float> Run(TensorBinding activation, const PreparedWeight &weights);

  private:
    std::unique_ptr<detail::ModuleState> state_;
};

// Reuses the same strict VM/HAL context and Coral-only execution checks.
// Inputs are copied at each call and staging buffers are reused. There are no
// immutable weights in this ABI and no cached activation values.
class PointwiseModule {
  public:
    PointwiseModule(const DeviceContext &, const std::string &vmfb_path,
                    contracts::PointwiseSpec);
    ~PointwiseModule();
    PointwiseModule(PointwiseModule &&) noexcept;
    PointwiseModule &operator=(PointwiseModule &&) noexcept;
    PointwiseModule(const PointwiseModule &) = delete;
    PointwiseModule &operator=(const PointwiseModule &) = delete;
    std::vector<float> Run(const std::vector<TensorBinding> &inputs);
  private:
    std::unique_ptr<detail::ModuleState> state_;
};

} // namespace coral::runtime
