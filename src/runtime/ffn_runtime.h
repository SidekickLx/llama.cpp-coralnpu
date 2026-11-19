// SPDX-License-Identifier: MIT
#pragma once
#include "contracts/ffn_artifact.h"
#include "runtime/matmul_runtime.h"

namespace coral::runtime {
namespace detail {
struct FFNWeightState;
}

// Immutable owning snapshots of W_gate[F,D], W_up[F,D], W_down[D,F].
// All three bindings are validated before any upload. Snapshots may be reused
// by modules with different M and identical D/F on the same DeviceContext.
// Borrowed host bindings need remain valid only until construction returns.
class PreparedFFNWeights {
  public:
    PreparedFFNWeights(const DeviceContext &, const contracts::FFNCompileSpec &,
                       TensorBinding gate, TensorBinding up, TensorBinding down);
    ~PreparedFFNWeights();
    PreparedFFNWeights(PreparedFFNWeights &&) noexcept;
    PreparedFFNWeights &operator=(PreparedFFNWeights &&) noexcept;
    PreparedFFNWeights(const PreparedFFNWeights &) = delete;
    PreparedFFNWeights &operator=(const PreparedFFNWeights &) = delete;

  private:
    std::unique_ptr<detail::FFNWeightState> state_;
    friend class FFNModule;
};

// Loads one complete, validated AOT package. No compiler, GGML scheduler or
// MLIR dependency is required at runtime. The module retains its validated
// VMFB bytes and device even after the source package or context wrapper dies.
// Like other runtime objects, this interface is confined to one host thread.
class FFNModule {
  public:
    FFNModule(const DeviceContext &, const std::string &artifact_directory,
              const contracts::FFNCompileSpec &);
    ~FFNModule();
    // Semantic identity of the artifact actually loaded into this module.
    std::string program_sha256() const;
    FFNModule(FFNModule &&) noexcept;
    FFNModule &operator=(FFNModule &&) noexcept;
    FFNModule(const FFNModule &) = delete;
    FFNModule &operator=(const FFNModule &) = delete;

    // One synchronous call: X, gate, up, down -> owned Y[M,D].
    // Invalid bindings are rejected before execution. An execution failure
    // poisons the shared context and returns no partial result or CPU replay.
    std::vector<float> Run(TensorBinding activation, const PreparedFFNWeights &);

  private:
    std::unique_ptr<detail::ModuleState> state_;
};
} // namespace coral::runtime
