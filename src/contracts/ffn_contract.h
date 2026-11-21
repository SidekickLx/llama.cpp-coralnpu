// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>

namespace coral::contracts {

// These are metadata contracts, not a computational IR or a serialized C++ ABI.
// See docs/m3-contracts.md and docs/m4-artifacts.md for validation and execution.
inline constexpr std::uint32_t kFFNContractVersion = 1;
inline constexpr char kFFNAbi[] = "coral.ffn.f32.v1";
inline constexpr char kFFNEntryPoint[] = "ffn";
inline constexpr char kFFNTransposedAbi[] = "coral.ffn.f32.weights-transposed.v1";

// Physical layout of the VM arguments. Public runtime preparation always
// accepts native GGML weights and owns the conversion into this layout.
enum class FFNWeightLayout { Native, Transposed };

enum class FFNNumericPolicy {
    // F32 tensors and dot accumulation; no reduced-precision substitution.
    // The compiler must disable Coral fast math. Results are tolerance-based.
    F32NoFastMathV1,
};

struct FFNShape {
    // Exact positive dimensions. Zero defaults represent an incomplete request.
    std::int64_t m = 0;
    std::int64_t d = 0;
    std::int64_t f = 0;
};

struct FFNCompileSpec {
    std::uint32_t contract_version = kFFNContractVersion;
    FFNShape shape;
    FFNNumericPolicy numeric_policy = FFNNumericPolicy::F32NoFastMathV1;
    FFNWeightLayout weight_layout = FFNWeightLayout::Native;
};

// Positional function ABI. Weights are external inputs, never embedded values.
// X[M,D], W_gate[F,D], W_up[F,D], W_down[D,F] -> Y[M,D].
// All tensors are dense row-major F32. These values are function argument
// positions only; serialize descriptive role names in manifests.
enum class FFNInputPosition : std::uint32_t {
    Activation = 0,
    GateWeight = 1,
    UpWeight = 2,
    DownWeight = 3,
};

// Stable IDs belong to a particular graph snapshot. They are not addresses,
// global model identities, weight-cache keys, or persistent artifact keys.
struct TensorId {
    std::uint64_t value = 0;
};

struct GraphSnapshotId {
    // Both fields must be nonzero. Any structural/semantic graph change
    // requires a new revision; activation value changes alone do not.
    std::uint64_t instance = 0;
    std::uint64_t revision = 0;
};

// Role bindings are separate from the mathematical program in StableHLO.
// A GGML node's TensorId identifies the tensor it produces.
struct FFNTensorIds {
    TensorId activation;
    TensorId gate_weight;
    TensorId up_weight;
    TensorId down_weight;
    TensorId output;
};

} // namespace coral::contracts
