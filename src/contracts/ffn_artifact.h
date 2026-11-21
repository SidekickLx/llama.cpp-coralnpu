// SPDX-License-Identifier: MIT
#pragma once
#include "contracts/ffn_contract.h"
#include <stdexcept>
#include <string>

namespace coral::contracts {
inline constexpr char kFFNArtifactSchema[] = "coral.ffn.artifact.v1";
inline constexpr char kFFNReflectionKey[] = "coral.m4.ffn.abi";
inline constexpr char kFFNPipeline[] = "coral.ffn.aot.f32.v1";
inline constexpr char kFFNNumericPolicy[] = "f32-no-fast-math-v1";

inline const char *FFNAbi(const FFNCompileSpec &spec) {
    return spec.weight_layout == FFNWeightLayout::Transposed ? kFFNTransposedAbi : kFFNAbi;
}
inline void ValidateFFNSpec(const FFNCompileSpec &spec) {
    const auto &s = spec.shape;
    constexpr std::int64_t limit = 2147483647 / sizeof(float);
    if (spec.contract_version != kFFNContractVersion ||
        spec.numeric_policy != FFNNumericPolicy::F32NoFastMathV1 ||
        (spec.weight_layout != FFNWeightLayout::Native &&
         spec.weight_layout != FFNWeightLayout::Transposed) ||
        s.m <= 0 || s.d <= 0 || s.f <= 0 ||
        s.m > limit || s.d > limit || s.f > limit ||
        s.m > limit / s.d || s.m > limit / s.f || s.d > limit / s.f) {
        throw std::invalid_argument("invalid or unsupported FFN compile specification");
    }
}

inline std::string FFNBindingTag(const FFNCompileSpec &spec) {
    ValidateFFNSpec(spec);
    const auto m = std::to_string(spec.shape.m);
    const auto d = std::to_string(spec.shape.d);
    const auto f = std::to_string(spec.shape.f);
    const auto rows = spec.weight_layout == FFNWeightLayout::Transposed ? d : f;
    const auto cols = spec.weight_layout == FFNWeightLayout::Transposed ? f : d;
    return std::string(FFNAbi(spec)) + ";f32;row-major;X=" + m + "x" + d +
           ";W_gate=" + rows + "x" + cols + ";W_up=" + rows + "x" + cols +
           ";W_down=" + cols + "x" + rows + ";Y=" + m + "x" + d;
}
} // namespace coral::contracts
