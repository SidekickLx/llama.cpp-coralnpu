// SPDX-License-Identifier: MIT
#pragma once
#include <cstddef>
#include <stdexcept>
#include <string>

namespace coral::contracts {
// Static standalone member ABI. Split SwiGLU takes gate first, then up.
// This describes function bindings; the computation remains standard StableHLO.
enum class PointwiseKind { Silu, Multiply, SwiGLU };
struct PointwiseSpec {
    PointwiseKind kind = PointwiseKind::Silu;
    std::size_t rows = 0;
    std::size_t cols = 0;
    const char *EntryPoint() const {
        switch (kind) {
        case PointwiseKind::Silu: return "silu";
        case PointwiseKind::Multiply: return "multiply";
        case PointwiseKind::SwiGLU: return "swiglu";
        }
        throw std::invalid_argument("unknown pointwise operation");
    }
    std::size_t Arity() const {
        (void)EntryPoint();
        return kind == PointwiseKind::Silu ? 1 : 2;
    }
    std::size_t Elements() const {
        constexpr std::size_t limit = 2147483647 / sizeof(float);
        if (!rows || !cols || cols > limit || rows > limit / cols) {
            throw std::invalid_argument("invalid or oversized pointwise shape");
        }
        return rows * cols;
    }
    std::string AbiTag() const {
        (void)Elements();
        return std::string("v1;f32;row-major;") + EntryPoint() + ";" +
               std::to_string(rows) + "x" + std::to_string(cols);
    }
};
inline constexpr char kPointwiseAbiKey[] = "coral.m3.pointwise.abi";
} // namespace coral::contracts
