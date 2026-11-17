// SPDX-License-Identifier: MIT
#pragma once
#include "planner/region_plan.h"
#include "ggml.h"
#include <optional>

namespace coral::planner {
// Borrowed role selection only. Matching is not semantic validation, fusion
// permission, a schedule, or an executable. Tensor names and payloads are ignored.
struct GGMLFFNMatch {
    FFNPattern kind;
    const ggml_tensor *x = nullptr;
    const ggml_tensor *gate = nullptr;
    const ggml_tensor *up = nullptr;
    const ggml_tensor *silu = nullptr;
    const ggml_tensor *hidden = nullptr;
    const ggml_tensor *down = nullptr;
};
std::optional<GGMLFFNMatch> MatchFFNPattern(const ggml_tensor *down);
} // namespace coral::planner
