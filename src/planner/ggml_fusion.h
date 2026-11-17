// SPDX-License-Identifier: MIT
#pragma once
#include "planner/region_plan.h"
#include "ggml.h"
#include <array>
#include <optional>
#include <variant>
#include <vector>

struct ggml_backend_graph_optimize_params;

namespace coral::planner {
enum class MemberSupport { Supported, Unsupported, MissingArtifact };

struct FFNPolicy {
    // The owner increments this when a graph/schedule/allocation is rebuilt.
    std::uint64_t generation = 0;
    // Conservative default. The application must explicitly declare no callbacks.
    bool evaluation_callbacks = true;
    MemberSupport (*query_member)(const ggml_tensor *, void *) = nullptr;
    void *query_user_data = nullptr;
};

struct TensorSnapshot {
    const ggml_tensor *tensor;
    ggml_tensor metadata;
};

// Process-local, borrowed metadata. This is a preparation plan, not an
// executable or serialized IR. Tensors must outlive all checks of this plan.
struct GGMLFFNPlan {
    FFNPattern pattern;
    contracts::FFNShape shape;
    std::uint64_t generation;
    std::array<const ggml_tensor *, 4> inputs; // X, gate, up, down weights
    std::vector<const ggml_tensor *> members; // actual split execution order
    std::vector<TensorSnapshot> snapshots;
    std::vector<int> uses;
};

// monostate means no structural FFN match. A rejected candidate never adds
// allocation dependencies. Success registers dependencies before allocation.
using FFNCheckResult = std::variant<std::monostate, GGMLFFNPlan, RegionRejection>;
FFNCheckResult PlanFFN(const ggml_cgraph &split, int output_index,
                      const FFNPolicy &policy,
                      ggml_backend_graph_optimize_params *allocation);

// Recheck after allocation and on every actual graph_compute range, including
// graph reuse and evaluation-callback subdivisions. Does not execute/skip nodes.
std::optional<RegionRejection> CheckFFNExecution(
    const GGMLFFNPlan &plan, const ggml_cgraph &execution, const FFNPolicy &policy);
} // namespace coral::planner
