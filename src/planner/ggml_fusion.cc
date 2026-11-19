// SPDX-License-Identifier: MIT
#include "planner/ggml_fusion.h"
#include "planner/ggml_ffn_pattern.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <set>

namespace coral::planner {
namespace {
using Code = RegionRejectionCode;
RegionRejection Reject(Code code, const char *message) {
    return {code, std::nullopt, message};
}
int Index(const ggml_cgraph &graph, const ggml_tensor *node) {
    for (int i = 0; i < graph.n_nodes; ++i) {
        if (graph.nodes[i] == node) {
            return i;
        }
    }
    return -1;
}
std::optional<RegionRejection> TensorCheck(const ggml_tensor *t) {
    if (!t || t->type != GGML_TYPE_F32) {
        return Reject(Code::UnsupportedDtype, "FFN requires F32 tensors");
    }
    if (t->view_src || t->view_offs || t->extra) {
        return Reject(Code::AliasOrView, "views and backend-specific storage are not admitted");
    }
    if (t->ne[0] <= 0 || t->ne[1] <= 0 || t->ne[2] != 1 || t->ne[3] != 1) {
        return Reject(Code::ShapeMismatch, "FFN requires positive unbatched matrices");
    }
    std::uint64_t bytes = sizeof(float);
    for (int axis = 0; axis < GGML_MAX_DIMS; ++axis) {
        const auto n = static_cast<std::uint64_t>(t->ne[axis]);
        if (n > static_cast<std::uint64_t>(std::numeric_limits<int32_t>::max()) / bytes) {
            return Reject(Code::ShapeMismatch, "tensor exceeds the F32 byte-addressing contract");
        }
        if (t->nb[axis] != bytes) {
            return Reject(Code::UnsupportedLayout, "FFN requires canonical dense strides");
        }
        bytes *= n;
    }
    return {};
}
std::optional<RegionRejection> NodeCheck(const ggml_tensor *t) {
    const int sources = t->op == GGML_OP_UNARY ? 1 : 2;
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if ((i < sources) != (t->src[i] != nullptr)) {
            return Reject(Code::UnsupportedPattern, "missing or unexpected operator source");
        }
    }
    for (int i = 0; i < GGML_MAX_OP_PARAMS / int(sizeof(int32_t)); ++i) {
        const auto value = t->op_params[i];
        if (t->op == GGML_OP_MUL_MAT && (i == 0 || i == 3)) {
            if (value != GGML_PREC_UNDEFINED && value != GGML_PREC_F32) {
                return Reject(Code::UnsupportedParameters, "unsupported matmul precision");
            }
        } else if (i == 0 && t->op == GGML_OP_UNARY) {
            if (value != GGML_UNARY_OP_SILU) {
                return Reject(Code::UnsupportedParameters, "unsupported unary activation");
            }
        } else if (i == 0 && t->op == GGML_OP_GLU) {
            if (value != GGML_GLU_OP_SWIGLU) {
                return Reject(Code::UnsupportedParameters, "unsupported GLU");
            }
        } else if (value) {
            return Reject(Code::UnsupportedParameters, "unknown parameter, swap, clamp or hint");
        }
    }
    return {};
}
bool Shape(const ggml_tensor *t, int64_t columns, int64_t rows) {
    return t->ne[0] == columns && t->ne[1] == rows;
}
std::optional<RegionRejection> Capability(const GGMLFFNPlan &plan, const FFNPolicy &policy) {
    if (!policy.query_member) {
        return Reject(Code::MissingMemberCapability, "member capability is unknown");
    }
    for (auto *node : plan.members) {
        const auto support = policy.query_member(node, policy.query_user_data);
        if (support == MemberSupport::MissingArtifact) {
            return Reject(Code::MissingMemberArtifact, "member has no prepared artifact");
        }
        if (support != MemberSupport::Supported) {
            return Reject(Code::MissingMemberCapability, "backend cannot execute every member");
        }
    }
    return {};
}
std::optional<RegionRejection> Uses(const ggml_cgraph &graph, const GGMLFFNPlan &plan) {
    if (!graph.use_counts || !graph.visited_hash_set.size || !graph.visited_hash_set.keys ||
        !graph.visited_hash_set.used) {
        return Reject(Code::MissingUseInformation, "parent graph use counts are unavailable");
    }
    std::vector<int> indices;
    std::vector<ggml_op> ops;
    for (auto *member : plan.members) {
        const int index = Index(graph, member);
        if (index < 0 || (!indices.empty() && index != indices.back() + 1)) {
            return Reject(Code::SplitMismatch, "members must be contiguous in one execution range");
        }
        const auto hash = ggml_hash_find(&graph.visited_hash_set, member);
        if (hash == GGML_HASHSET_FULL || !ggml_bitset_get(graph.visited_hash_set.used, hash)) {
            return Reject(Code::MissingUseInformation, "member has no parent graph use count");
        }
        indices.push_back(index);
        ops.push_back(member->op);
    }
    const int output = indices.back();
    // The view retains GLOBAL use counts. Counting just split-local consumers
    // would miss an intermediate read in a later backend split.
    if (!ggml_can_fuse_subgraph_ext(
            &graph, indices.data(), indices.size(), ops.data(), &output, 1)) {
        return Reject(Code::ExternalIntermediateUse, "upstream fusion check rejected uses, flags or views");
    }
    return {};
}
std::optional<RegionRejection> Bindings(const GGMLFFNPlan &plan, bool require_data) {
    std::vector<const ggml_tensor *> tensors(plan.inputs.begin(), plan.inputs.end());
    tensors.push_back(plan.members.back());
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        auto *a = tensors[i];
        if (!a->data) {
            if (require_data) {
                return Reject(Code::InputLifetimeNotProven, "live boundary storage is missing");
            }
            continue;
        }
        const auto start_a = reinterpret_cast<std::uintptr_t>(a->data);
        const auto size_a = static_cast<std::uint64_t>(a->ne[0]) * a->ne[1] * sizeof(float);
        if (size_a > std::numeric_limits<std::uintptr_t>::max() - start_a) {
            return Reject(Code::AliasOrView, "boundary address range overflows");
        }
        for (std::size_t j = 0; j < i; ++j) {
            auto *b = tensors[j];
            if (!b->data) {
                continue;
            }
            // Only repeated references to the identical read-only leaf weight
            // are admitted. Equal addresses on different tensors prove nothing.
            if (a == b && i < 4 && i > 0 && j > 0) {
                continue;
            }
            const auto start_b = reinterpret_cast<std::uintptr_t>(b->data);
            const auto size_b = static_cast<std::uint64_t>(b->ne[0]) * b->ne[1] * sizeof(float);
            if (start_a < start_b + size_b && start_b < start_a + size_a) {
                return Reject(Code::AliasOrView, "FFN boundaries have overlapping storage");
            }
        }
    }
    return {};
}
bool SameMetadata(const ggml_tensor &a, const ggml_tensor &b) {
    return a.type == b.type && a.op == b.op && a.flags == b.flags &&
           a.view_src == b.view_src && a.view_offs == b.view_offs && a.extra == b.extra &&
           std::memcmp(a.ne, b.ne, sizeof(a.ne)) == 0 &&
           std::memcmp(a.nb, b.nb, sizeof(a.nb)) == 0 &&
           std::memcmp(a.src, b.src, sizeof(a.src)) == 0 &&
           std::memcmp(a.op_params, b.op_params, sizeof(a.op_params)) == 0;
}
} // namespace

FFNCheckResult PlanFFN(const ggml_cgraph &split, int output_index,
                      const FFNPolicy &policy, ggml_backend_graph_optimize_params *allocation) {
    if (output_index < 0 || output_index >= split.n_nodes) {
        return Reject(Code::UnsupportedPattern, "invalid FFN anchor index");
    }
    const auto match = MatchFFNPattern(split.nodes[output_index]);
    if (!match) {
        return std::monostate{};
    }
    const auto &p = *match;
    if (!policy.generation) {
        return Reject(Code::StaleGraphOrSchedule, "a live generation is required");
    }
    if (policy.evaluation_callbacks) {
        return Reject(Code::DebugObservation, "evaluation callbacks disable FFN fusion");
    }
    GGMLFFNPlan plan{p.kind, {}, policy.generation,
                    {p.x, p.gate->src[0], p.up->src[0], p.down->src[0]}, {}, {}, {}};
    std::map<int, const ggml_tensor *> ordered;
    for (auto *member : {p.gate, p.up, p.silu, p.hidden, p.down}) {
        if (!member) {
            continue;
        }
        const int index = Index(split, member);
        if (index < 0) {
            return Reject(Code::SplitMismatch, "FFN crosses the supplied backend split");
        }
        if (!ordered.emplace(index, member).second) {
            return Reject(Code::UnsupportedPattern, "FFN members must be distinct");
        }
    }
    for (const auto &item : ordered) {
        plan.members.push_back(item.second);
    }
    if (plan.members.back() != p.down) {
        return Reject(Code::InputsNotReady, "down projection must be the last member");
    }
    std::set<const ggml_tensor *> seen;
    for (auto *input : plan.inputs) {
        if (auto error = TensorCheck(input)) {
            return *error;
        }
        if (seen.insert(input).second) {
            plan.snapshots.push_back({input, *input});
        }
        const int producer = Index(split, input);
        if (producer >= ordered.begin()->first) {
            return Reject(Code::InputsNotReady, "external input is produced inside or after the region");
        }
    }
    for (int i = 1; i < 4; ++i) {
        auto *weight = plan.inputs[i];
        if (weight->op != GGML_OP_NONE || weight == plan.inputs[0]) {
            return Reject(Code::UnsupportedPattern, "weights must be separate read-only leaf inputs");
        }
    }
    for (auto *member : plan.members) {
        if (auto error = TensorCheck(member)) {
            return *error;
        }
        if (auto error = NodeCheck(member)) {
            return *error;
        }
        for (auto *source : member->src) {
            if (source && !seen.count(source)) {
                return Reject(Code::InputsNotReady, "member sources are not topologically ready");
            }
        }
        seen.insert(member);
        plan.snapshots.push_back({member, *member});
    }
    plan.shape = {p.x->ne[1], p.x->ne[0], p.gate->ne[0]};
    const auto m = plan.shape.m, d = plan.shape.d, f = plan.shape.f;
    if (!Shape(plan.inputs[1], d, f) || !Shape(plan.inputs[2], d, f) ||
        !Shape(plan.inputs[3], f, d) || !Shape(p.gate, f, m) ||
        !Shape(p.up, f, m) || !Shape(p.hidden, f, m) ||
        !Shape(p.down, d, m) || (p.silu && !Shape(p.silu, f, m))) {
        return Reject(Code::ShapeMismatch, "FFN dimensions or gate/up bindings disagree");
    }
    if (auto error = Uses(split, plan)) {
        return *error;
    }
    if (auto error = Capability(plan, policy)) {
        return *error;
    }
    if (auto error = Bindings(plan, false)) {
        return *error;
    }
    if (!allocation || !allocation->add_alloc_dep) {
        return Reject(Code::InputLifetimeNotProven, "pre-allocation lifetime hook is required");
    }
    for (auto *member : plan.members) {
        plan.uses.push_back(ggml_node_get_use_count(&split, Index(split, member)));
    }
    // Register only after all admission checks. Protect X and weights until
    // the down projection, where the future fused invocation will complete.
    std::set<const ggml_tensor *> protected_inputs;
    for (auto *input : plan.inputs) {
        if (protected_inputs.insert(input).second) {
            allocation->add_alloc_dep(allocation->user_data,
                                      const_cast<ggml_tensor *>(input),
                                      const_cast<ggml_tensor *>(p.down));
        }
    }
    return plan;
}

std::optional<RegionRejection> CheckFFNExecution(
    const GGMLFFNPlan &plan, const ggml_cgraph &execution, const FFNPolicy &policy) {
    if (!policy.generation || policy.generation != plan.generation || plan.members.empty()) {
        return Reject(Code::StaleGraphOrSchedule, "plan generation is stale");
    }
    if (policy.evaluation_callbacks) {
        return Reject(Code::DebugObservation, "evaluation callbacks disable FFN fusion");
    }
    // Establish that every saved member is in the caller's LIVE graph before
    // dereferencing it. A retained plan may otherwise point into a freed arena.
    if (auto error = Uses(execution, plan)) {
        return error;
    }
    // Validate members first. Matching their live source pointers establishes
    // that each saved boundary input still names a source of this live region.
    for (auto *member : plan.members) {
        for (const auto &snapshot : plan.snapshots) {
            if (snapshot.tensor == member && !SameMetadata(snapshot.metadata, *member)) {
                return Reject(Code::StaleGraphOrSchedule, "member semantics changed after planning");
            }
        }
    }
    for (const auto &snapshot : plan.snapshots) {
        if (!SameMetadata(snapshot.metadata, *snapshot.tensor)) {
            return Reject(Code::StaleGraphOrSchedule, "source semantics changed after planning");
        }
    }
    for (std::size_t i = 0; i < plan.members.size(); ++i) {
        if (ggml_node_get_use_count(&execution, Index(execution, plan.members[i])) != plan.uses[i]) {
            return Reject(Code::StaleGraphOrSchedule, "parent graph uses changed");
        }
    }
    if (auto error = Capability(plan, policy)) {
        return error;
    }
    return Bindings(plan, true);
}
} // namespace coral::planner
