// SPDX-License-Identifier: MIT
#include "planner/ggml_ffn_pattern.h"

namespace coral::planner {
namespace {
bool IsSilu(const ggml_tensor *node) {
    return node && node->op == GGML_OP_UNARY && node->op_params[0] == GGML_UNARY_OP_SILU;
}
} // namespace
std::optional<GGMLFFNMatch> MatchFFNPattern(const ggml_tensor *down) {
    if (!down || down->op != GGML_OP_MUL_MAT || !down->src[0] || !down->src[1]) {
        return {};
    }
    GGMLFFNMatch p{};
    p.down = down;
    p.hidden = down->src[1];
    if (p.hidden->op == GGML_OP_GLU && p.hidden->op_params[0] == GGML_GLU_OP_SWIGLU) {
        p.kind = FFNPattern::SplitSwiGlu;
        p.gate = p.hidden->src[0];
        p.up = p.hidden->src[1];
    } else if (p.hidden->op == GGML_OP_MUL) {
        p.kind = FFNPattern::SiluMultiply;
        const bool left = IsSilu(p.hidden->src[0]);
        const bool right = IsSilu(p.hidden->src[1]);
        if (left == right) {
            return {};
        }
        p.silu = p.hidden->src[left ? 0 : 1];
        p.gate = p.silu->src[0];
        p.up = p.hidden->src[left ? 1 : 0];
    } else {
        return {};
    }
    if (!p.gate || !p.up || p.gate->op != GGML_OP_MUL_MAT ||
        p.up->op != GGML_OP_MUL_MAT || !p.gate->src[0] || !p.up->src[0] ||
        !p.gate->src[1] || p.gate->src[1] != p.up->src[1]) {
        return {};
    }
    p.x = p.gate->src[1];
    return p;
}
} // namespace coral::planner
