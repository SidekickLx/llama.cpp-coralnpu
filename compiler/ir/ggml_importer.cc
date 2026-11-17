// SPDX-License-Identifier: MIT
#include "ggml_importer.h"
#include "planner/ggml_ffn_pattern.h"
#include <set>
#include "stablehlo_ops.h"
#include "stablehlo_program.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "stablehlo/dialect/StablehloOps.h"
#include <limits>
#include <map>

namespace coral::compiler {
namespace {
using namespace mlir;

LogicalResult Reject(Location loc, llvm::StringRef code, llvm::StringRef message) {
    return emitError(loc) << '[' << code << "] " << message;
}

bool ValidId(uint64_t id) {
    return id != 0 && id <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
}

llvm::StringRef SourceOp(const ggml_tensor &node) {
    switch (node.op) {
        case GGML_OP_MUL_MAT: return "MUL_MAT";
        case GGML_OP_UNARY: return "UNARY";
        case GGML_OP_MUL: return "MUL";
        case GGML_OP_GLU: return "GLU";
        default: return "unsupported";
    }
}

LogicalResult ValidateNode(const ggml_tensor &node, Location loc, std::size_t count) {
    std::size_t required = 2;
    switch (node.op) {
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL:
        case GGML_OP_GLU:
            break;
        case GGML_OP_UNARY:
            required = 1;
            break;
        default:
            return Reject(loc, "operator", "only MUL_MAT, SILU, MUL and split SWIGLU are admitted");
    }
    if (count != required) {
        return Reject(loc, "operands", "wrong number of SSA operands");
    }
    for (std::size_t slot = 0; slot < GGML_MAX_SRC; ++slot) {
        if ((slot < required) != (node.src[slot] != nullptr)) {
            return Reject(loc, "operands", "missing source or unexpected extra GGML source");
        }
    }
    for (std::size_t word = 0; word < GGML_MAX_OP_PARAMS / sizeof(int32_t); ++word) {
        const auto value = node.op_params[word];
        if (node.op == GGML_OP_MUL_MAT && (word == 0 || word == 3)) {
            if (value != GGML_PREC_UNDEFINED && value != GGML_PREC_F32) {
                return Reject(loc, "parameters", "MUL_MAT requires default/F32 precision");
            }
        } else if (word == 0 && node.op == GGML_OP_UNARY) {
            if (value != GGML_UNARY_OP_SILU) {
                return Reject(loc, "parameters", "only UNARY(SILU) is admitted");
            }
        } else if (word == 0 && node.op == GGML_OP_GLU) {
            if (value != GGML_GLU_OP_SWIGLU) {
                return Reject(loc, "parameters", "only split GLU(SWIGLU) is admitted");
            }
        } else if (value != 0) {
            return Reject(loc, "parameters", "unknown parameters, hints, swaps or clamps are unsupported");
        }
    }
    if (node.op == GGML_OP_MUL_MAT) {
        if (node.src[0]->ne[0] != node.src[1]->ne[0] ||
            node.ne[0] != node.src[0]->ne[1] || node.ne[1] != node.src[1]->ne[1]) {
            return Reject(loc, "shape", "MUL_MAT source/output dimensions disagree");
        }
    } else {
        // GGML can repeat/broadcast operands. The initial dense FFN contract
        // intentionally admits only exact shape equality at every operand.
        for (std::size_t slot = 0; slot < required; ++slot) {
            for (int axis = 0; axis < GGML_MAX_DIMS; ++axis) {
                if (node.ne[axis] != node.src[slot]->ne[axis]) {
                    return Reject(loc, "shape", "elementwise broadcasting or shape changes are unsupported");
                }
            }
        }
    }
    return success();
}
} // namespace

mlir::FailureOr<mlir::RankedTensorType> ImportGGMLTensorType(
    mlir::MLIRContext &context, mlir::Location loc, const ggml_tensor *tensor) {
    if (!tensor || tensor->type != GGML_TYPE_F32) {
        Reject(loc, "tensor-type", "expected a non-null F32 tensor");
        return mlir::failure();
    }
    if (tensor->extra || tensor->view_src || tensor->view_offs) {
        Reject(loc, "layout", "backend-specific storage and view aliases require an explicit rule");
        return mlir::failure();
    }
    if (tensor->ne[0] <= 0 || tensor->ne[1] <= 0 || tensor->ne[2] != 1 || tensor->ne[3] != 1) {
        Reject(loc, "shape", "only positive, unbatched GGML matrices are admitted");
        return mlir::failure();
    }
    uint64_t bytes = sizeof(float);
    for (int axis = 0; axis < GGML_MAX_DIMS; ++axis) {
        const auto extent = static_cast<uint64_t>(tensor->ne[axis]);
        if (extent > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / bytes) {
            Reject(loc, "shape", "dense tensor byte count overflows");
            return mlir::failure();
        }
        if (tensor->nb[axis] != bytes) {
            Reject(loc, "layout", "expected canonical dense F32 strides");
            return mlir::failure();
        }
        bytes *= extent;
    }
    // GGML ne[0] is the contiguous column dimension. StableHLO uses [row,col].
    return mlir::RankedTensorType::get(
        {tensor->ne[1], tensor->ne[0]}, mlir::Float32Type::get(&context));
}

mlir::FailureOr<mlir::Value> ImportGGMLNode(
    mlir::OpBuilder &builder, mlir::Location loc, const ggml_tensor &node,
    llvm::ArrayRef<mlir::Value> operands) {
    auto result_type = ImportGGMLTensorType(*builder.getContext(), loc, &node);
    if (mlir::failed(result_type) || mlir::failed(ValidateNode(node, loc, operands.size()))) {
        return mlir::failure();
    }
    for (std::size_t slot = 0; slot < operands.size(); ++slot) {
        auto source_type = ImportGGMLTensorType(*builder.getContext(), loc, node.src[slot]);
        if (mlir::failed(source_type)) {
            return mlir::failure();
        }
        if (!operands[slot] || operands[slot].getType() != *source_type) {
            Reject(loc, "ssa-type", "SSA operand type differs from its GGML source");
            return mlir::failure();
        }
    }
    switch (node.op) {
        case GGML_OP_MUL_MAT:
            // GGML's weight,activation order differs from StableHLO's dot order.
            return BuildDenseDot(builder, loc, *result_type, operands[1], operands[0]);
        case GGML_OP_UNARY:
            return BuildSilu(builder, loc, operands[0]);
        case GGML_OP_MUL: {
            const mlir::Value result = builder.create<mlir::stablehlo::MulOp>(
                loc, operands[0], operands[1]);
            return result;
        }
        case GGML_OP_GLU: {
            // The pinned split SwiGLU definition is SiLU(src[0]) * src[1].
            // Equal dimensions do not make the gate/up roles interchangeable.
            const auto gate = BuildSilu(builder, loc, operands[0]);
            const mlir::Value result = builder.create<mlir::stablehlo::MulOp>(
                loc, gate, operands[1]);
            return result;
        }
        default:
            return mlir::failure(); // ValidateNode has rejected this case.
    }
}

static mlir::OwningOpRef<mlir::ModuleOp> ImportFunction(
    mlir::MLIRContext &context, const GGMLImportRequest &request,
    const planner::GGMLFFNMatch *ffn) {
    RegisterDialects(context);
    mlir::OpBuilder builder(&context);
    const auto loc = builder.getUnknownLoc();
    if (!ValidId(request.graph.instance) || !ValidId(request.graph.revision) ||
        request.entry_point.empty() || request.operations.empty() || request.outputs.empty()) {
        Reject(loc, "request", "graph identity, entry, operations and outputs are required");
        return {};
    }
    llvm::DenseMap<const ggml_tensor *, uint64_t> identities;
    std::map<uint64_t, const ggml_tensor *> ids;
    llvm::SmallVector<mlir::Type> input_types;
    llvm::SmallVector<mlir::Type> output_types;
    // Validate identity and layout before constructing any function.
    for (int group_index = 0; group_index < 2; ++group_index) {
        const auto group = group_index == 0 ? request.inputs : request.operations;
        for (const auto &ref : group) {
            const auto existing = identities.find(ref.tensor);
            const bool repeated_role = ffn && group_index == 0 && existing != identities.end() &&
                                       existing->second == ref.id.value;
            if (!ref.tensor || !ValidId(ref.id.value) ||
                (!repeated_role && (existing != identities.end() || ids.count(ref.id.value)))) {
                Reject(loc, "identity", "inputs/operations need unique nonzero tensor identities");
                return {};
            }
            auto type = ImportGGMLTensorType(context, loc, ref.tensor);
            if (mlir::failed(type)) {
                return {};
            }
            identities[ref.tensor] = ref.id.value;
            ids.emplace(ref.id.value, ref.tensor);
        }
    }
    for (const auto &ref : request.inputs) {
        input_types.push_back(*ImportGGMLTensorType(context, loc, ref.tensor));
    }
    for (const auto &ref : request.outputs) {
        const auto found = identities.find(ref.tensor);
        if (found == identities.end() || found->second != ref.id.value) {
            Reject(loc, "identity", "output must reference an admitted tensor and its exact ID");
            return {};
        }
        output_types.push_back(*ImportGGMLTensorType(context, loc, ref.tensor));
    }
    mlir::OwningOpRef<mlir::ModuleOp> module(mlir::ModuleOp::create(loc));
    auto function = mlir::func::FuncOp::create(
        loc, request.entry_point, builder.getFunctionType(input_types, output_types));
    function->setAttr("coral.ggml_graph_instance", builder.getI64IntegerAttr(request.graph.instance));
    function->setAttr("coral.ggml_graph_revision", builder.getI64IntegerAttr(request.graph.revision));
    if (ffn) {
        function->setAttr("coral.ffn_abi", builder.getStringAttr(contracts::kFFNAbi));
        function->setAttr("coral.numeric_policy", builder.getStringAttr("f32_no_fast_math_v1"));
    }
    module->push_back(function);
    auto *block = function.addEntryBlock();
    builder.setInsertionPointToEnd(block);
    llvm::DenseMap<const ggml_tensor *, mlir::Value> values;
    for (std::size_t i = 0; i < request.inputs.size(); ++i) {
        const auto &ref = request.inputs[i];
        values[ref.tensor] = block->getArgument(i);
        function.setArgAttr(i, "coral.ggml_tensor_id", builder.getI64IntegerAttr(ref.id.value));
        if (ffn) {
            const char *roles[] = {"activation", "gate_weight", "up_weight", "down_weight"};
            function.setArgAttr(i, "coral.ffn_role", builder.getStringAttr(roles[i]));
        }
    }
    for (const auto &ref : request.operations) {
        llvm::SmallVector<mlir::Value> operands;
        for (int slot = 0; slot < GGML_MAX_SRC; ++slot) {
            auto *source = ref.tensor->src[slot];
            if (!source) {
                continue;
            }
            // Shared GGML weight identity does not collapse the FFN ABI.
            // Select the weight argument by the verified consumer edge.
            if (ffn && slot == 0 && ref.tensor->op == GGML_OP_MUL_MAT) {
                const unsigned role = ref.tensor == ffn->gate ? 1 :
                                      ref.tensor == ffn->up ? 2 : 3;
                operands.push_back(block->getArgument(role));
                continue;
            }
            const auto found = values.find(source);
            if (found == values.end()) {
                Reject(loc, "dependency", "source is outside the boundary or appears after its user");
                return {};
            }
            operands.push_back(found->second);
        }
        const auto source_loc = mlir::NameLoc::get(
            builder.getStringAttr("ggml.tensor." + std::to_string(ref.id.value)));
        auto *previous = block->empty() ? nullptr : &block->back();
        auto result = ImportGGMLNode(builder, source_loc, *ref.tensor, operands);
        if (mlir::failed(result)) {
            return {};
        }
        // Preserve origin on every generated op, including SiLU decomposition.
        auto *operation = previous ? previous->getNextNode() : &block->front();
        for (; operation; operation = operation->getNextNode()) {
            operation->setAttr("coral.ggml_tensor_id", builder.getI64IntegerAttr(ref.id.value));
            operation->setAttr("coral.ggml_op", builder.getStringAttr(SourceOp(*ref.tensor)));
            operation->setAttr("coral.ggml_op_params", builder.getDenseI32ArrayAttr(ref.tensor->op_params));
        }
        values[ref.tensor] = *result;
    }
    llvm::SmallVector<mlir::Value> outputs;
    for (std::size_t i = 0; i < request.outputs.size(); ++i) {
        const auto &ref = request.outputs[i];
        outputs.push_back(values.lookup(ref.tensor));
        function.setResultAttr(i, "coral.ggml_tensor_id", builder.getI64IntegerAttr(ref.id.value));
    }
    builder.create<mlir::func::ReturnOp>(loc, outputs);
    if (mlir::failed(VerifyModule(*module))) {
        return {};
    }
    return module;
}
mlir::OwningOpRef<mlir::ModuleOp> ImportGGMLFunction(
    mlir::MLIRContext &context, const GGMLImportRequest &request) {
    return ImportFunction(context, request, nullptr);
}

mlir::OwningOpRef<mlir::ModuleOp> ImportGGMLFFNFunction(
    mlir::MLIRContext &context, const GGMLFFNImportRequest &request) {
    RegisterDialects(context);
    const auto loc = mlir::UnknownLoc::get(&context);
    const auto &spec = request.spec;
    if (spec.contract_version != contracts::kFFNContractVersion ||
        spec.numeric_policy != contracts::FFNNumericPolicy::F32NoFastMathV1 ||
        spec.shape.m <= 0 || spec.shape.d <= 0 || spec.shape.f <= 0) {
        Reject(loc, "ffn-contract", "unsupported FFN contract, numeric policy or dimensions");
        return {};
    }
    const auto match = planner::MatchFFNPattern(request.output.tensor);
    if (!match) {
        Reject(loc, "ffn-pattern", "output must anchor a complete admitted FFN pattern");
        return {};
    }
    const auto &p = *match;
    const ggml_tensor *roles[] = {p.x, p.gate->src[0], p.up->src[0], p.down->src[0]};
    for (std::size_t i = 0; i < request.inputs.size(); ++i) {
        if (request.inputs[i].tensor != roles[i] ||
            (i > 0 && (roles[i] == roles[0] || roles[i]->op != GGML_OP_NONE))) {
            Reject(loc, "ffn-roles", "FFN arguments must preserve activation and leaf-weight roles");
            return {};
        }
    }
    if (p.x->ne[1] != spec.shape.m || p.x->ne[0] != spec.shape.d ||
        roles[1]->ne[1] != spec.shape.f || roles[2]->ne[1] != spec.shape.f ||
        roles[3]->ne[1] != spec.shape.d || roles[3]->ne[0] != spec.shape.f ||
        p.down->ne[1] != spec.shape.m || p.down->ne[0] != spec.shape.d) {
        Reject(loc, "ffn-contract", "selected region differs from the requested M/D/F shape");
        return {};
    }
    std::set<const ggml_tensor *> expected{p.gate, p.up, p.hidden, p.down};
    if (p.silu) {
        expected.insert(p.silu);
    }
    const std::size_t count = p.silu ? 5 : 4;
    if (expected.size() != count || request.operations.size() != count) {
        Reject(loc, "ffn-members", "FFN members must be complete and distinct");
        return {};
    }
    for (const auto &ref : request.operations) {
        if (!expected.erase(ref.tensor)) {
            Reject(loc, "ffn-members", "unexpected or repeated FFN member");
            return {};
        }
    }
    const GGMLImportRequest generic{
        request.graph, contracts::kFFNEntryPoint, request.inputs,
        request.operations, llvm::ArrayRef<GGMLTensorRef>(&request.output, 1)};
    return ImportFunction(context, generic, &p);
}

} // namespace coral::compiler
