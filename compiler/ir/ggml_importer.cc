// SPDX-License-Identifier: MIT
#include "ggml_importer.h"
#include "stablehlo_ops.h"
#include "stablehlo_program.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
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
        default: return "unsupported";
    }
}

LogicalResult ValidateNode(const ggml_tensor &node, Location loc, std::size_t count) {
    if (node.op != GGML_OP_MUL_MAT) {
        return Reject(loc, "operator", "only MUL_MAT is currently admitted");
    }
    if (count != 2 || !node.src[0] || !node.src[1]) {
        return Reject(loc, "operands", "MUL_MAT requires two sources");
    }
    for (int slot = 2; slot < GGML_MAX_SRC; ++slot) {
        if (node.src[slot]) {
            return Reject(loc, "operands", "unexpected extra GGML source");
        }
    }
    for (std::size_t word = 0; word < GGML_MAX_OP_PARAMS / sizeof(int32_t); ++word) {
        const auto value = node.op_params[word];
        if (value != 0 && !((word == 0 || word == 3) && value == GGML_PREC_F32)) {
            return Reject(loc, "parameters", "MUL_MAT requires default/F32 precision and no hints");
        }
    }
    if (node.src[0]->ne[0] != node.src[1]->ne[0] ||
        node.ne[0] != node.src[0]->ne[1] || node.ne[1] != node.src[1]->ne[1]) {
        return Reject(loc, "shape", "MUL_MAT source/output dimensions disagree");
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
    // Source order is weight,activation in GGML and activation,weight in HLO.
    return BuildDenseDot(builder, loc, *result_type, operands[1], operands[0]);
}

mlir::OwningOpRef<mlir::ModuleOp> ImportGGMLFunction(
    mlir::MLIRContext &context, const GGMLImportRequest &request) {
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
    for (const auto group : {request.inputs, request.operations}) {
        for (const auto &ref : group) {
            if (!ref.tensor || !ValidId(ref.id.value) || identities.count(ref.tensor) ||
                ids.count(ref.id.value)) {
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
    module->push_back(function);
    auto *block = function.addEntryBlock();
    builder.setInsertionPointToEnd(block);
    llvm::DenseMap<const ggml_tensor *, mlir::Value> values;
    for (std::size_t i = 0; i < request.inputs.size(); ++i) {
        const auto &ref = request.inputs[i];
        values[ref.tensor] = block->getArgument(i);
        function.setArgAttr(i, "coral.ggml_tensor_id", builder.getI64IntegerAttr(ref.id.value));
    }
    for (const auto &ref : request.operations) {
        llvm::SmallVector<mlir::Value> operands;
        for (auto *source : ref.tensor->src) {
            if (!source) {
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
} // namespace coral::compiler
