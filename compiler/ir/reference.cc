// SPDX-License-Identifier: MIT
#include "reference.h"
#include "stablehlo_program.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/reference/Api.h"
#include "stablehlo/reference/Configuration.h"

namespace coral::compiler {
namespace {
using namespace mlir;
namespace hlo = mlir::stablehlo;
// Reference execution is intentionally bounded and diagnostic. These limits are
// not backend capability claims and do not change the FFN mathematical ABI.
constexpr int64_t kMaxElements = 1 << 20;
constexpr int64_t kMaxDotProducts = 10000000;

bool SmallF32Tensor(Type type) {
    const auto tensor = llvm::dyn_cast<RankedTensorType>(type);
    if (!tensor || !tensor.hasStaticShape() || !tensor.getElementType().isF32()) {
        return false;
    }
    int64_t elements = 1;
    for (auto size : tensor.getShape()) {
        if (size <= 0 || size > kMaxElements / elements) {
            return false;
        }
        elements *= size;
    }
    return true;
}

bool Finite(DenseElementsAttr value) {
    for (const auto number : value.getValues<llvm::APFloat>()) {
        if (!number.isFinite()) {
            return false;
        }
    }
    return true;
}

LogicalResult CheckProgram(ModuleOp module, func::FuncOp function) {
    if (function.isExternal() || !llvm::hasSingleElement(function.getBody())) {
        return module.emitError("reference execution requires one defined, single-block function");
    }
    int64_t total_elements = 0;
    int64_t dot_products = 0;
    int operations = 0;
    for (auto type : function.getFunctionType().getInputs()) {
        if (!SmallF32Tensor(type)) {
            return function.emitError("reference inputs must be small static F32 tensors");
        }
        total_elements += llvm::cast<RankedTensorType>(type).getNumElements();
    }
    for (auto &operation : function.front()) {
        if (++operations > 1024 || !llvm::isa<
                hlo::ConstantOp, hlo::DotGeneralOp, hlo::BroadcastInDimOp,
                hlo::NegOp, hlo::ExpOp, hlo::AddOp, hlo::DivOp, hlo::MulOp,
                hlo::PadOp, hlo::SliceOp,
                func::ReturnOp>(operation)) {
            return operation.emitError("operation is outside the M3 reference subset");
        }
        for (auto type : operation.getResultTypes()) {
            if (!SmallF32Tensor(type)) {
                return operation.emitError("reference values must be small static F32 tensors");
            }
            total_elements += llvm::cast<RankedTensorType>(type).getNumElements();
        }
        if (total_elements > kMaxElements) {
            return operation.emitError("reference tensor element budget exceeded");
        }
        if (auto constant = llvm::dyn_cast<hlo::ConstantOp>(operation)) {
            auto value = llvm::dyn_cast<DenseElementsAttr>(constant.getValue());
            if (!value || !Finite(value)) {
                return constant.emitError("reference constant must be finite dense F32");
            }
        }
        if (auto dot = llvm::dyn_cast<hlo::DotGeneralOp>(operation)) {
            const auto lhs = llvm::cast<RankedTensorType>(dot.getLhs().getType());
            const auto rhs = llvm::cast<RankedTensorType>(dot.getRhs().getType());
            const auto dims = dot.getDotDimensionNumbers();
            const llvm::SmallVector<int64_t> contracting = {1};
            const llvm::SmallVector<int64_t> transposed = {0};
            if (lhs.getRank() != 2 || rhs.getRank() != 2 ||
                !dims.getLhsBatchingDimensions().empty() ||
                !dims.getRhsBatchingDimensions().empty() ||
                dims.getLhsContractingDimensions() != llvm::ArrayRef<int64_t>(contracting) ||
                (dims.getRhsContractingDimensions() != llvm::ArrayRef<int64_t>(contracting) &&
                 dims.getRhsContractingDimensions() != llvm::ArrayRef<int64_t>(transposed)) ||
                dot.getAlgorithm()) {
                return dot.emitError("reference dot requires unbatched matrices, RHS contract axis 0 or 1, and no algorithm override");
            }
            // Each matrix has already passed the bounded element-count check.
            const int64_t output_axis = dims.getRhsContractingDimensions()[0] == 0 ? 1 : 0;
            dot_products += lhs.getDimSize(0) * lhs.getDimSize(1) * rhs.getDimSize(output_axis);
            if (dot_products > kMaxDotProducts) {
                return dot.emitError("reference dot product budget exceeded");
            }
        }
    }
    return success();
}
} // namespace

mlir::FailureOr<llvm::SmallVector<mlir::DenseElementsAttr>> EvaluateReference(
    mlir::ModuleOp module, llvm::StringRef entry,
    llvm::ArrayRef<mlir::DenseElementsAttr> inputs) {
    if (mlir::failed(VerifyModule(module))) {
        return mlir::failure();
    }
    auto function = module.lookupSymbol<mlir::func::FuncOp>(entry);
    if (!function || !llvm::hasSingleElement(module.getOps<mlir::func::FuncOp>())) {
        module.emitError("reference module must contain exactly one function matching the entry");
        return mlir::failure();
    }
    if (mlir::failed(CheckProgram(module, function))) {
        return mlir::failure();
    }
    if (inputs.size() != function.getNumArguments()) {
        function.emitError("reference input count does not match function signature");
        return mlir::failure();
    }
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        if (!inputs[i] || inputs[i].getType() != function.getArgument(i).getType() ||
            !Finite(inputs[i])) {
            function.emitError("reference input type/shape mismatch or non-finite data");
            return mlir::failure();
        }
    }
    mlir::OwningOpRef<mlir::ModuleOp> clone(
        llvm::cast<mlir::ModuleOp>(module->clone()));
    mlir::stablehlo::InterpreterConfiguration config;
    config.mainFunction = entry.str();
    // No project fallback is installed. Only the admitted standard operations
    // above may reach the official reference implementation.
    auto results = mlir::stablehlo::evalModule(*clone, inputs, config);
    if (mlir::failed(results)) {
        return mlir::failure();
    }
    if (results->size() != function.getNumResults()) {
        function.emitError("reference result count mismatch");
        return mlir::failure();
    }
    for (std::size_t i = 0; i < results->size(); ++i) {
        const auto value = (*results)[i];
        if (value.getType() != function.getResultTypes()[i] || !Finite(value)) {
            function.emitError("reference result type mismatch or non-finite output");
            return mlir::failure();
        }
    }
    return results;
}
} // namespace coral::compiler
