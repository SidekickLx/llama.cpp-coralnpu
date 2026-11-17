// SPDX-License-Identifier: MIT
#include "stablehlo_ops.h"
#include "stablehlo/dialect/StablehloOps.h"
namespace coral::compiler {
using namespace mlir;
namespace hlo = mlir::stablehlo;
mlir::Value BuildDenseDot(OpBuilder &builder, Location loc, RankedTensorType result,
          Value activation, Value weight) {
    // GGML stores weights as [N,K]. Both contracting dimensions are 1;
    // no transpose node, batch dimension, or implicit broadcast is needed.
    const auto dims = hlo::DotDimensionNumbersAttr::get(
        builder.getContext(), {}, {}, {1}, {1});
    const auto precision = hlo::PrecisionAttr::get(
        builder.getContext(), hlo::Precision::DEFAULT);
    return builder.create<hlo::DotGeneralOp>(
        loc, result, activation, weight, dims,
        builder.getArrayAttr({precision, precision}), hlo::DotAlgorithmAttr{});
}

mlir::Value BuildSilu(OpBuilder &builder, Location loc, Value input) {
    const auto type = llvm::cast<RankedTensorType>(input.getType());
    const auto scalar_type = RankedTensorType::get({}, builder.getF32Type());
    const auto one_attr = DenseElementsAttr::get(scalar_type, builder.getF32FloatAttr(1.0));
    const mlir::Value one = builder.create<hlo::ConstantOp>(loc, one_attr);
    const mlir::Value broadcast = builder.create<hlo::BroadcastInDimOp>(
        loc, type, one, builder.getDenseI64ArrayAttr({}));
    const mlir::Value negative = builder.create<hlo::NegOp>(loc, input);
    const mlir::Value exponential = builder.create<hlo::ExpOp>(loc, negative);
    const mlir::Value denominator = builder.create<hlo::AddOp>(loc, broadcast, exponential);
    return builder.create<hlo::DivOp>(loc, input, denominator);
}
} // namespace coral::compiler
