// SPDX-License-Identifier: MIT
#pragma once
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

namespace coral::compiler {
// Callers validate types/shapes before these standard operation constructors.
mlir::Value BuildDenseDot(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::RankedTensorType result, mlir::Value activation,
                         mlir::Value weight);
mlir::Value BuildSilu(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value input);
} // namespace coral::compiler
