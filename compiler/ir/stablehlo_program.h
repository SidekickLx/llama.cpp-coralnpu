// SPDX-License-Identifier: MIT
#pragma once
#include "contracts/ffn_contract.h"
#include "contracts/pointwise_contract.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

namespace coral::compiler {
// The caller owns the context, which must outlive every module and attribute.
// These APIs build standard StableHLO operations, never a project dialect.
mlir::OwningOpRef<mlir::ModuleOp> BuildPointwise(
    mlir::MLIRContext &context, const contracts::PointwiseSpec &spec);
void RegisterDialects(mlir::MLIRContext &context);
mlir::OwningOpRef<mlir::ModuleOp> BuildFFN(
    mlir::MLIRContext &context, const contracts::FFNCompileSpec &spec);
mlir::OwningOpRef<mlir::ModuleOp> BuildMatmul(
    mlir::MLIRContext &context, int64_t m, int64_t k, int64_t n);
mlir::OwningOpRef<mlir::ModuleOp> ReadModule(
    mlir::MLIRContext &context, llvm::StringRef path);
mlir::LogicalResult VerifyModule(mlir::ModuleOp module);
mlir::LogicalResult WriteModule(mlir::ModuleOp module, llvm::StringRef path);
} // namespace coral::compiler
