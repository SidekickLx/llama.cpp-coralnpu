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
mlir::LogicalResult PrepareFFNAOT(mlir::ModuleOp, const contracts::FFNCompileSpec &,
                                  llvm::StringRef ir_path, llvm::StringRef metadata_path);
mlir::LogicalResult OptimizeFFN(mlir::ModuleOp, const contracts::FFNCompileSpec &,
                                llvm::StringRef ir_path, llvm::StringRef plan_path);
mlir::LogicalResult SpecializeFFNRows(mlir::ModuleOp, const contracts::FFNCompileSpec &,
                                      int64_t capacity, llvm::StringRef core_path,
                                      llvm::StringRef boundary_path, llvm::StringRef plan_path);
mlir::LogicalResult BindFFNReflection(mlir::ModuleOp, llvm::StringRef reflection,
                                      llvm::StringRef output);
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
