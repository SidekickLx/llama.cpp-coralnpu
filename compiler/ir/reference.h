// SPDX-License-Identifier: MIT
#pragma once
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace coral::compiler {
// Diagnostic execution of a small, static F32 module using upstream StableHLO.
// The input module is cloned: upstream interpreter preprocessing cannot mutate
// the caller's verified program. No runtime fallback or backend is involved.
mlir::FailureOr<llvm::SmallVector<mlir::DenseElementsAttr>> EvaluateReference(
    mlir::ModuleOp module, llvm::StringRef entry,
    llvm::ArrayRef<mlir::DenseElementsAttr> inputs);
} // namespace coral::compiler
