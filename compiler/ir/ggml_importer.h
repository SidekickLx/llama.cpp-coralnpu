// SPDX-License-Identifier: MIT
#pragma once
#include "contracts/ffn_contract.h"
#include <array>
#include "ggml.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/LogicalResult.h"

namespace coral::compiler {
// Borrowed source references only, not an owned computational IR. IDs are
// nonzero within the supplied graph snapshot and never encode addresses.
struct GGMLTensorRef {
    const ggml_tensor *tensor = nullptr;
    contracts::TensorId id;
};

struct GGMLImportRequest {
    contracts::GraphSnapshotId graph;
    llvm::StringRef entry_point = "imported";
    llvm::ArrayRef<GGMLTensorRef> inputs;
    llvm::ArrayRef<GGMLTensorRef> operations; // explicit topological order
    llvm::ArrayRef<GGMLTensorRef> outputs;    // explicit function result order
};

struct GGMLFFNImportRequest {
    contracts::GraphSnapshotId graph;
    contracts::FFNCompileSpec spec;
    // Fixed role order: X, gate weight, up weight, down weight. Repeated weight
    // identities are allowed but keep distinct positional function arguments.
    std::array<GGMLTensorRef, 4> inputs;
    llvm::ArrayRef<GGMLTensorRef> operations;
    GGMLTensorRef output;
};

// Imports the complete selected region with the fixed FFN signature and
// explicit semantic-role metadata. Reuses the per-op importer and verifier.
// It does not grant runtime/fusion authority or inspect physical storage.
mlir::OwningOpRef<mlir::ModuleOp> ImportGGMLFFNFunction(
    mlir::MLIRContext &context, const GGMLFFNImportRequest &request);

// Rejects unadmitted layouts/types without reading data or calling GGML kernels.
mlir::FailureOr<mlir::RankedTensorType> ImportGGMLTensorType(
    mlir::MLIRContext &context, mlir::Location loc, const ggml_tensor *tensor);

// Operands are in GGML src[] order, not a guessed semantic-role order. This
// low-level API lets a later FFN ABI binder assign values per operand edge.
// A validation failure emits no operations into the caller's block.
mlir::FailureOr<mlir::Value> ImportGGMLNode(
    mlir::OpBuilder &builder, mlir::Location loc, const ggml_tensor &node,
    llvm::ArrayRef<mlir::Value> operands);

// Imports exactly the selected operations. Unique boundary tensors become
// function arguments; their producers are not recursively absorbed. This is
// semantic conversion only, not matching, fusion legality, or placement proof.
mlir::OwningOpRef<mlir::ModuleOp> ImportGGMLFunction(
    mlir::MLIRContext &context, const GGMLImportRequest &request);
} // namespace coral::compiler
