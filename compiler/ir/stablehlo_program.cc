// SPDX-License-Identifier: MIT
#include "stablehlo_program.h"
#include "stablehlo_ops.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "stablehlo/dialect/StablehloOps.h"
#include <limits>

namespace coral::compiler {
namespace {
using namespace mlir;
namespace hlo = mlir::stablehlo;

bool ValidMatrix(int64_t rows, int64_t columns) {
    return rows > 0 && columns > 0 &&
           rows <= std::numeric_limits<int64_t>::max() / columns;
}

} // namespace

void RegisterDialects(mlir::MLIRContext &context) {
    context.loadDialect<mlir::func::FuncDialect, mlir::stablehlo::StablehloDialect>();
}

mlir::LogicalResult VerifyModule(mlir::ModuleOp module) {
    // MLIR handles SSA/function structure; StableHLO handles operation types,
    // shapes, dimension numbers, broadcasts and result consistency.
    return mlir::verify(module);
}

mlir::OwningOpRef<mlir::ModuleOp> BuildMatmul(
    mlir::MLIRContext &context, int64_t m, int64_t k, int64_t n) {
    RegisterDialects(context);
    mlir::OpBuilder builder(&context);
    const auto loc = builder.getUnknownLoc();
    if (!ValidMatrix(m, k) || !ValidMatrix(n, k) || !ValidMatrix(m, n)) {
        mlir::emitError(loc, "matmul dimensions must be positive and have representable element counts");
        return {};
    }
    mlir::OwningOpRef<mlir::ModuleOp> module(mlir::ModuleOp::create(loc));
    const auto x = mlir::RankedTensorType::get({m, k}, builder.getF32Type());
    const auto w = mlir::RankedTensorType::get({n, k}, builder.getF32Type());
    const auto y = mlir::RankedTensorType::get({m, n}, builder.getF32Type());
    auto function = mlir::func::FuncOp::create(loc, "matmul", builder.getFunctionType({x, w}, {y}));
    module->push_back(function);
    auto *block = function.addEntryBlock();
    builder.setInsertionPointToStart(block);
    const auto result = BuildDenseDot(builder, loc, y, block->getArgument(0), block->getArgument(1));
    builder.create<mlir::func::ReturnOp>(loc, result);
    if (mlir::failed(VerifyModule(*module))) {
        return {};
    }
    return module;
}

mlir::OwningOpRef<mlir::ModuleOp> BuildFFN(
    mlir::MLIRContext &context, const contracts::FFNCompileSpec &spec) {
    RegisterDialects(context);
    mlir::OpBuilder builder(&context);
    const auto loc = builder.getUnknownLoc();
    const auto &shape = spec.shape;
    if (spec.contract_version != contracts::kFFNContractVersion ||
        spec.numeric_policy != contracts::FFNNumericPolicy::F32NoFastMathV1 ||
        !ValidMatrix(shape.m, shape.d) || !ValidMatrix(shape.f, shape.d) ||
        !ValidMatrix(shape.m, shape.f)) {
        mlir::emitError(loc, "unsupported FFN contract, numeric policy, or dimensions");
        return {};
    }
    mlir::OwningOpRef<mlir::ModuleOp> module(mlir::ModuleOp::create(loc));
    const auto f32 = builder.getF32Type();
    const auto x = mlir::RankedTensorType::get({shape.m, shape.d}, f32);
    const auto weight = mlir::RankedTensorType::get({shape.f, shape.d}, f32);
    const auto down_weight = mlir::RankedTensorType::get({shape.d, shape.f}, f32);
    const auto hidden_type = mlir::RankedTensorType::get({shape.m, shape.f}, f32);
    auto function = mlir::func::FuncOp::create(
        loc, contracts::kFFNEntryPoint,
        builder.getFunctionType({x, weight, weight, down_weight}, {x}));
    function->setAttr("coral.ffn_abi", builder.getStringAttr(contracts::kFFNAbi));
    function->setAttr("coral.numeric_policy", builder.getStringAttr("f32_no_fast_math_v1"));
    module->push_back(function);
    auto *block = function.addEntryBlock();
    builder.setInsertionPointToStart(block);
    const auto gate = BuildDenseDot(builder, loc, hidden_type, block->getArgument(0), block->getArgument(1));
    const auto up = BuildDenseDot(builder, loc, hidden_type, block->getArgument(0), block->getArgument(2));
    const auto activated_gate = BuildSilu(builder, loc, gate);
    const mlir::Value hidden = builder.create<mlir::stablehlo::MulOp>(loc, activated_gate, up);
    const auto result = BuildDenseDot(builder, loc, x, hidden, block->getArgument(3));
    builder.create<mlir::func::ReturnOp>(loc, result);
    if (mlir::failed(VerifyModule(*module))) {
        return {};
    }
    return module;
}

mlir::OwningOpRef<mlir::ModuleOp> ReadModule(
    mlir::MLIRContext &context, llvm::StringRef path) {
    RegisterDialects(context);
    auto module = mlir::parseSourceFile<mlir::ModuleOp>(path, &context);
    if (!module || mlir::failed(VerifyModule(*module))) {
        return {};
    }
    return module;
}

mlir::LogicalResult WriteModule(mlir::ModuleOp module, llvm::StringRef path) {
    if (mlir::failed(VerifyModule(module))) {
        return mlir::failure();
    }
    std::error_code error;
    // Do not overwrite earlier evidence or another generated module.
    llvm::raw_fd_ostream output(path, error, llvm::sys::fs::CD_CreateNew,
                               llvm::sys::fs::FA_Write, llvm::sys::fs::OF_None);
    if (error) {
        return module.emitError() << "cannot create output: " << error.message();
    }
    module.print(output);
    output << '\n';
    output.close();
    if (output.has_error()) {
        output.clear_error();
        return module.emitError("failed to write module");
    }
    return mlir::success();
}
} // namespace coral::compiler
