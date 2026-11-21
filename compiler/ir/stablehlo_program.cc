// SPDX-License-Identifier: MIT
#include "stablehlo_program.h"
#include "stablehlo_ops.h"
#include "contracts/ffn_artifact.h"
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

mlir::OwningOpRef<mlir::ModuleOp> BuildPointwise(
    mlir::MLIRContext &context, const contracts::PointwiseSpec &spec) {
    RegisterDialects(context);
    mlir::OpBuilder builder(&context);
    const auto loc = builder.getUnknownLoc();
    // The CLI validates the bounded static ABI before entering this builder.
    // Recheck here for callers using the library directly.
    constexpr std::size_t limit = 2147483647 / sizeof(float);
    if (!spec.rows || !spec.cols || spec.cols > limit || spec.rows > limit / spec.cols ||
        (spec.kind != contracts::PointwiseKind::Silu &&
         spec.kind != contracts::PointwiseKind::Multiply &&
         spec.kind != contracts::PointwiseKind::SwiGLU)) {
        mlir::emitError(loc, "invalid pointwise specification");
        return {};
    }
    mlir::OwningOpRef<mlir::ModuleOp> module(mlir::ModuleOp::create(loc));
    const auto type = mlir::RankedTensorType::get(
        {static_cast<int64_t>(spec.rows), static_cast<int64_t>(spec.cols)}, builder.getF32Type());
    llvm::SmallVector<mlir::Type> inputs(spec.Arity(), type);
    auto function = mlir::func::FuncOp::create(
        loc, spec.EntryPoint(), builder.getFunctionType(inputs, {type}));
    function->setAttr("iree.reflection", builder.getDictionaryAttr({
        builder.getNamedAttr(contracts::kPointwiseAbiKey, builder.getStringAttr(spec.AbiTag()))}));
    module->push_back(function);
    auto *block = function.addEntryBlock();
    builder.setInsertionPointToStart(block);
    mlir::Value result = block->getArgument(0);
    if (spec.kind != contracts::PointwiseKind::Multiply) {
        result = BuildSilu(builder, loc, result);
    }
    if (spec.kind != contracts::PointwiseKind::Silu) {
        result = builder.create<mlir::stablehlo::MulOp>(loc, result, block->getArgument(1));
    }
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
    const bool transposed = spec.weight_layout == contracts::FFNWeightLayout::Transposed;
    const auto rows = transposed ? shape.d : shape.f;
    const auto cols = transposed ? shape.f : shape.d;
    const auto weight = mlir::RankedTensorType::get({rows, cols}, f32);
    const auto down_weight = mlir::RankedTensorType::get({cols, rows}, f32);
    const auto hidden_type = mlir::RankedTensorType::get({shape.m, shape.f}, f32);
    auto function = mlir::func::FuncOp::create(
        loc, contracts::kFFNEntryPoint,
        builder.getFunctionType({x, weight, weight, down_weight}, {x}));
    function->setAttr("coral.ffn_abi", builder.getStringAttr(contracts::FFNAbi(spec)));
    function->setAttr("coral.numeric_policy", builder.getStringAttr("f32_no_fast_math_v1"));
    module->push_back(function);
    auto *block = function.addEntryBlock();
    builder.setInsertionPointToStart(block);
    const auto gate = BuildDenseDot(builder, loc, hidden_type, block->getArgument(0), block->getArgument(1), transposed);
    const auto up = BuildDenseDot(builder, loc, hidden_type, block->getArgument(0), block->getArgument(2), transposed);
    const auto activated_gate = BuildSilu(builder, loc, gate);
    const mlir::Value hidden = builder.create<mlir::stablehlo::MulOp>(loc, activated_gate, up);
    const auto result = BuildDenseDot(builder, loc, x, hidden, block->getArgument(3), transposed);
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
