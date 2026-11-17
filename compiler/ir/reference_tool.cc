// SPDX-License-Identifier: MIT
#include "reference.h"
#include "stablehlo_program.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <chrono>
#include <cmath>

namespace {
mlir::FailureOr<llvm::SmallVector<mlir::DenseElementsAttr>> ReadInputs(
    mlir::MLIRContext &context, llvm::StringRef path) {
    const auto buffer = llvm::MemoryBuffer::getFile(path);
    if (!buffer) {
        llvm::errs() << "cannot read input JSON\n";
        return mlir::failure();
    }
    auto parsed = llvm::json::parse(buffer.get()->getBuffer());
    if (!parsed) {
        llvm::errs() << llvm::toString(parsed.takeError()) << '\n';
        return mlir::failure();
    }
    auto *object = parsed->getAsObject();
    auto *inputs = object ? object->getArray("inputs") : nullptr;
    if (!inputs || inputs->size() > 16) {
        llvm::errs() << "expected inputs array with at most 16 tensors\n";
        return mlir::failure();
    }
    llvm::SmallVector<mlir::DenseElementsAttr> result;
    for (const auto &input : *inputs) {
        const auto *tensor = input.getAsObject();
        const auto *shape = tensor ? tensor->getArray("shape") : nullptr;
        const auto *values = tensor ? tensor->getArray("values") : nullptr;
        if (!shape || !values || tensor->getString("dtype") != "f32" || shape->size() > 4) {
            llvm::errs() << "each input requires dtype=f32, shape (rank <=4), and flat values\n";
            return mlir::failure();
        }
        llvm::SmallVector<int64_t> dimensions;
        int64_t count = 1;
        for (const auto &value : *shape) {
            auto dimension = value.getAsInteger();
            if (!dimension || *dimension <= 0 || *dimension > (1 << 20) / count) {
                llvm::errs() << "input shape exceeds the reference element budget\n";
                return mlir::failure();
            }
            count *= *dimension;
            dimensions.push_back(*dimension);
        }
        if (values->size() != static_cast<std::size_t>(count)) {
            llvm::errs() << "input value count differs from shape\n";
            return mlir::failure();
        }
        llvm::SmallVector<float> payload;
        for (const auto &value : *values) {
            const auto number = value.getAsNumber();
            if (!number || !std::isfinite(*number) ||
                !std::isfinite(static_cast<float>(*number))) {
                llvm::errs() << "input values must be finite F32 numbers\n";
                return mlir::failure();
            }
            payload.push_back(static_cast<float>(*number));
        }
        const auto type = mlir::RankedTensorType::get(
            dimensions, mlir::Float32Type::get(&context));
        result.push_back(mlir::DenseElementsAttr::get(type, llvm::ArrayRef<float>(payload)));
    }
    return result;
}
} // namespace

int main(int argc, char **argv) {
    if (argc != 5) {
        llvm::errs() << "Usage: coral-stablehlo-reference MODULE ENTRY INPUT_JSON NEW_OUTPUT_JSON\n";
        return 1;
    }
    mlir::MLIRContext context;
    auto module = coral::compiler::ReadModule(context, argv[1]);
    if (!module) {
        return 1;
    }
    auto inputs = ReadInputs(context, argv[3]);
    if (mlir::failed(inputs)) {
        return 1;
    }
    const auto start = std::chrono::steady_clock::now();
    auto results = coral::compiler::EvaluateReference(*module, argv[2], *inputs);
    const double elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    if (mlir::failed(results)) {
        return 1;
    }
    llvm::json::Array outputs;
    for (const auto tensor : *results) {
        llvm::json::Array shape;
        for (auto size : tensor.getType().getShape()) {
            shape.push_back(size);
        }
        llvm::json::Array values;
        for (const auto number : tensor.getValues<llvm::APFloat>()) {
            values.push_back(static_cast<double>(number.convertToFloat()));
        }
        outputs.push_back(llvm::json::Object{
            {"dtype", "f32"}, {"shape", std::move(shape)}, {"values", std::move(values)}});
    }
    llvm::json::Object report{
        {"schema", "coral-stablehlo-reference-v1"}, {"entry", argv[2]},
        {"evaluation_ms", elapsed_ms}, {"outputs", std::move(outputs)}};
    std::error_code error;
    llvm::raw_fd_ostream output(argv[4], error, llvm::sys::fs::CD_CreateNew,
                               llvm::sys::fs::FA_Write, llvm::sys::fs::OF_None);
    if (error) {
        llvm::errs() << "cannot create output: " << error.message() << '\n';
        return 1;
    }
    output << llvm::formatv("{0:2}", llvm::json::Value(std::move(report))) << '\n';
    output.close();
    if (output.has_error()) {
        output.clear_error();
        llvm::errs() << "failed to write reference output\n";
        return 1;
    }
    return 0;
}
