// SPDX-License-Identifier: MIT
#include "compiler/ir/ggml_importer.h"
#include "compiler/ir/reference.h"
#include "compiler/ir/stablehlo_program.h"
#include "tests/compiler/ggml_fixture.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <string>

namespace {
using coral::compiler::GGMLTensorRef;
using Clock = std::chrono::steady_clock;

bool WriteJSON(const std::string &path, llvm::json::Value value) {
    std::error_code error;
    llvm::raw_fd_ostream output(path, error, llvm::sys::fs::CD_CreateNew,
                               llvm::sys::fs::FA_Write, llvm::sys::fs::OF_None);
    if (error) {
        llvm::errs() << error.message() << '\n';
        return false;
    }
    output << llvm::formatv("{0:2}", value) << '\n';
    output.close();
    const bool ok = !output.has_error();
    output.clear_error();
    return ok;
}

llvm::json::Object TensorJSON(mlir::DenseElementsAttr tensor) {
    llvm::json::Array shape, values;
    for (auto size : tensor.getType().getShape()) {
        shape.push_back(size);
    }
    for (const auto value : tensor.getValues<llvm::APFloat>()) {
        values.push_back(static_cast<double>(value.convertToFloat()));
    }
    return llvm::json::Object{{"dtype", "f32"}, {"shape", std::move(shape)}, {"values", std::move(values)}};
}

mlir::DenseElementsAttr Payload(mlir::MLIRContext &context, const ggml_tensor *tensor) {
    // Test-only readback from the CPU fixture. The production importer never
    // accesses data, buffers, or backend APIs.
    const auto type = mlir::RankedTensorType::get(
        {tensor->ne[1], tensor->ne[0]}, mlir::Float32Type::get(&context));
    const auto *values = static_cast<const float *>(tensor->data);
    return mlir::DenseElementsAttr::get(
        type, llvm::ArrayRef<float>(values, type.getNumElements()));
}

std::string ProgramText(mlir::ModuleOp module) {
    std::string text;
    llvm::raw_string_ostream output(text);
    module.print(output);
    return text;
}
} // namespace

static int CheckImport(const CoralGGMLFixture *fixture, const char *output_directory, bool ffn) {
    if (!fixture || fixture->version != 1 || !fixture->name || !output_directory) {
        llvm::errs() << "invalid fixture ABI\n";
        return 1;
    }
    mlir::MLIRContext context;
    llvm::SmallVector<GGMLTensorRef> inputs, operations, outputs;
    std::map<const ggml_tensor *, uint64_t> identities;
    for (std::size_t i = 0; i < fixture->input_count; ++i) {
        const auto id = identities.size() + 1;
        const auto inserted = identities.emplace(fixture->inputs[i], id);
        inputs.push_back({fixture->inputs[i], {inserted.first->second}});
    }
    for (std::size_t i = 0; i < fixture->operation_count; ++i) {
        const auto id = identities.size() + 1;
        identities.emplace(fixture->operations[i], id);
        operations.push_back({fixture->operations[i], {id}});
    }
    for (std::size_t i = 0; i < fixture->output_count; ++i) {
        outputs.push_back({fixture->outputs[i], {identities[fixture->outputs[i]]}});
    }
    coral::compiler::GGMLImportRequest request{
        {1, 1}, "imported", inputs, operations, outputs};
    const llvm::StringRef name(fixture->name);
    if (name == "bad-id-zero") { inputs[0].id.value = 0; }
    if (name == "bad-id-duplicate") { operations[0].id = inputs[0].id; }
    if (name == "bad-output-id") { outputs[0].id.value += 100; }
    if (name == "bad-graph-id") { request.graph.revision = 0; }

    const auto import_start = Clock::now();
    mlir::OwningOpRef<mlir::ModuleOp> module;
    const char *entry = ffn ? coral::contracts::kFFNEntryPoint : "imported";
    if (ffn) {
        if (inputs.size() != 4 || outputs.size() != 1) {
            llvm::errs() << "FFN fixture requires four roles and one output\n";
            return 1;
        }
        coral::compiler::GGMLFFNImportRequest region;
        region.graph = request.graph;
        region.spec.shape = {inputs[0].tensor->ne[1], inputs[0].tensor->ne[0],
                             inputs[1].tensor->ne[1]};
        std::copy(inputs.begin(), inputs.end(), region.inputs.begin());
        region.operations = operations;
        region.output = outputs.front();
        if (name == "bad-region-contract") { ++region.spec.contract_version; }
        if (name == "bad-region-shape") { ++region.spec.shape.d; }
        if (name == "bad-region-policy") {
            region.spec.numeric_policy = static_cast<coral::contracts::FFNNumericPolicy>(99);
        }
        if (name == "bad-region-shared-id") { ++region.inputs[2].id.value; }
        module = coral::compiler::ImportGGMLFFNFunction(context, region);
    } else {
        module = coral::compiler::ImportGGMLFunction(context, request);
    }
    const double import_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - import_start).count();
    const std::string directory(output_directory);
    if (!module) {
        WriteJSON(directory + "/report.json", llvm::json::Object{
            {"status", "rejected"}, {"case", fixture->name}});
        return 2;
    }
    if (mlir::failed(coral::compiler::WriteModule(*module, directory + "/imported.mlir"))) {
        return 1;
    }
    const auto original_text = ProgramText(*module);
    llvm::SmallVector<mlir::DenseElementsAttr> payloads;
    llvm::json::Array input_json, expected_json;
    for (const auto &ref : inputs) {
        auto payload = Payload(context, ref.tensor);
        payloads.push_back(payload);
        input_json.push_back(TensorJSON(payload));
    }
    for (const auto &ref : outputs) {
        expected_json.push_back(TensorJSON(Payload(context, ref.tensor)));
    }
    if (!WriteJSON(directory + "/inputs.json",
                   llvm::json::Object{{"inputs", std::move(input_json)}}) ||
        !WriteJSON(directory + "/ggml-output.json",
                   llvm::json::Object{{"outputs", std::move(expected_json)}})) {
        return 1;
    }
    const auto reference_start = Clock::now();
    auto results = coral::compiler::EvaluateReference(*module, entry, payloads);
    const double reference_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - reference_start).count();
    if (mlir::failed(results) || results->size() != outputs.size() ||
        ProgramText(*module) != original_text) {
        llvm::errs() << "reference execution failed or mutated the imported program\n";
        return 1;
    }
    llvm::json::Array reference_json, comparisons;
    std::size_t failed_values = 0;
    for (std::size_t output = 0; output < results->size(); ++output) {
        const auto tensor = (*results)[output];
        reference_json.push_back(TensorJSON(tensor));
        const auto *expected = static_cast<const float *>(outputs[output].tensor->data);
        double max_abs = 0, max_rel = 0;
        llvm::json::Array failing_indices;
        std::size_t index = 0;
        for (const auto value : tensor.getValues<llvm::APFloat>()) {
            const double actual = value.convertToFloat();
            const double target = expected[index];
            const double absolute = std::abs(actual - target);
            max_abs = std::max(max_abs, absolute);
            if (target != 0) {
                max_rel = std::max(max_rel, absolute / std::abs(target));
            }
            if (!std::isfinite(actual) || !std::isfinite(target) ||
                absolute > 1e-4 + 1e-4 * std::abs(target)) {
                failing_indices.push_back(static_cast<int64_t>(index));
                ++failed_values;
            }
            ++index;
        }
        comparisons.push_back(llvm::json::Object{
            {"tensor_id", static_cast<int64_t>(outputs[output].id.value)},
            {"elements", static_cast<int64_t>(index)},
            {"max_abs_error", max_abs}, {"max_rel_error_nonzero_reference", max_rel},
            {"failing_indices", std::move(failing_indices)}});
    }
    if (!WriteJSON(directory + "/reference-output.json",
                   llvm::json::Object{{"outputs", std::move(reference_json)}}) ||
        !WriteJSON(directory + "/report.json", llvm::json::Object{
            {"status", failed_values ? "failed" : "passed"}, {"case", fixture->name},
            {"atol", 1e-4}, {"rtol", 1e-4},
            {"import_ms", import_ms}, {"reference_ms", reference_ms},
            {"comparisons", std::move(comparisons)}})) {
        return 1;
    }
    return failed_values ? 1 : 0;
}

extern "C" __attribute__((visibility("default")))
int coral_check_ggml_import(const CoralGGMLFixture *fixture, const char *directory) {
    return CheckImport(fixture, directory, false);
}

extern "C" __attribute__((visibility("default")))
int coral_check_ggml_ffn_import(const CoralGGMLFixture *fixture, const char *directory) {
    return CheckImport(fixture, directory, true);
}
