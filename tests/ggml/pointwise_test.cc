// SPDX-License-Identifier: MIT
#include "runtime/matmul_runtime.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace {
void Require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}
using coral::contracts::PointwiseKind;
using coral::contracts::PointwiseSpec;
using coral::runtime::TensorBinding;

void InvalidBinding(coral::runtime::PointwiseModule &module,
                    const std::vector<TensorBinding> &bindings) {
    bool rejected = false;
    try {
        module.Run(bindings);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    Require(rejected, "malformed binding accepted");
}

int Run(int argc, char **argv) {
    if (argc != 5) throw std::invalid_argument("Usage: pointwise-test KIND M N VMFB");
    PointwiseSpec spec;
    const std::string kind(argv[1]);
    if (kind == "silu") spec.kind = PointwiseKind::Silu;
    else if (kind == "multiply") spec.kind = PointwiseKind::Multiply;
    else if (kind == "swiglu") spec.kind = PointwiseKind::SwiGLU;
    else throw std::invalid_argument("unknown operation");
    spec.rows = std::stoull(argv[2]);
    spec.cols = std::stoull(argv[3]);
    const auto count = spec.Elements();
    double max_error = 0;
    // Context recreation also exercises VMFB ownership and simulator teardown.
    for (int cycle = 0; cycle < 2; ++cycle) {
        std::unique_ptr<ggml_context, decltype(&ggml_free)> context(
            ggml_init({4 * 1024 * 1024, nullptr, false}), ggml_free);
        Require(context != nullptr, "GGML context allocation");
        auto *a = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, spec.cols, spec.rows);
        auto *b = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, spec.cols, spec.rows);
        ggml_tensor *output = nullptr;
        if (spec.kind == PointwiseKind::Silu) output = ggml_silu(context.get(), a);
        else if (spec.kind == PointwiseKind::Multiply) output = ggml_mul(context.get(), a, b);
        else output = ggml_swiglu_split(context.get(), a, b);
        auto *graph = ggml_new_graph(context.get());
        ggml_build_forward_expand(graph, output);
        coral::runtime::DeviceContext device;
        coral::runtime::PointwiseModule module(device, argv[4], spec);
        std::vector<float> left(count), right(count);
        std::vector<TensorBinding> bindings;
        for (std::size_t i = 0; i < spec.Arity(); ++i) {
            bindings.push_back(TensorBinding::Dense(left, spec.rows, spec.cols));
        }
        auto invalid = bindings;
        invalid.pop_back();
        InvalidBinding(module, invalid);
        invalid = bindings;
        --invalid.back().element_count;
        InvalidBinding(module, invalid);
        invalid = bindings;
        invalid.back().row_stride_bytes += sizeof(float);
        InvalidBinding(module, invalid);
        invalid = bindings;
        invalid.front().data = nullptr;
        InvalidBinding(module, invalid);
        Require(device.stats().calls_attempted == 0, "invalid binding started execution");
        for (int repeat = 0; repeat < 4; ++repeat) {
            for (std::size_t i = 0; i < count; ++i) {
                left[i] = static_cast<float>(static_cast<int>((i * 37 + repeat * 11) % 401) - 200) / 10;
                right[i] = static_cast<float>(static_cast<int>((i * 13 + repeat * 7) % 97) - 48) / 13;
            }
            std::memcpy(a->data, left.data(), count * sizeof(float));
            std::memcpy(b->data, right.data(), count * sizeof(float));
            Require(ggml_graph_compute_with_ctx(context.get(), graph, 1) == GGML_STATUS_SUCCESS,
                    "GGML CPU reference execution");
            bindings[0] = TensorBinding::Dense(left, spec.rows, spec.cols);
            if (spec.Arity() == 2) bindings[1] = TensorBinding::Dense(right, spec.rows, spec.cols);
            const auto actual = module.Run(bindings);
            const auto *expected = static_cast<const float *>(output->data);
            for (std::size_t i = 0; i < count; ++i) {
                const double error = std::abs(static_cast<double>(actual[i]) - expected[i]);
                max_error = std::max(max_error, error);
                Require(std::isfinite(actual[i]) && error <= 2e-6 + 1e-5 * std::abs(expected[i]),
                        "Coral versus GGML CPU mismatch");
            }
        }
        const auto stats = device.stats();
        Require(stats.calls_completed == 4 && stats.calls_attempted == 4, "call counts");
        Require(stats.module_loads == 1 && stats.binding_allocations == spec.Arity(), "buffer reuse");
        Require(stats.host_executable_load_attempts == 0 && stats.failed_dispatches == 0 &&
                stats.observer_errors == 0, "unexpected host load or dispatch failure");
        Require(stats.upload_bytes == 4 * spec.Arity() * count * sizeof(float), "stale input upload");
        std::cout << std::setprecision(12)
                  << "{\"status\":\"passed\",\"op\":\"" << kind << "\",\"rows\":" << spec.rows
                  << ",\"cols\":" << spec.cols << ",\"cycle\":" << cycle
                  << ",\"max_abs_error\":" << max_error
                  << ",\"calls\":" << stats.calls_completed
                  << ",\"dispatches\":" << stats.coral_dispatches
                  << ",\"instruction_samples\":" << stats.dispatches_with_instruction_count
                  << ",\"kernel_instructions\":" << stats.kernel_instructions
                  << ",\"host_loads\":" << stats.host_executable_load_attempts
                  << ",\"invoke_ms\":" << stats.invoke_ms << "}\n";
    }
    return 0;
}
} // namespace
int main(int argc, char **argv) {
    try {
        return Run(argc, argv);
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
