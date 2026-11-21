// SPDX-License-Identifier: MIT
#include "runtime/ffn_runtime.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace {
using coral::contracts::FFNCompileSpec;
using coral::runtime::DeviceContext;
using coral::runtime::FFNModule;
using coral::runtime::PreparedFFNWeights;
using coral::runtime::TensorBinding;
using Clock = std::chrono::steady_clock;

void Require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}

struct Values {
    std::vector<float> gate, up, down;
};
void Fill(std::vector<float> &values, int seed, float scale) {
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(static_cast<int>((i * 37 + seed * 11) % 101) - 50) * scale;
    }
}
Values Weights(const FFNCompileSpec &spec, int seed) {
    const auto count = spec.shape.d * spec.shape.f;
    Values result{std::vector<float>(count), std::vector<float>(count), std::vector<float>(count)};
    Fill(result.gate, seed, 0.02f / std::sqrt(static_cast<float>(spec.shape.d)));
    Fill(result.up, seed + 7, 0.02f / std::sqrt(static_cast<float>(spec.shape.d)));
    Fill(result.down, seed + 17, 0.02f / std::sqrt(static_cast<float>(spec.shape.f)));
    return result;
}
PreparedFFNWeights Prepare(const DeviceContext &device, const FFNCompileSpec &spec, const Values &w) {
    return PreparedFFNWeights(device, spec,
                              TensorBinding::Dense(w.gate, spec.shape.f, spec.shape.d),
                              TensorBinding::Dense(w.up, spec.shape.f, spec.shape.d),
                              TensorBinding::Dense(w.down, spec.shape.d, spec.shape.f));
}

struct Comparison {
    double max_abs_error = 0;
    double max_tolerance_ratio = 0;
    double cpu_ms = 0;
    std::size_t calls = 0;
};
void Compare(const FFNCompileSpec &spec, const std::vector<float> &x, const Values &w,
             const std::vector<float> &actual, Comparison &summary) {
    const auto start = Clock::now();
    const auto &s = spec.shape;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(
        ggml_init({16 * 1024 * 1024, nullptr, false}), ggml_free);
    Require(context != nullptr, "GGML reference allocation");
    auto *input = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, s.d, s.m);
    auto *gate = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, s.d, s.f);
    auto *up = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, s.d, s.f);
    auto *down = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, s.f, s.d);
    std::memcpy(input->data, x.data(), x.size() * sizeof(float));
    std::memcpy(gate->data, w.gate.data(), w.gate.size() * sizeof(float));
    std::memcpy(up->data, w.up.data(), w.up.size() * sizeof(float));
    std::memcpy(down->data, w.down.data(), w.down.size() * sizeof(float));
    auto *g = ggml_mul_mat(context.get(), gate, input);
    auto *u = ggml_mul_mat(context.get(), up, input);
    auto *activation = ggml_swiglu_split(context.get(), g, u);
    auto *output = ggml_mul_mat(context.get(), down, activation);
    auto *graph = ggml_new_graph(context.get());
    ggml_build_forward_expand(graph, output);
    Require(ggml_graph_compute_with_ctx(context.get(), graph, 1) == GGML_STATUS_SUCCESS,
            "GGML CPU FFN reference failed");
    summary.cpu_ms += std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    Require(actual.size() == x.size(), "FFN output size");
    const auto *expected = static_cast<const float *>(output->data);
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double error = std::abs(static_cast<double>(actual[i]) - expected[i]);
        const double tolerance = 2e-5 + 2e-4 * std::abs(expected[i]);
        summary.max_abs_error = std::max(summary.max_abs_error, error);
        summary.max_tolerance_ratio = std::max(summary.max_tolerance_ratio, error / tolerance);
        Require(std::isfinite(actual[i]) && std::isfinite(expected[i]) && error <= tolerance,
                "complete Coral FFN disagrees with GGML CPU");
    }
    ++summary.calls;
}
void InvalidCall(FFNModule &module, TensorBinding x, const PreparedFFNWeights &weights) {
    bool rejected = false;
    try {
        module.Run(x, weights);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    Require(rejected, "invalid FFN call was accepted");
}
void InvalidWeights(const DeviceContext &device, const FFNCompileSpec &spec,
                    const Values &w, int bad_role) {
    TensorBinding bindings[] = {
        TensorBinding::Dense(w.gate, spec.shape.f, spec.shape.d),
        TensorBinding::Dense(w.up, spec.shape.f, spec.shape.d),
        TensorBinding::Dense(w.down, spec.shape.d, spec.shape.f)};
    --bindings[bad_role].element_count;
    bool rejected = false;
    try {
        PreparedFFNWeights bad(device, spec, bindings[0], bindings[1], bindings[2]);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    Require(rejected, "invalid FFN weights were accepted");
}

void RunProfile(const std::string &directory, const FFNCompileSpec &spec, bool detach) {
    Comparison comparison;
    for (int cycle = 0; cycle < 2; ++cycle) {
        auto device = std::make_unique<DeviceContext>();
        const auto constructor_start = Clock::now();
        FFNModule module(*device, directory, spec);
        const double constructor_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - constructor_start).count();
        const auto &s = spec.shape;
        auto host = Weights(spec, cycle + 1);
        const auto original = host;
        for (int role = 0; role < 3; ++role) InvalidWeights(*device, spec, host, role);
        Require(device->stats().binding_allocations == 0, "bad weights partially uploaded");
        auto prepared = Prepare(*device, spec, host);
        // Changing the borrowed storage must never change an existing snapshot.
        host = Weights(spec, cycle + 30);
        const auto changed = host;
        auto fresh = Prepare(*device, spec, host);
        host = {};
        std::vector<float> x(s.m * s.d);
        Fill(x, cycle + 2, 0.02f);
        auto binding = TensorBinding::Dense(x, s.m, s.d);
        auto invalid = binding;
        invalid.data = nullptr;
        InvalidCall(module, invalid, prepared);
        invalid = binding;
        --invalid.element_count;
        InvalidCall(module, invalid, prepared);
        invalid = binding;
        invalid.row_stride_bytes += sizeof(float);
        InvalidCall(module, invalid, prepared);
        invalid = binding;
        ++invalid.rows;
        InvalidCall(module, invalid, prepared);
        {
            DeviceContext other;
            auto other_weights = Prepare(other, spec, original);
            InvalidCall(module, binding, other_weights);
        }
        auto wrong_spec = spec;
        ++wrong_spec.shape.d;
        auto wrong = Prepare(*device, wrong_spec, Weights(wrong_spec, 1));
        InvalidCall(module, binding, wrong);
        Require(device->stats().calls_attempted == 0, "bad binding reached invocation");
        const auto allocations = device->stats().binding_allocations;
        auto output = module.Run(binding, prepared);
        Compare(spec, x, original, output, comparison);
        Fill(x, cycle + 10, 0.02f);
        output = module.Run(binding, prepared);
        Compare(spec, x, original, output, comparison);
        const auto original_output = output;
        output = module.Run(binding, fresh);
        Compare(spec, x, changed, output, comparison);
        Require(output != original_output, "distinct weights did not affect the result");
        output = module.Run(binding, prepared);
        Compare(spec, x, original, output, comparison);

        auto shared_values = original;
        shared_values.up = shared_values.gate;
        auto shared = PreparedFFNWeights(*device, spec,
            TensorBinding::Dense(shared_values.gate, s.f, s.d),
            TensorBinding::Dense(shared_values.gate, s.f, s.d),
            TensorBinding::Dense(shared_values.down, s.d, s.f));
        output = module.Run(binding, shared);
        Compare(spec, x, shared_values, output, comparison);

        FFNModule moved_module(std::move(module));
        PreparedFFNWeights moved_weights(std::move(prepared));
        InvalidCall(module, binding, moved_weights);
        InvalidCall(moved_module, binding, prepared);
        output = moved_module.Run(binding, moved_weights);
        Compare(spec, x, original, output, comparison);
        const auto stats = device->stats();
        Require(stats.module_loads == 1 && stats.calls_attempted == 6 &&
                stats.calls_completed == 6, "module or invocation reuse failed");
        Require(stats.binding_allocations == allocations + 4, "activation allocation was not reused");
        const auto weight_elements = 9 * s.d * s.f + 3 * wrong_spec.shape.d * s.f;
        Require(stats.upload_bytes == static_cast<std::uint64_t>((weight_elements + 6 * s.m * s.d) * 4),
                "weight snapshot or activation copy accounting mismatch");
        Require(stats.download_bytes == static_cast<std::uint64_t>(6 * s.m * s.d * 4),
                "output copy accounting mismatch");
        Require(stats.coral_dispatches > 0 &&
                stats.dispatches_with_instruction_count == stats.coral_dispatches &&
                stats.kernel_instructions > 0 && stats.host_executable_load_attempts == 0 &&
                stats.failed_dispatches == 0 && stats.observer_errors == 0,
                "incomplete device evidence");

        // The last cycle can hide a harness-owned copy after load. Run must
        // use owned VMFB bytes and retain the device after its wrapper dies.
        if (detach && cycle == 1) std::filesystem::rename(directory, directory + ".detached");
        device.reset();
        output = moved_module.Run(binding, moved_weights);
        Compare(spec, x, original, output, comparison);
        std::cout << std::setprecision(12)
                  << "{\"status\":\"passed\",\"cycle\":" << cycle
                  << ",\"m\":" << s.m << ",\"d\":" << s.d << ",\"f\":" << s.f
                  << ",\"calls_with_stats\":" << stats.calls_completed
                  << ",\"context_lifetime_call\":true"
                  << ",\"max_abs_error\":" << comparison.max_abs_error
                  << ",\"max_tolerance_ratio\":" << comparison.max_tolerance_ratio
                  << ",\"coral_dispatches\":" << stats.coral_dispatches
                  << ",\"instruction_samples\":" << stats.dispatches_with_instruction_count
                  << ",\"kernel_instructions\":" << stats.kernel_instructions
                  << ",\"host_loads\":" << stats.host_executable_load_attempts
                  << ",\"binding_allocations\":" << stats.binding_allocations
                  << ",\"upload_bytes\":" << stats.upload_bytes
                  << ",\"download_bytes\":" << stats.download_bytes
                  << ",\"constructor_ms\":" << constructor_ms
                  << ",\"module_load_ms\":" << stats.module_load_ms
                  << ",\"invoke_ms\":" << stats.invoke_ms
                  << ",\"cpu_reference_ms_cumulative\":" << comparison.cpu_ms << "}\n";
    }
}

void CrossProfile(const std::string &first, const FFNCompileSpec &spec,
                  std::int64_t other_m, const std::string &second) {
    DeviceContext device;
    auto other = spec;
    other.shape.m = other_m;
    FFNModule decode(device, first, spec);
    FFNModule prefill(device, second, other);
    auto values = Weights(spec, 1);
    auto weights = Prepare(device, spec, values);
    Comparison comparison;
    std::vector<float> x(spec.shape.m * spec.shape.d, 0.15f);
    auto output = decode.Run(TensorBinding::Dense(x, spec.shape.m, spec.shape.d), weights);
    Compare(spec, x, values, output, comparison);
    x.assign(other.shape.m * other.shape.d, 0.25f);
    output = prefill.Run(TensorBinding::Dense(x, other.shape.m, other.shape.d), weights);
    Compare(other, x, values, output, comparison);
    Require(device.stats().module_loads == 2 && device.stats().binding_allocations == 5 &&
            device.stats().calls_completed == 2, "cross-profile weights were not reused");
    std::cout << "{\"status\":\"passed\",\"cross_profile_weight_reuse\":true}\n";
}

void InvalidRows(FFNModule &module, TensorBinding binding,
                 const PreparedFFNWeights &weights) {
    bool rejected = false;
    try {
        module.RunRows(binding, weights);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    Require(rejected, "invalid row boundary was accepted");
}

void Bucket(const std::string &directory, FFNCompileSpec spec, const std::string &layout) {
    Require(spec.shape.m == 8, "bucket fixture requires capacity 8");
    if (layout == "transposed")
        spec.weight_layout = coral::contracts::FFNWeightLayout::Transposed;
    else Require(layout == "native", "unknown test layout");
    DeviceContext device;
    FFNModule module(device, directory, spec);
    const auto original = Weights(spec, 2);
    auto prepared = Prepare(device, spec, original);
    std::vector<float> x(9 * spec.shape.d);
    auto invalid = TensorBinding::Dense(x, 9, spec.shape.d);
    InvalidRows(module, invalid, prepared);
    invalid.rows = 0;
    InvalidRows(module, invalid, prepared);
    x.resize(spec.shape.d);
    invalid = TensorBinding::Dense(x, 1, spec.shape.d);
    invalid.data = nullptr;
    InvalidRows(module, invalid, prepared);
    invalid = TensorBinding::Dense(x, 1, spec.shape.d);
    --invalid.element_count;
    InvalidRows(module, invalid, prepared);
    invalid = TensorBinding::Dense(x, 1, spec.shape.d);
    invalid.row_stride_bytes += sizeof(float);
    InvalidRows(module, invalid, prepared);
    InvalidCall(module, TensorBinding::Dense(x, 1, spec.shape.d), prepared);
    // The new boundary API cannot silently use weights from the other ABI,
    // even when the physical dimensions happen to be square.
    auto wrong_layout = spec;
    wrong_layout.weight_layout = layout == "native"
        ? coral::contracts::FFNWeightLayout::Transposed : coral::contracts::FFNWeightLayout::Native;
    auto wrong = Prepare(device, wrong_layout, original);
    InvalidRows(module, TensorBinding::Dense(x, 1, spec.shape.d), wrong);
    DeviceContext other;
    auto foreign = Prepare(other, spec, original);
    InvalidRows(module, TensorBinding::Dense(x, 1, spec.shape.d), foreign);
    Require(device.stats().calls_attempted == 0, "bad boundary reached VM execution");
    Comparison comparison;
    std::uint64_t padded = 0;
    // Reuse one module and owning weight set across growing and shrinking M.
    for (const auto rows : {8, 1, 7, 2, 5, 1}) {
        auto logical = spec;
        logical.shape.m = rows;
        x.resize(rows * spec.shape.d);
        Fill(x, rows, 0.02f);
        const auto output = module.RunRows(TensorBinding::Dense(x, rows, spec.shape.d), prepared);
        Compare(logical, x, original, output, comparison);
        padded += 8 - rows;
    }
    // An existing snapshot remains immutable; replacement weights require
    // explicit preparation, but reuse the same compiled module.
    const auto changed = Weights(spec, 7);
    auto fresh = Prepare(device, spec, changed);
    auto logical = spec;
    logical.shape.m = 1;
    const auto changed_output = module.RunRows(TensorBinding::Dense(x, 1, spec.shape.d), fresh);
    Compare(logical, x, changed, changed_output, comparison);
    padded += 7;
    const auto stats = device.stats();
    Require(stats.module_loads == 1 && stats.calls_completed == 7 &&
            stats.calls_attempted == 7 && stats.ffn_padded_rows == padded &&
            stats.ffn_cropped_rows == padded && stats.host_executable_load_attempts == 0 &&
            stats.failed_dispatches == 0 && stats.observer_errors == 0, "bucket reuse accounting failed");
    std::cout << "{\"status\":\"passed\",\"calls\":7,\"negative_cases\":8,\"padded_rows\":"
              << padded << ",\"max_abs_error\":" << comparison.max_abs_error << "}\n";
}

void Reject(const std::string &directory, const FFNCompileSpec &spec,
            const std::string &message, const std::string &valid_directory, bool poison) {
    DeviceContext device;
    bool rejected = false;
    if (!poison) {
        try {
            FFNModule bad(device, directory, spec);
        } catch (const std::runtime_error &error) {
            std::cerr << error.what() << '\n';
            rejected = std::string(error.what()).find(message) != std::string::npos;
        }
        Require(rejected, "artifact failed at an unexpected gate or was accepted");
        const auto stats = device.stats();
        Require(stats.module_loads == 0 && stats.calls_attempted == 0 &&
                stats.coral_dispatches == 0 && stats.host_executable_load_attempts == 0,
                "bad artifact reached execution");
        FFNModule good(device, valid_directory, spec);
    } else {
        FFNModule bad(device, directory, spec);
        auto values = Weights(spec, 1);
        auto weights = Prepare(device, spec, values);
        std::vector<float> x(spec.shape.m * spec.shape.d, 0.1f);
        auto binding = TensorBinding::Dense(x, spec.shape.m, spec.shape.d);
        try {
            bad.Run(binding, weights);
        } catch (const std::runtime_error &error) {
            std::cerr << error.what() << '\n';
            rejected = std::string(error.what()).find(message) != std::string::npos;
        }
        Require(rejected && device.stats().calls_attempted == 1 &&
                device.stats().calls_completed == 0, "execution failure exposed a result");
        rejected = false;
        try {
            bad.Run(binding, weights);
        } catch (const std::runtime_error &error) {
            std::cerr << error.what() << '\n';
            rejected = true;
        }
        Require(rejected && device.stats().calls_attempted == 1, "poisoned context retried");
    }
    const auto stats = device.stats();
    std::cout << "{\"status\":\"passed\",\"negative_gate\":true"
              << ",\"calls_attempted\":" << stats.calls_attempted
              << ",\"calls_completed\":" << stats.calls_completed
              << ",\"coral_dispatches\":" << stats.coral_dispatches
              << ",\"host_loads\":" << stats.host_executable_load_attempts << "}\n";
}
} // namespace

int main(int argc, char **argv) {
    try {
        if (argc != 6 && argc != 7 && argc != 8) throw std::invalid_argument(
            "Usage: ffn-runtime-test run|owned ARTIFACT M D F; reject|poison ARTIFACT M D F MESSAGE VALID");
        FFNCompileSpec spec;
        spec.shape = {std::stoll(argv[3]), std::stoll(argv[4]), std::stoll(argv[5])};
        coral::contracts::ValidateFFNSpec(spec);
        const std::string mode(argv[1]);
        if (argc == 6 && (mode == "run" || mode == "owned")) {
            RunProfile(argv[2], spec, mode == "owned");
        } else if (argc == 7 && mode == "bucket") {
            Bucket(argv[2], spec, argv[6]);
        } else if (argc == 8 && mode == "cross") {
            CrossProfile(argv[2], spec, std::stoll(argv[6]), argv[7]);
        } else if (argc == 8 && (mode == "reject" || mode == "poison")) {
            Reject(argv[2], spec, argv[6], argv[7], mode == "poison");
        } else throw std::invalid_argument("invalid test mode");
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
