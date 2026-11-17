// SPDX-License-Identifier: MIT
#include "ggml/ggml_coral.h"
#include "support/json.h"
#include "ggml-alloc.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace adapter = coral::ggml_backend;
using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
using Backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
using Scheduler = std::unique_ptr<ggml_backend_sched, decltype(&ggml_backend_sched_free)>;

void Require(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

Context NewContext() {
    Context context(ggml_init({4 * 1024 * 1024, nullptr, true}), ggml_free);
    Require(context != nullptr, "create GGML context");
    return context;
}

struct Evidence {
    std::vector<std::string> checks;
    std::vector<std::string> placements;
    std::vector<adapter::NodeEvent> events;
    double max_error = 0;
    std::uint64_t expected_failures = 0;

    static void OnCoral(const adapter::NodeEvent &event, void *user) {
        static_cast<Evidence *>(user)->events.push_back(event);
    }
    void Pass(const std::string &name) { checks.push_back(name); }
};

ggml_tensor DenseTensor(std::int64_t columns, std::int64_t rows) {
    ggml_tensor tensor{};
    tensor.type = GGML_TYPE_F32;
    tensor.ne[0] = columns;
    tensor.ne[1] = rows;
    tensor.ne[2] = tensor.ne[3] = 1;
    tensor.nb[0] = sizeof(float);
    tensor.nb[1] = columns * sizeof(float);
    tensor.nb[2] = tensor.nb[3] = rows * tensor.nb[1];
    return tensor;
}

// Malformed metadata belongs in capability queries only. It must never be
// passed to GGML kernels, which are allowed to assert on invalid graph shapes.
struct Query {
    ggml_tensor weight = DenseTensor(16, 8);
    ggml_tensor input = DenseTensor(16, 1);
    ggml_tensor output = DenseTensor(8, 1);

    Query() {
        output.op = GGML_OP_MUL_MAT;
        output.src[0] = &weight;
        output.src[1] = &input;
    }
    Query(const Query &) = delete;
    Query &operator=(const Query &) = delete;
};

void CheckQuery(ggml_backend_dev_t device, Query &query, bool structural,
                bool supported, const std::string &name, Evidence &evidence) {
    Require(adapter::Compatible(&query.output) == structural, name + ": structural result");
    // Repeat queries to catch lazy initialization or state-dependent answers.
    for (int repeat = 0; repeat < 3; ++repeat) {
        Require(ggml_backend_dev_supports_op(device, &query.output) == supported,
                name + ": device support result");
    }
    evidence.Pass(name);
}

void Capabilities(ggml_backend_dev_t device, const std::filesystem::path &artifacts,
                  Evidence &evidence) {
    // The Python harness supplies a private directory containing invalid VMFB
    // bytes. Registration has read its index; retain both files under new names.
    // Subsequent queries must work with neither original path present.
    std::filesystem::rename(artifacts / "profiles.tsv", artifacts / "profiles.retained.tsv");
    std::filesystem::rename(artifacts / "m1.vmfb", artifacts / "m1.retained.vmfb");
    {
        Query q;
        CheckQuery(device, q, true, true, "default_precision", evidence);
        Require(ggml_prec_set_acc(&q.output, GGML_PREC_F32), "set F32 accumulator");
        Require(ggml_prec_set_src(&q.output, GGML_PREC_F32, 1), "set F32 input");
        CheckQuery(device, q, true, true, "explicit_f32", evidence);
    }
    for (const auto type : {GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q4_0}) {
        for (int slot = 0; slot < 3; ++slot) {
            Query q;
            ggml_tensor *tensors[] = {&q.weight, &q.input, &q.output};
            tensors[slot]->type = type;
            CheckQuery(device, q, false, false,
                       "dtype_" + std::string(ggml_type_name(type)) + "_slot_" + std::to_string(slot),
                       evidence);
        }
    }
    for (int slot = 0; slot < 3; ++slot) {
        Query q;
        ggml_tensor *tensors[] = {&q.weight, &q.input, &q.output};
        // Use M=16 so a row stride is observable for every tensor.
        q.input = DenseTensor(16, 16);
        q.output.ne[1] = 16;
        q.output.nb[2] = q.output.nb[3] = 16 * q.output.nb[1];
        tensors[slot]->nb[1] += sizeof(float);
        CheckQuery(device, q, false, false, "row_stride_slot_" + std::to_string(slot), evidence);
    }
    for (int slot = 0; slot < 3; ++slot) {
        for (int axis : {2, 3}) {
            Query q;
            ggml_tensor *tensors[] = {&q.weight, &q.input, &q.output};
            tensors[slot]->ne[axis] = 2;
            CheckQuery(device, q, false, false,
                       "batch_axis_" + std::to_string(axis) + "_slot_" + std::to_string(slot),
                       evidence);
        }
        Query q;
        ggml_tensor *tensors[] = {&q.weight, &q.input, &q.output};
        tensors[slot]->extra = &q;
        CheckQuery(device, q, false, false, "extra_slot_" + std::to_string(slot), evidence);
    }
    for (const auto precision : {GGML_PREC_BF16, GGML_PREC_F16, GGML_PREC_Q8, GGML_PREC_Q4}) {
        Query q;
        q.output.op_params[0] = precision;
        CheckQuery(device, q, false, false, "acc_precision_" + std::to_string(precision), evidence);
        q.output.op_params[0] = 0;
        q.output.op_params[3] = precision;
        CheckQuery(device, q, false, false, "src_precision_" + std::to_string(precision), evidence);
    }
    for (std::size_t word = 0; word < GGML_MAX_OP_PARAMS / sizeof(std::int32_t); ++word) {
        Query q;
        q.output.op_params[word] = 12345;
        CheckQuery(device, q, false, false, "unknown_parameter_" + std::to_string(word), evidence);
    }
    {
        Query q;
        ggml_mul_mat_set_hint(&q.output, GGML_HINT_SRC0_IS_HADAMARD);
        CheckQuery(device, q, false, false, "hadamard_hint", evidence);
    }
    {
        Query q;
        q.input = DenseTensor(16, 2);
        q.output = DenseTensor(8, 2);
        q.output.op = GGML_OP_MUL_MAT;
        q.output.src[0] = &q.weight;
        q.output.src[1] = &q.input;
        CheckQuery(device, q, true, false, "missing_shape_profile", evidence);
    }
    {
        Query q;
        q.weight.ne[0] = 0;
        CheckQuery(device, q, false, false, "zero_extent", evidence);
    }
    {
        Query q;
        q.weight.ne[0] = -1;
        CheckQuery(device, q, false, false, "negative_extent", evidence);
    }
    {
        Query q;
        q.weight = DenseTensor(16, 1LL << 30);
        CheckQuery(device, q, false, false, "oversized_tensor", evidence);
    }
    {
        Query q;
        q.input = DenseTensor(8, 1);
        CheckQuery(device, q, false, false, "contracting_dimension_mismatch", evidence);
    }
    {
        Query q;
        q.output.ne[0] = 7;
        CheckQuery(device, q, false, false, "output_shape_mismatch", evidence);
    }
    {
        Query q;
        q.weight.nb[0] = 8;
        CheckQuery(device, q, false, false, "inner_stride", evidence);
    }
    {
        Query q;
        q.output.op = GGML_OP_ADD;
        CheckQuery(device, q, false, false, "other_operation", evidence);
    }
    for (int slot = 0; slot < 2; ++slot) {
        Query q;
        q.output.src[slot] = nullptr;
        CheckQuery(device, q, false, false, "null_source_" + std::to_string(slot), evidence);
    }
    Require(!adapter::Compatible(nullptr) && !ggml_backend_dev_supports_op(device, nullptr),
            "null operation");
    evidence.Pass("null_operation");
    ggml_backend_buffer_type non_host{};
    Require(ggml_backend_dev_supports_buft(device, ggml_backend_cpu_buffer_type()), "host buffer support");
    Require(!ggml_backend_dev_supports_buft(device, &non_host), "reject non-host buffer");
    Require(!ggml_backend_dev_supports_buft(device, nullptr), "reject null buffer");
    evidence.Pass("buffer_types_host_nonhost_null");
    const auto stats = adapter::GetStats();
    Require(stats.graph_calls == 0 && stats.runtime.module_loads == 0 &&
            stats.runtime.calls_attempted == 0 && stats.failures == 0, "queries executed work");
    evidence.Pass("queries_without_artifact_files_or_execution");
}

void Store(ggml_tensor *tensor, int k, int row, int batch, float value) {
    auto *address = static_cast<char *>(tensor->data) +
                    k * tensor->nb[0] + row * tensor->nb[1] + batch * tensor->nb[2];
    if (tensor->type == GGML_TYPE_F32) {
        std::memcpy(address, &value, sizeof(value));
    } else if (tensor->type == GGML_TYPE_F16) {
        const auto encoded = ggml_fp32_to_fp16(value);
        std::memcpy(address, &encoded, sizeof(encoded));
    } else {
        Require(tensor->type == GGML_TYPE_BF16, "unexpected test tensor dtype");
        const auto encoded = ggml_fp32_to_bf16(value);
        std::memcpy(address, &encoded, sizeof(encoded));
    }
}

float Read(const ggml_tensor *tensor, int k, int row, int batch) {
    const auto *address = static_cast<const char *>(tensor->data) +
                          k * tensor->nb[0] + row * tensor->nb[1] + batch * tensor->nb[2];
    if (tensor->type == GGML_TYPE_F32) {
        float value;
        std::memcpy(&value, address, sizeof(value));
        return value;
    }
    if (tensor->type == GGML_TYPE_F16) {
        ggml_fp16_t value;
        std::memcpy(&value, address, sizeof(value));
        return ggml_fp16_to_fp32(value);
    }
    ggml_bf16_t value;
    std::memcpy(&value, address, sizeof(value));
    return ggml_bf16_to_fp32(value);
}

void Fill(ggml_tensor *tensor, int seed) {
    for (int batch = 0; batch < tensor->ne[2]; ++batch) {
        for (int row = 0; row < tensor->ne[1]; ++row) {
            for (int k = 0; k < tensor->ne[0]; ++k) {
                // Binary fractions are exact in F32, F16 and BF16. This keeps
                // layout/routing checks independent of conversion tolerance.
                const float value = ((k * 3 + row * 5 + batch * 7 + seed * 11) % 19 - 9) / 8.0f;
                Store(tensor, k, row, batch, value);
            }
        }
    }
}

std::vector<float> Reference(const ggml_tensor *weight, const ggml_tensor *input) {
    std::vector<float> result;
    const int broadcast = input->ne[2] / weight->ne[2];
    for (int batch = 0; batch < input->ne[2]; ++batch) {
        for (int m = 0; m < input->ne[1]; ++m) {
            for (int n = 0; n < weight->ne[1]; ++n) {
                double value = 0;
                for (int k = 0; k < input->ne[0]; ++k) {
                    value += double(Read(input, k, m, batch)) * Read(weight, k, n, batch / broadcast);
                }
                result.push_back(static_cast<float>(value));
            }
        }
    }
    return result;
}

void CheckOutput(ggml_tensor *output, const std::vector<float> &reference, Evidence &evidence) {
    std::vector<float> actual(reference.size());
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double error = std::abs(double(actual[i]) - reference[i]);
        evidence.max_error = std::max(evidence.max_error, error);
        Require(std::isfinite(actual[i]) && error <= 1e-6 + 1e-5 * std::abs(reference[i]),
                "scalar reference mismatch at element " + std::to_string(i));
    }
}

// Input owners outlive scheduler/backend objects in RunGraphCase. Even cached
// immutable weights keep valid addresses until the Coral backend is destroyed.
struct Fixture {
    Context weights_context = NewContext();
    Context inputs_context = NewContext();
    Context graph_context = NewContext();
    Buffer weights_buffer{nullptr, ggml_backend_buffer_free};
    Buffer inputs_buffer{nullptr, ggml_backend_buffer_free};
    ggml_tensor *weight = nullptr;
    ggml_tensor *second_weight = nullptr;
    ggml_tensor *input = nullptr;
    std::vector<ggml_tensor *> outputs;
    ggml_cgraph *graph = nullptr;

    explicit Fixture(const std::string &name) {
        const int m = (name == "dense_prefill" || name == "weight_stride" ||
                       name == "activation_stride") ? 16 : (name == "missing_profile" ? 2 : 1);
        const int input_batches = (name == "broadcast" || name == "batch3d") ? 2 : 1;
        const int weight_batches = name == "batch3d" ? 2 : 1;
        const auto type = name == "f16_weight" ? GGML_TYPE_F16 :
                          name == "bf16_weight" ? GGML_TYPE_BF16 : GGML_TYPE_F32;
        if (name == "weight_stride") {
            auto *storage = ggml_new_tensor_2d(weights_context.get(), type, 17, 8);
            weight = ggml_view_2d(weights_context.get(), storage, 16, 8, 17 * sizeof(float), 0);
        } else {
            weight = ggml_new_tensor_3d(weights_context.get(), type, 16, 8, weight_batches);
            if (name == "cache_view_mutable" || name == "cache_view_readonly") {
                weight = ggml_view_2d(weights_context.get(), weight, 16, 8,
                                      16 * sizeof(float), 0);
            }
        }
        if (name == "cache_immutable" || name == "mixed_graph" || name == "preflight_failure") {
            second_weight = ggml_new_tensor_2d(weights_context.get(), GGML_TYPE_F32, 16, 8);
        }
        if (name == "activation_stride") {
            auto *storage = ggml_new_tensor_2d(inputs_context.get(), GGML_TYPE_F32, 17, m);
            input = ggml_view_2d(inputs_context.get(), storage, 16, m, 17 * sizeof(float), 0);
        } else {
            input = ggml_new_tensor_3d(inputs_context.get(), GGML_TYPE_F32, 16, m, input_batches);
        }
        ggml_set_name(weight, "weight");
        ggml_set_name(input, "activation");
        outputs.push_back(ggml_mul_mat(graph_context.get(), weight, input));
        ggml_set_name(outputs[0], "matmul");
        if (name == "explicit_f32") {
            Require(ggml_prec_set_acc(outputs[0], GGML_PREC_F32), "F32 acc setup");
            Require(ggml_prec_set_src(outputs[0], GGML_PREC_F32, 1), "F32 src setup");
        } else if (name == "precision_bf16_acc" || name == "precision_f16_acc") {
            const auto precision = name == "precision_bf16_acc" ? GGML_PREC_BF16 : GGML_PREC_F16;
            Require(ggml_prec_set_acc(outputs[0], precision), "non-F32 acc setup");
        } else if (name == "precision_bf16_src" || name == "precision_q8_src") {
            const auto precision = name == "precision_bf16_src" ? GGML_PREC_BF16 : GGML_PREC_Q8;
            Require(ggml_prec_set_src(outputs[0], precision, 1), "non-F32 src setup");
        }
        if (second_weight) {
            outputs.push_back(ggml_mul_mat(graph_context.get(), second_weight, input));
            ggml_set_name(outputs[1], "second_matmul");
            if (name == "mixed_graph" || name == "preflight_failure") {
                Require(ggml_prec_set_acc(outputs[1], GGML_PREC_BF16), "mixed precision setup");
            }
        }
        graph = ggml_new_graph_custom(graph_context.get(), 128, false);
        for (auto *output : outputs) {
            ggml_set_output(output);
            ggml_build_forward_expand(graph, output);
        }
        weights_buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(
            weights_context.get(), ggml_backend_cpu_buffer_type()));
        inputs_buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(
            inputs_context.get(), ggml_backend_cpu_buffer_type()));
        Require(weights_buffer && inputs_buffer, "allocate host inputs");
        if (name != "cache_optin_mutable") {
            ggml_backend_buffer_set_usage(weights_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }
        ggml_backend_buffer_clear(weights_buffer.get(), 0);
        ggml_backend_buffer_clear(inputs_buffer.get(), 0);
        Fill(weight, 1);
        Fill(input, 2);
        if (second_weight) {
            Fill(second_weight, 7);
        }
    }
};

bool UsesImmutablePromise(const std::string &name) {
    return name == "cache_immutable" || name == "cache_optin_mutable" ||
           name == "cache_view_readonly";
}

bool ExpectsCoral(const std::string &name) {
    return name == "dense_decode" || name == "dense_prefill" || name == "explicit_f32" ||
           name.rfind("cache_", 0) == 0 || name == "mixed_graph";
}

void CheckExecution(const adapter::Stats &before, std::size_t first_event,
                    std::uint64_t count, const Evidence &evidence) {
    const auto after = adapter::GetStats();
    Require(after.matmuls - before.matmuls == count, "wrong executed Coral matmul count");
    Require(evidence.events.size() - first_event == count, "missing Coral node events");
    std::uint64_t dispatches = 0, instructions = 0;
    for (std::size_t i = first_event; i < evidence.events.size(); ++i) {
        Require(evidence.events[i].dispatches > 0 && evidence.events[i].instructions > 0,
                "missing actual NPU execution");
        dispatches += evidence.events[i].dispatches;
        instructions += evidence.events[i].instructions;
    }
    Require(after.runtime.coral_dispatches - before.runtime.coral_dispatches == dispatches &&
            after.runtime.kernel_instructions - before.runtime.kernel_instructions == instructions,
            "trace/runtime counter mismatch");
    Require(after.runtime.calls_completed - before.runtime.calls_completed == count,
            "incomplete runtime calls");
    Require(after.failures == before.failures, "unexpected backend failure");
}

void RunGraphCase(const std::string &name, ggml_backend_dev_t device, Evidence &evidence) {
    Fixture fixture(name);
    Backend cpu(ggml_backend_cpu_init(), ggml_backend_free);
    Backend coral(ggml_backend_dev_init(device, nullptr), ggml_backend_free);
    Require(cpu && coral, "initialize backends: " + adapter::LastError());
    ggml_backend_cpu_set_n_threads(cpu.get(), 1);
    ggml_backend_t backends[] = {coral.get(), cpu.get()};
    Scheduler scheduler(ggml_backend_sched_new(backends, nullptr, 2, 128, false, true),
                        ggml_backend_sched_free);
    Require(scheduler != nullptr, "create scheduler");
    const bool expected_coral = ExpectsCoral(name);
    Require(ggml_backend_dev_supports_op(device, fixture.outputs[0]) == expected_coral,
            "unexpected capability for graph");
    Require(ggml_backend_sched_alloc_graph(scheduler.get(), fixture.graph), "allocate scheduled graph");
    for (std::size_t i = 0; i < fixture.outputs.size(); ++i) {
        const bool on_coral = expected_coral && !(name == "mixed_graph" && i == 1);
        const auto assigned = ggml_backend_sched_get_tensor_backend(scheduler.get(), fixture.outputs[i]);
        Require(assigned == (on_coral ? coral.get() : cpu.get()), "incorrect automatic placement");
        evidence.placements.push_back(ggml_backend_name(assigned));
    }
    const bool mutable_weight = name == "cache_default_mutable" ||
                                name == "cache_optin_mutable" || name == "cache_view_mutable";
    const bool readonly_view = name == "cache_view_readonly";
    const int repeats = (mutable_weight || readonly_view || name == "cache_immutable") ? 3 : 1;
    std::vector<float> original;
    for (int run = 0; run < repeats; ++run) {
        if (run > 0) {
            if (mutable_weight) {
                // Keep both the tensor and backing address unchanged: a stale
                // pointer-keyed cache would return the previous result.
                Fill(fixture.weight, run + 3);
            } else {
                Fill(fixture.input, run + 8);
            }
        }
        std::vector<std::vector<float>> expected;
        expected.push_back(Reference(fixture.weight, fixture.input));
        if (fixture.second_weight) {
            expected.push_back(Reference(fixture.second_weight, fixture.input));
            if (name == "cache_immutable") {
                Require(expected[0] != expected[1], "same-shape weights need distinct answers");
            }
        }
        if (run == 0) {
            original = expected[0];
        } else {
            Require(expected[0] != original, "mutation did not change expected output");
        }
        const auto before = adapter::GetStats();
        const auto first_event = evidence.events.size();
        const auto status = ggml_backend_sched_graph_compute(scheduler.get(), fixture.graph);
        Require(status == GGML_STATUS_SUCCESS, "scheduled compute: " + adapter::LastError());
        ggml_backend_sched_synchronize(scheduler.get());
        for (std::size_t i = 0; i < fixture.outputs.size(); ++i) {
            CheckOutput(fixture.outputs[i], expected[i], evidence);
        }
        const std::uint64_t coral_count = name == "cache_immutable" ? 2 : (expected_coral ? 1 : 0);
        CheckExecution(before, first_event, coral_count, evidence);
        const auto after = adapter::GetStats();
        if (mutable_weight || readonly_view) {
            Require(after.weight_cache_hits == before.weight_cache_hits &&
                    after.weight_preparations - before.weight_preparations == 1,
                    "ineligible weight reused a cached snapshot");
        } else if (name == "cache_immutable") {
            Require(after.weight_cache_hits - before.weight_cache_hits == (run > 0 ? 2 : 0) &&
                    after.weight_preparations - before.weight_preparations == (run > 0 ? 0 : 2),
                    "immutable weight cache identity/reuse failure");
        }
        if (run > 0) {
            Require(after.runtime.module_loads == before.runtime.module_loads, "module reloaded");
        }
        evidence.Pass(name + "_run_" + std::to_string(run));
    }
    if (!expected_coral) {
        Require(adapter::GetStats().runtime.module_loads == 0, "CPU-only case loaded a VMFB");
    }
}

void FailureCase(const std::string &name, ggml_backend_dev_t device, Evidence &evidence) {
    Fixture fixture(name);
    Buffer output_buffer(ggml_backend_alloc_ctx_tensors_from_buft(
                             fixture.graph_context.get(), ggml_backend_cpu_buffer_type()),
                         ggml_backend_buffer_free);
    Require(output_buffer != nullptr, "allocate sentinel outputs");
    const std::vector<float> sentinel(8, -9876.5f);
    for (auto *output : fixture.outputs) {
        ggml_backend_tensor_set(output, sentinel.data(), 0, sentinel.size() * sizeof(float));
    }
    Backend coral(ggml_backend_dev_init(device, nullptr), ggml_backend_free);
    Require(coral != nullptr, "initialize failure-test backend");
    const auto status = ggml_backend_graph_compute(coral.get(), fixture.graph);
    Require(status == GGML_STATUS_FAILED, "invalid graph returned success");
    const auto error = adapter::LastError();
    if (name == "preflight_failure") {
        Require(error.find("unsupported operation") != std::string::npos, "wrong preflight error");
        Require(adapter::GetStats().runtime.module_loads == 0, "preflight began numerical work");
    } else {
        Require(error.find("ABI") != std::string::npos || error.find("abi") != std::string::npos,
                "expected VMFB ABI mismatch: " + error);
    }
    for (auto *output : fixture.outputs) {
        CheckOutput(output, sentinel, evidence);
    }
    Require(evidence.events.empty() && adapter::GetStats().runtime.coral_dispatches == 0,
            "rejected graph dispatched");
    evidence.Pass("failure_preserves_all_outputs");
    // Even a now-valid graph must not revive a backend after its failure.
    if (name == "preflight_failure") {
        Require(ggml_prec_set_acc(fixture.outputs[1], GGML_PREC_F32), "repair precision for retry");
    }
    const auto retry = ggml_backend_graph_compute(coral.get(), fixture.graph);
    Require(retry == GGML_STATUS_FAILED &&
            adapter::LastError().find("unusable after a graph failure") != std::string::npos,
            "failed backend was reused");
    for (auto *output : fixture.outputs) {
        CheckOutput(output, sentinel, evidence);
    }
    evidence.expected_failures = 2;
    Require(adapter::GetStats().failures == 2 && adapter::GetStats().matmuls == 0,
            "failure accounting");
    evidence.Pass("failed_backend_stays_failed");
}

void PrintEvidence(const std::string &name, const Evidence &evidence) {
    const auto stats = adapter::GetStats();
    Require(stats.failures == evidence.expected_failures, "unexpected failure count");
    Require(stats.runtime.host_executable_load_attempts == 0 &&
            stats.runtime.failed_dispatches == 0 && stats.runtime.observer_errors == 0,
            "unexpected host execution or dispatch failure");
    std::cout << "{\"case\":" << coral::JsonString(name) << ",\"status\":\"passed\",\"checks\":[";
    for (std::size_t i = 0; i < evidence.checks.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << coral::JsonString(evidence.checks[i]);
    }
    std::cout << "],\"placements\":[";
    for (std::size_t i = 0; i < evidence.placements.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << coral::JsonString(evidence.placements[i]);
    }
    std::cout << "],\"events\":[";
    for (std::size_t i = 0; i < evidence.events.size(); ++i) {
        if (i) std::cout << ',';
        const auto &event = evidence.events[i];
        std::cout << "{\"dispatches\":" << event.dispatches << ",\"instructions\":" << event.instructions
                  << ",\"weight_cache_hit\":" << (event.weight_cache_hit ? "true" : "false") << '}';
    }
    std::cout << "],\"max_abs_error\":" << evidence.max_error
              << ",\"matmuls\":" << stats.matmuls
              << ",\"module_loads\":" << stats.runtime.module_loads
              << ",\"dispatches\":" << stats.runtime.coral_dispatches
              << ",\"instruction_samples\":" << stats.runtime.dispatches_with_instruction_count
              << ",\"kernel_instructions\":" << stats.runtime.kernel_instructions
              << ",\"weight_preparations\":" << stats.weight_preparations
              << ",\"weight_cache_hits\":" << stats.weight_cache_hits
              << ",\"expected_failures\":" << stats.failures << "}\n";
}
} // namespace

int main(int argc, char **argv) {
    try {
        Require(argc == 3, "usage: coral-adapter-test CASE PRIVATE_ARTIFACT_DIRECTORY");
        const std::string name = argv[1];
        Evidence evidence;
        ggml_backend_register(ggml_backend_cpu_reg());
        adapter::Options options;
        options.artifact_directory = argv[2];
        options.immutable_model_weights = UsesImmutablePromise(name);
        options.trace = Evidence::OnCoral;
        options.trace_user_data = &evidence;
        adapter::Register(options);
        auto *device = ggml_backend_dev_by_name("CORAL");
        Require(device != nullptr, "find registered Coral device");
        Require(ggml_backend_dev_count() == 2, "only CPU and Coral may be registered");
        if (name == "capabilities") {
            Capabilities(device, argv[2], evidence);
        } else if (name == "preflight_failure" || name == "abi_failure") {
            FailureCase(name, device, evidence);
        } else {
            RunGraphCase(name, device, evidence);
        }
        PrintEvidence(name, evidence);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "coral-adapter-test: " << error.what() << '\n';
        return 1;
    }
}
