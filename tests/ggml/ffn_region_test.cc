// SPDX-License-Identifier: MIT
#include "ggml/ggml_coral.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "planner/ggml_fusion.h"
#include "support/json.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace adapter = coral::ggml_backend;
using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using Backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
using Scheduler = std::unique_ptr<ggml_backend_sched, decltype(&ggml_backend_sched_free)>;
void Require(bool value, const std::string &message) {
    if (!value) throw std::runtime_error(message + ": " + adapter::LastError());
}
std::vector<float> Read(ggml_tensor *tensor) {
    std::vector<float> result(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, result.data(), 0, ggml_nbytes(tensor));
    return result;
}
double maximum_error = 0;
void Compare(const std::vector<float> &actual, const std::vector<float> &reference) {
    Require(actual.size() == reference.size(), "comparison size");
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double error = std::abs(double(actual[i]) - reference[i]);
        maximum_error = std::max(maximum_error, error);
        Require(std::isfinite(actual[i]) && error <= 1e-4 + 1e-4 * std::abs(reference[i]),
                "numerical mismatch at " + std::to_string(i) + " actual=" + std::to_string(actual[i]) + " reference=" + std::to_string(reference[i]));
    }
}
void Fill(ggml_tensor *tensor, int seed) {
    std::vector<float> values(ggml_nelements(tensor));
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = float(int((i * 7 + seed * 11) % 31) - 15) / 128;
    }
    ggml_backend_tensor_set(tensor, values.data(), 0, ggml_nbytes(tensor));
}
struct Fixture {
    Context context{ggml_init({4 << 20, nullptr, true}), ggml_free};
    Context weight_context{ggml_init({1 << 20, nullptr, true}), ggml_free};
    Buffer weights{nullptr, ggml_backend_buffer_free};
    ggml_cgraph *graph = nullptr;
    ggml_tensor *seed = nullptr, *x = nullptr, *gate = nullptr, *y = nullptr, *extra = nullptr;
    std::vector<ggml_tensor *> weight_tensors;
    int rows;
    Fixture(ggml_backend_t cpu, int m, int spelling, const std::string &mode) : rows(m) {
        Require(context && weight_context, "context allocation");
        seed = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 64, rows);
        ggml_set_input(seed);
        // A CPU producer whose result is not an externally preserved output.
        // The scheduler must keep this activation alive for the region.
        x = ggml_scale(context.get(), seed, 0.5f);
        y = AddFFN(x, spelling);
        if (mode == "chain" || mode == "multiple-splits") {
            auto *next = mode == "chain" ? y : ggml_scale(context.get(), y, 0.75f);
            y = AddFFN(next, spelling);
        }
        ggml_set_output(y);
        graph = ggml_new_graph_custom(context.get(), 128, false);
        ggml_build_forward_expand(graph, y);
        if (mode == "external-consumer") {
            extra = ggml_add(context.get(), gate, gate);
        } else if (mode == "external-view") {
            extra = ggml_view_2d(context.get(), gate, 172, rows, gate->nb[1], 0);
        }
        if (extra) {
            ggml_set_output(extra);
            ggml_build_forward_expand(graph, extra);
        }
        weights.reset(ggml_backend_alloc_ctx_tensors(weight_context.get(), cpu));
        Require(weights != nullptr, "weight allocation");
        ggml_backend_buffer_set_usage(weights.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    ggml_tensor *AddFFN(ggml_tensor *input, int spelling) {
        auto *wg = ggml_new_tensor_2d(weight_context.get(), GGML_TYPE_F32, 64, 172);
        auto *wu = ggml_new_tensor_2d(weight_context.get(), GGML_TYPE_F32, 64, 172);
        auto *wd = ggml_new_tensor_2d(weight_context.get(), GGML_TYPE_F32, 172, 64);
        weight_tensors.insert(weight_tensors.end(), {wg, wu, wd});
        gate = ggml_mul_mat(context.get(), wg, input);
        auto *up = ggml_mul_mat(context.get(), wu, input);
        ggml_tensor *hidden = nullptr;
        if (spelling == 0) {
            hidden = ggml_swiglu_split(context.get(), gate, up);
        } else {
            auto *silu = ggml_silu(context.get(), gate);
            hidden = spelling == 1 ? ggml_mul(context.get(), silu, up) : ggml_mul(context.get(), up, silu);
        }
        return ggml_mul_mat(context.get(), wd, hidden);
    }
    void Set(int activation, int weight) {
        Fill(seed, activation);
        for (std::size_t i = 0; i < weight_tensors.size(); ++i) {
            Fill(weight_tensors[i], weight + int(i) * 3);
        }
    }
};
struct Observation {
    int callbacks = 0;
    std::uint64_t events = 0, covered = 0, dispatches = 0, instructions = 0;
    static bool Eval(ggml_tensor *, bool ask, void *user) {
        if (!ask) ++static_cast<Observation *>(user)->callbacks;
        return true;
    }
    static void FFN(const adapter::FFNEvent &event, void *user) {
        auto &self = *static_cast<Observation *>(user);
        Require(event.dispatches && event.instructions && event.program_sha256.size() == 64,
                "incomplete region execution evidence");
        ++self.events;
        self.covered += event.plan.members.size();
        self.dispatches += event.dispatches;
        self.instructions += event.instructions;
    }
};
void Scheduled(ggml_backend_t cpu, int rows, int spelling,
               const std::string &mode, bool immutable) {
    const auto planning_before = adapter::GetStats();
    Fixture f(cpu, rows, spelling, mode);
    Fixture reference(cpu, rows, spelling, mode);
    Buffer reference_storage(ggml_backend_alloc_ctx_tensors(reference.context.get(), cpu),
                             ggml_backend_buffer_free);
    Backend coral_owner(ggml_backend_dev_init(ggml_backend_dev_by_name("CORAL"), nullptr),
                        ggml_backend_free);
    const auto coral = coral_owner.get();
    Require(coral != nullptr, "backend initialization");
    ggml_backend_t backends[] = {coral, cpu};
    Scheduler scheduler(ggml_backend_sched_new(backends, nullptr, 2, 128, false, true),
                        ggml_backend_sched_free);
    Require(scheduler && reference_storage, "scheduler allocation");
    Observation callback;
    if (mode == "callbacks" || mode == "callback-subdivision") {
        ggml_backend_sched_set_eval_callback(scheduler.get(), Observation::Eval, &callback);
    }
    // Exercise reservation with a different arena, then destroy its tensors.
    // Reservation plans must never be dereferenced while executing this graph.
    {
        Fixture reservation(cpu, rows, spelling, "normal");
        Require(ggml_backend_sched_reserve(scheduler.get(), reservation.graph), "reserve");
        ggml_backend_sched_reset(scheduler.get());
    }
    Require(ggml_backend_sched_alloc_graph(scheduler.get(), f.graph), "allocate graph");
    const auto before = adapter::GetStats();
    Require(before.jit.requests == planning_before.jit.requests &&
            before.jit.worker_launches == planning_before.jit.worker_launches,
            "reservation/allocation unexpectedly invoked JIT");
    const bool rejected = mode == "external-consumer" || mode == "external-view" ||
        mode == "callbacks" || mode == "callback-subdivision" || mode == "stale-metadata" ||
        mode == "missing" || mode == "disabled" || mode == "missing-member";
    const std::uint64_t per_run = (mode == "chain" || mode == "multiple-splits") ? 2 : 1;
    if (mode == "stale-metadata") {
        // Still executable as a member, but different from the admitted plan.
        f.gate->op_params[0] = GGML_PREC_F32;
        reference.gate->op_params[0] = GGML_PREC_F32;
    }
    for (int run = 0; run < 3; ++run) {
        const bool changed = mode == "weight-generation";
        const int weight_seed = immutable && !changed ? 2 : 2 + run * 3;
        if (changed && run > 0) adapter::InvalidateModelWeights();
        f.Set(run + 1, weight_seed);
        reference.Set(run + 1, weight_seed);
        Require(ggml_backend_graph_compute(cpu, reference.graph) == GGML_STATUS_SUCCESS, "CPU oracle");
        Require(ggml_backend_sched_graph_compute(scheduler.get(), f.graph) == GGML_STATUS_SUCCESS,
                "scheduled Coral");
        Compare(Read(f.y), Read(reference.y));
        if (f.extra) Compare(Read(f.extra), Read(reference.extra));
        // The allocation dependency preserves the produced X through the region.
        if (!rejected && per_run == 1) Compare(Read(f.x), Read(reference.x));
    }
    const auto after = adapter::GetStats();
    const auto regions = after.ffn_calls - before.ffn_calls;
    Require(after.graph_optimizations == before.graph_optimizations, "reuse rebuilt graph");
    Require(regions == (rejected ? 0 : 3 * per_run), "unexpected complete region count");
    if (!rejected) {
        Require(after.matmuls == before.matmuls && after.pointwise_calls == before.pointwise_calls,
                "region executed a member twice");
        Require(after.ffn_members_skipped - before.ffn_members_skipped ==
                    regions * (spelling ? 5 : 4), "wrong skipped member count");
        Require(after.ffn_weight_preparations - before.ffn_weight_preparations ==
                    (immutable && mode != "weight-generation" ? per_run : 3 * per_run), "wrong weight preparation count");
        Require(after.ffn_weight_cache_hits - before.ffn_weight_cache_hits ==
                    (immutable && mode != "weight-generation" ? 2 * per_run : 0), "wrong weight cache hit count");
        if (mode == "weight-generation") {
            Require(after.weight_cache_resets - before.weight_cache_resets == 2,
                    "weight changes did not rebuild the device cache");
        }
        if (mode == "multiple-splits") {
            Require(ggml_backend_sched_get_n_splits(scheduler.get()) >= 4, "missing multiple splits");
        }
    } else {
        Require(after.matmuls - before.matmuls == 9, "fallback lost matmuls");
        if (mode == "stale-metadata") {
            Require(after.ffn_guard_rejections - before.ffn_guard_rejections == 3,
                    "metadata guard did not run on reuse");
        }
    }
    if (mode == "callbacks" || mode == "callback-subdivision") {
        Require(callback.callbacks >= 15, "evaluation callbacks lost intermediate observations");
    }
    std::cout << "{\"case\":" << coral::JsonString(mode) << ",\"m\":" << rows
              << ",\"spelling\":" << spelling << ",\"ffn_calls\":" << regions
              << ",\"callbacks\":" << callback.callbacks << "}\n";
}
void Direct(ggml_backend_t coral, ggml_backend_t cpu) {
    Fixture f(cpu, 1, 0, "normal");
    Buffer storage(ggml_backend_alloc_ctx_tensors(f.context.get(), cpu), ggml_backend_buffer_free);
    f.Set(1, 2);
    Require(ggml_backend_graph_compute(cpu, f.graph) == GGML_STATUS_SUCCESS, "direct CPU");
    const auto expected = Read(f.y);
    // Omit the unsupported scale node; no preallocation plan exists for this view.
    auto view = ggml_graph_view(f.graph, 1, f.graph->n_nodes);
    const auto before = adapter::GetStats();
    Require(ggml_backend_graph_compute(coral, &view) == GGML_STATUS_SUCCESS, "direct members");
    Compare(Read(f.y), expected);
    Require(adapter::GetStats().ffn_calls == before.ffn_calls, "unplanned view fused");
}
void Corrupt(ggml_backend_t coral, ggml_backend_t cpu) {
    Fixture f(cpu, 1, 0, "normal");
    ggml_backend_t backends[] = {coral, cpu};
    Scheduler scheduler(ggml_backend_sched_new(backends, nullptr, 2, 128, false, true),
                        ggml_backend_sched_free);
    Require(ggml_backend_sched_alloc_graph(scheduler.get(), f.graph), "corrupt allocation");
    f.Set(1, 2);
    Fill(f.y, 19);
    const auto marker = Read(f.y);
    const auto before = adapter::GetStats();
    Require(ggml_backend_sched_graph_compute(scheduler.get(), f.graph) == GGML_STATUS_FAILED,
            "corrupt region was accepted");
    Require(Read(f.y) == marker, "corrupt region published output");
    Require(adapter::GetStats().runtime.calls_attempted == before.runtime.calls_attempted &&
            adapter::GetStats().matmuls == before.matmuls, "corrupt region replayed members");
    const auto launches = adapter::GetStats().jit.worker_launches;
    Require(ggml_backend_sched_graph_compute(scheduler.get(), f.graph) == GGML_STATUS_FAILED,
            "failed backend allowed retry");
    Require(adapter::GetStats().jit.worker_launches == launches, "failed backend relaunched a worker");
}
} // namespace
int main(int argc, char **argv) {
    try {
        Require(argc == 3 || argc == 4, "Usage: coral-ffn-region-test MODE CATALOG [JIT_CONFIG]");
        const std::string mode = argv[1];
        if (mode == "jit-spec") {
            Require(argc == 4, "jit-spec requires a configuration");
            coral::jit::FFNJitClient client(argv[3]);
            const std::vector<coral::contracts::FFNShape> invalid{
                {0, 64, 172}, {-1, 64, 172}, {1, 0, 172}, {1, 64, 0},
                {536870912, 64, 172}};
            for (const auto shape : invalid) {
                bool rejected = false;
                try {
                    (void)client.Specification(shape);
                } catch (const std::invalid_argument &) {
                    rejected = true;
                }
                Require(rejected, "bucket policy concealed an invalid logical shape");
            }
            std::cout << "{\"status\":\"passed\",\"invalid_logical_shapes\":5}\n";
            return 0;
        }
        Observation events;
        adapter::Options options;
        options.artifact_directory = argv[2];
        if (argc == 4) options.jit_configuration = argv[3];
        options.evaluation_callbacks = mode == "callbacks";
        options.enable_ffn = mode != "disabled";
        options.immutable_model_weights = mode == "immutable" || mode == "weight-generation";
        options.ffn_trace = Observation::FFN;
        options.trace_user_data = &events;
        adapter::Register(options);
        Backend cpu(ggml_backend_cpu_init(), ggml_backend_free);
        ggml_backend_cpu_set_n_threads(cpu.get(), 1);
        if (mode == "catalog") {
            std::cout << "{\"status\":\"passed\"}\n";
            return 0;
        }
        // Each scheduled fixture destroys its backend/cache before its weights.
        // Independent contexts must prepare their own snapshots.
        if (mode == "corrupt" || mode == "jit-failure") {
            Backend coral(ggml_backend_dev_init(ggml_backend_dev_by_name("CORAL"), nullptr),
                          ggml_backend_free);
            Corrupt(coral.get(), cpu.get());
        } else {
            const auto cases = mode == "mutable" || mode == "immutable"
                ? std::vector<std::string>{"normal", "chain", "multiple-splits"}
                : std::vector<std::string>{mode};
            for (const auto &name : cases) {
                for (int rows : {1, 5}) {
                    for (int spelling = 0; spelling < 3; ++spelling) {
                        Scheduled(cpu.get(), rows, spelling, name, options.immutable_model_weights);
                    }
                }
            }
            if (mode == "mutable") {
                Backend coral(ggml_backend_dev_init(ggml_backend_dev_by_name("CORAL"), nullptr),
                              ggml_backend_free);
                Direct(coral.get(), cpu.get());
            }
        }
        const auto stats = adapter::GetStats();
        Require(stats.runtime.host_executable_load_attempts == 0 &&
                stats.runtime.failed_dispatches == 0 && stats.runtime.observer_errors == 0,
                "runtime observer/host failure");
        Require(stats.ffn_calls == events.events && stats.ffn_members_skipped == events.covered,
                "region event coverage mismatch");
        Require(stats.runtime.dispatches_with_instruction_count == stats.runtime.coral_dispatches,
                "missing instruction samples");
        std::cout << "{\"status\":\"passed\",\"mode\":" << coral::JsonString(mode)
                  << ",\"ffn_calls\":" << stats.ffn_calls
                  << ",\"members_skipped\":" << stats.ffn_members_skipped
                  << ",\"dispatches\":" << stats.runtime.coral_dispatches
                  << ",\"instructions\":" << stats.runtime.kernel_instructions
                  << ",\"jit_requests\":" << stats.jit.requests
                  << ",\"jit_worker_launches\":" << stats.jit.worker_launches
                  << ",\"jit_compilations\":" << stats.jit.compilations
                  << ",\"jit_disk_hits\":" << stats.jit.disk_hits
                  << ",\"max_abs_error\":" << maximum_error << "}\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FFN region test: " << error.what() << '\n';
        return 1;
    }
}
