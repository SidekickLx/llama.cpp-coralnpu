// SPDX-License-Identifier: MIT
#include "planner/ggml_fusion.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace coral::planner;
using Code = RegionRejectionCode;
std::vector<std::string> passed;
void Require(bool ok, const char *message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}
struct Fixture {
    ggml_context *context;
    ggml_cgraph *graph;
    ggml_tensor *x, *wg, *wu, *wd, *gate, *up, *silu = nullptr, *hidden, *y;
    explicit Fixture(int spelling = 0, bool shared_weight = false) {
        context = ggml_init({1 << 20, nullptr, true});
        Require(context != nullptr, "fixture context allocation failed");
        x = ggml_new_tensor_2d(context, GGML_TYPE_F32, 3, 2);
        wg = ggml_new_tensor_2d(context, GGML_TYPE_F32, 3, 4);
        wu = shared_weight ? wg : ggml_new_tensor_2d(context, GGML_TYPE_F32, 3, 4);
        wd = ggml_new_tensor_2d(context, GGML_TYPE_F32, 4, 3);
        ggml_set_input(x);
        gate = ggml_mul_mat(context, wg, x);
        up = ggml_mul_mat(context, wu, x);
        if (spelling) {
            silu = ggml_silu(context, gate);
            hidden = spelling == 1 ? ggml_mul(context, silu, up) : ggml_mul(context, up, silu);
        } else {
            hidden = ggml_swiglu_split(context, gate, up);
        }
        y = ggml_mul_mat(context, wd, hidden);
        ggml_set_output(y);
        graph = ggml_new_graph_custom(context, 64, false);
        ggml_build_forward_expand(graph, y);
    }
    ~Fixture() { ggml_free(context); }
    Fixture(const Fixture &) = delete;
    Fixture &operator=(const Fixture &) = delete;
};
struct Capabilities {
    const ggml_tensor *rejected = nullptr;
    MemberSupport answer = MemberSupport::Unsupported;
    static MemberSupport Query(const ggml_tensor *node, void *user) {
        auto &self = *static_cast<Capabilities *>(user);
        return node == self.rejected ? self.answer : MemberSupport::Supported;
    }
};
struct Dependencies {
    std::vector<std::pair<ggml_tensor *, ggml_tensor *>> edges;
    static void Add(void *user, ggml_tensor *tensor, ggml_tensor *until) {
        static_cast<Dependencies *>(user)->edges.emplace_back(tensor, until);
    }
    ggml_backend_graph_optimize_params Params() { return {Add, this}; }
};
FFNPolicy Policy(Capabilities &caps, std::uint64_t generation = 1) {
    return {generation, false, Capabilities::Query, &caps};
}
void ExpectRejected(const FFNCheckResult &result, Code code, const Dependencies &deps) {
    auto *error = std::get_if<RegionRejection>(&result);
    Require(error && error->code == code, "wrong planner rejection");
    Require(deps.edges.empty(), "rejected region modified allocation dependencies");
}
void Positive(const std::string &name, int spelling, bool shared) {
    Fixture fixture(spelling, shared);
    Capabilities caps;
    Dependencies deps;
    auto params = deps.Params();
    auto result = PlanFFN(*fixture.graph, fixture.graph->n_nodes - 1, Policy(caps), &params);
    auto *plan = std::get_if<GGMLFFNPlan>(&result);
    Require(plan != nullptr, "valid FFN rejected");
    Require(plan->shape.m == 2 && plan->shape.d == 3 && plan->shape.f == 4, "wrong FFN shape");
    Require(plan->inputs[1] == fixture.wg && plan->inputs[2] == fixture.wu, "swapped weight roles");
    Require(deps.edges.size() == (shared ? 3u : 4u), "wrong lifetime dependency count");
    for (const auto &edge : deps.edges) {
        Require(edge.second == fixture.y, "wrong lifetime anchor");
    }
    passed.push_back(name);
}
void Negative(const std::string &name, Code code) {
    Fixture f;
    Capabilities caps;
    Dependencies deps;
    auto params = deps.Params();
    auto policy = Policy(caps);
    ggml_cgraph view = ggml_graph_view(f.graph, 0, f.graph->n_nodes);
    int anchor = view.n_nodes - 1;
    if (name == "external-consumer" || name == "external-view") {
        ggml_tensor *outside = name == "external-consumer"
            ? ggml_add(f.context, f.gate, f.gate)
            : ggml_view_2d(f.context, f.gate, 4, 2, f.gate->nb[1], 0);
        ggml_build_forward_expand(f.graph, outside);
        view = ggml_graph_view(f.graph, 0, anchor + 1);
    } else if (name == "cross-split") {
        view = ggml_graph_view(f.graph, 1, f.graph->n_nodes);
        anchor = view.n_nodes - 1;
    } else if (name == "view-input") {
        f.x->view_src = f.wg;
    } else if (name == "dtype") {
        f.up->type = GGML_TYPE_F16;
    } else if (name == "stride") {
        ++f.x->nb[1];
    } else if (name == "shape") {
        f.wd->ne[0] = 3; f.wd->nb[1] = 12; f.wd->nb[2] = f.wd->nb[3] = 36;
    } else if (name == "unknown-parameter") {
        f.hidden->op_params[15] = 1;
    } else if (name == "extra-source") {
        f.hidden->src[2] = f.x;
    } else if (name == "observed-intermediate") {
        ggml_set_output(f.gate);
    } else if (name == "debug-callback") {
        policy.evaluation_callbacks = true;
    } else if (name == "unknown-capability") {
        policy.query_member = nullptr;
    } else if (name == "unsupported-member") {
        caps.rejected = f.hidden;
    } else if (name == "missing-member-artifact") {
        caps.rejected = f.up; caps.answer = MemberSupport::MissingArtifact;
    } else if (name == "missing-lifetime-hook") {
        params.add_alloc_dep = nullptr;
    } else if (name == "missing-use-counts") {
        view.use_counts = nullptr;
    } else if (name == "generation-zero") {
        policy.generation = 0;
    } else if (name == "overlap") {
        f.x->data = reinterpret_cast<void *>(0x1000);
        f.wg->data = reinterpret_cast<void *>(0x1004);
    } else if (name == "producer-order") {
        std::swap(view.nodes[0], view.nodes[anchor]);
    } else {
        throw std::runtime_error("unknown negative test");
    }
    if (name == "producer-order") {
        anchor = 0;
    }
    auto result = PlanFFN(view, anchor, policy, &params);
    ExpectRejected(result, code, deps);
    passed.push_back(name);
}

struct Probe {
    Capabilities caps;
    FFNPolicy policy = Policy(caps);
    std::vector<GGMLFFNPlan> plans;
    std::vector<Code> rejections;
    std::exception_ptr error;
    int optimizations = 0, live_ok = 0, range_rejections = 0, callbacks = 0;
    ggml_status (*original_compute)(ggml_backend_t, ggml_cgraph *) = nullptr;
    bool omit_dependencies = false;
    int lifetime_rejections = 0;
    static void DiscardDependency(void *, ggml_tensor *, ggml_tensor *) {}
    static Probe *active;
    const ggml_tensor *unsupported = nullptr;
    ggml_backend_dev_t original_device = nullptr;
    bool (*original_support)(ggml_backend_dev_t, const ggml_tensor *) = nullptr;
    static bool Supports(ggml_backend_dev_t, const ggml_tensor *node) {
        return node != active->unsupported &&
               active->original_support(active->original_device, node);
    }

    static void Optimize(ggml_backend_t, ggml_cgraph *graph,
                         ggml_backend_graph_optimize_params *params) noexcept {
        auto &self = *active;
        try {
            ++self.optimizations;
            for (int i = 0; i < graph->n_nodes; ++i) {
                ggml_backend_graph_optimize_params control{DiscardDependency, nullptr};
                auto *allocation = self.omit_dependencies ? &control : params;
                auto result = PlanFFN(*graph, i, self.policy, allocation);
                if (auto *plan = std::get_if<GGMLFFNPlan>(&result)) {
                    self.plans.push_back(*plan);
                } else if (auto *rejected = std::get_if<RegionRejection>(&result)) {
                    self.rejections.push_back(rejected->code);
                }
            }
        } catch (...) {
            self.error = std::current_exception();
        }
    }

    static ggml_status Compute(ggml_backend_t backend, ggml_cgraph *graph) noexcept {
        auto &self = *active;
        try {
            if (self.error) {
                return GGML_STATUS_FAILED;
            }
            for (const auto &plan : self.plans) {
                auto error = CheckFFNExecution(plan, *graph, self.policy);
                if (!error) {
                    ++self.live_ok;
                } else if (error->code == Code::SplitMismatch) {
                    ++self.range_rejections;
                } else if (self.omit_dependencies && error->code == Code::AliasOrView) {
                    ++self.lifetime_rejections;
                } else {
                    throw std::runtime_error("unexpected live FFN guard rejection: " + error->message);
                }
            }
            // This probe only validates the plan. It always computes the
            // original CPU nodes; it cannot count as fused Coral execution.
            return self.original_compute(backend, graph);
        } catch (...) {
            self.error = std::current_exception();
            return GGML_STATUS_FAILED;
        }
    }
    static bool Eval(ggml_tensor *, bool ask, void *user) noexcept {
        if (!ask) {
            ++static_cast<Probe *>(user)->callbacks;
        }
        return true;
    }
    void Check() {
        if (error) {
            std::rethrow_exception(error);
        }
    }
};
Probe *Probe::active = nullptr;

using Backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
using Scheduler = std::unique_ptr<ggml_backend_sched, decltype(&ggml_backend_sched_free)>;
void Fill(ggml_tensor *tensor, float offset) {
    std::vector<float> values(ggml_nelements(tensor));
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = offset + float(i % 7) * 0.0625f;
    }
    ggml_backend_tensor_set(tensor, values.data(), 0, ggml_nbytes(tensor));
}
std::vector<float> Read(ggml_tensor *tensor) {
    std::vector<float> values(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, values.data(), 0, ggml_nbytes(tensor));
    return values;
}

void Scheduled(const std::string &name) {
    Fixture f;
    Probe probe;
    Probe::active = &probe;
    Backend cpu(ggml_backend_cpu_init(), ggml_backend_free);
    Backend other(ggml_backend_cpu_init(), ggml_backend_free);
    Require(cpu && other, "CPU test backend initialization failed");
    ggml_backend_cpu_set_n_threads(cpu.get(), 1);
    ggml_backend_cpu_set_n_threads(other.get(), 1);
    probe.original_compute = cpu->iface.graph_compute;
    cpu->iface.graph_optimize = Probe::Optimize;
    cpu->iface.graph_compute = Probe::Compute;
    // Distinct capability prevents the scheduler from upgrading the external
    // consumer back to the higher-priority CPU test backend.
    ggml_backend_device limited_device = *cpu->device;
    probe.original_device = cpu->device;
    probe.original_support = cpu->device->iface.supports_op;
    limited_device.iface.supports_op = Probe::Supports;
    cpu->device = &limited_device;
    ggml_backend_t backends[] = {cpu.get(), other.get()};
    const bool cross = name == "scheduler-cross-backend-consumer";
    Scheduler scheduler(ggml_backend_sched_new(backends, nullptr, cross ? 2 : 1, 64, false, true),
                        ggml_backend_sched_free);
    Require(scheduler != nullptr, "scheduler allocation failed");
    if (cross) {
        auto *extra = ggml_add(f.context, f.gate, f.gate);
        probe.unsupported = extra;
        ggml_set_output(extra);
        ggml_build_forward_expand(f.graph, extra);
        for (int i = 0; i < f.graph->n_nodes; ++i) {
            ggml_backend_sched_set_tensor_backend(scheduler.get(), f.graph->nodes[i],
                                                 f.graph->nodes[i] == extra ? other.get() : cpu.get());
        }
    }
    probe.omit_dependencies = name == "scheduler-lifetime-control";
    const bool observed = name == "scheduler-observed";
    const bool subdivide = name == "scheduler-callback-subdivision";
    probe.policy.evaluation_callbacks = observed;
    if (observed || subdivide) {
        ggml_backend_sched_set_eval_callback(scheduler.get(), Probe::Eval, &probe);
    }
    Require(ggml_backend_sched_alloc_graph(scheduler.get(), f.graph), "scheduler graph allocation failed");
    probe.Check();
    const int optimizations = probe.optimizations;
    Require(optimizations > 0, "upstream graph_optimize was not called");
    if (cross || observed) {
        Require(probe.plans.empty(), "unsafe scheduled region was accepted");
        const Code expected = cross ? Code::ExternalIntermediateUse : Code::DebugObservation;
        Require(std::find(probe.rejections.begin(), probe.rejections.end(), expected) != probe.rejections.end(),
                "missing scheduled region rejection");
        if (cross) {
            Require(ggml_backend_sched_get_n_splits(scheduler.get()) == 2, "fixture did not cross backends");
        }
    } else {
        Require(probe.plans.size() == 1, "expected one scheduled plan");
    }

    for (int run = 0; run < 2; ++run) {
        Fill(f.x, 0.125f + run);
        Fill(f.wg, -0.125f); Fill(f.wu, 0.25f); Fill(f.wd, 0.0625f);
        const auto original_x = Read(f.x);
        const auto status = ggml_backend_sched_graph_compute(scheduler.get(), f.graph);
        probe.Check();
        Require(status == GGML_STATUS_SUCCESS, "scheduled CPU execution failed");
        if (!probe.plans.empty() && !probe.omit_dependencies) {
            Require(original_x == Read(f.x), "allocation dependencies did not preserve X through down projection");
            Require(f.x->data != f.y->data, "output aliased the protected activation");
        }
        for (float value : Read(f.y)) {
            Require(std::isfinite(value), "CPU fixture produced non-finite output");
        }
        Require(probe.optimizations == optimizations, "graph reuse unexpectedly rebuilt the plan");
    }
    if (observed || subdivide) {
        Require(probe.callbacks == f.graph->n_nodes * 2, "evaluation callback was dropped");
    }
    if (subdivide) {
        Require(probe.live_ok == 0 && probe.range_rejections == f.graph->n_nodes * 2,
                "partial execution range was treated as a complete FFN");
    } else if (probe.omit_dependencies) {
        Require(probe.lifetime_rejections == 2 && probe.live_ok == 0,
                "control did not expose unsafe allocator reuse without dependencies");
    } else if (!cross && !observed) {
        Require(probe.live_ok == 2, "reused full execution range did not pass guards");
        const auto &plan = probe.plans.front();
        auto policy = probe.policy;
        ++policy.generation;
        auto error = CheckFFNExecution(plan, *f.graph, policy);
        Require(error && error->code == Code::StaleGraphOrSchedule, "stale schedule accepted");
        policy = probe.policy;
        policy.evaluation_callbacks = true;
        error = CheckFFNExecution(plan, *f.graph, policy);
        Require(error && error->code == Code::DebugObservation, "new callback policy was ignored");
        policy = probe.policy;
        f.hidden->op_params[15] = 1;
        error = CheckFFNExecution(plan, *f.graph, policy);
        Require(error && error->code == Code::StaleGraphOrSchedule, "changed op parameters accepted");
        f.hidden->op_params[15] = 0;
        auto *saved = f.y->data;
        f.y->data = f.x->data;
        error = CheckFFNExecution(plan, *f.graph, policy);
        Require(error && error->code == Code::AliasOrView, "live output/input alias accepted");
        f.y->data = saved;
        probe.caps.rejected = f.hidden;
        error = CheckFFNExecution(plan, *f.graph, policy);
        Require(error && error->code == Code::MissingMemberCapability, "changed member support accepted");
        passed.push_back("live-stale-generation");
        passed.push_back("live-callback-policy");
        passed.push_back("live-mutated-parameters");
        passed.push_back("live-output-alias");
        passed.push_back("live-capability-change");
    }
    passed.push_back(name);
}
} // namespace

int main() {
    try {
        Positive("split-swiglu", 0, false);
        Positive("expanded-silu-mul", 1, false);
        Positive("expanded-reversed-mul", 2, false);
        Positive("shared-weight-identity", 0, true);
        Negative("external-consumer", Code::ExternalIntermediateUse);
        Negative("external-view", Code::ExternalIntermediateUse);
        Negative("cross-split", Code::SplitMismatch);
        Negative("view-input", Code::AliasOrView);
        Negative("dtype", Code::UnsupportedDtype);
        Negative("stride", Code::UnsupportedLayout);
        Negative("shape", Code::ShapeMismatch);
        Negative("unknown-parameter", Code::UnsupportedParameters);
        Negative("extra-source", Code::UnsupportedPattern);
        Negative("observed-intermediate", Code::ExternalIntermediateUse);
        Negative("debug-callback", Code::DebugObservation);
        Negative("unknown-capability", Code::MissingMemberCapability);
        Negative("unsupported-member", Code::MissingMemberCapability);
        Negative("missing-member-artifact", Code::MissingMemberArtifact);
        Negative("missing-lifetime-hook", Code::InputLifetimeNotProven);
        Negative("missing-use-counts", Code::MissingUseInformation);
        Negative("generation-zero", Code::StaleGraphOrSchedule);
        Negative("overlap", Code::AliasOrView);
        Negative("producer-order", Code::InputsNotReady);
        Scheduled("scheduler-reuse-and-lifetime");
        Scheduled("scheduler-lifetime-control");
        Scheduled("scheduler-cross-backend-consumer");
        Scheduled("scheduler-observed");
        Scheduled("scheduler-callback-subdivision");
        std::cout << "{\"status\":\"passed\",\"cases\":[";
        for (std::size_t i = 0; i < passed.size(); ++i) {
            std::cout << (i ? "," : "") << "\"" << passed[i] << "\"";
        }
        std::cout << "],\"count\":" << passed.size() << "}\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FFN planner test: " << error.what() << "\n";
        if (!passed.empty()) {
            std::cerr << "last passed: " << passed.back() << "\n";
        }
        return 1;
    }
}
