// SPDX-License-Identifier: MIT
#include "ggml/ggml_coral.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace {
namespace adapter = coral::ggml_backend;
using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using Backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
void Require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}
struct Fixture {
    Context context{ggml_init({4 * 1024 * 1024, nullptr, true}), ggml_free};
    ggml_tensor *a = nullptr, *b = nullptr, *out = nullptr;
    ggml_cgraph *graph = nullptr;
    Fixture(int operation, int rows) {
        Require(context != nullptr, "context allocation");
        a = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 172, rows);
        b = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 172, rows);
        if (operation == 0) out = ggml_silu(context.get(), a);
        if (operation == 1) out = ggml_mul(context.get(), a, b);
        if (operation == 2) out = ggml_swiglu_split(context.get(), a, b);
        graph = ggml_new_graph(context.get());
        ggml_build_forward_expand(graph, out);
    }
};
int checks = 0;
void Query(ggml_backend_dev_t device, const ggml_tensor *node, bool structural, bool supported) {
    Require(adapter::CompatiblePointwise(node) == structural, "structural capability mismatch");
    for (int repeat = 0; repeat < 3; ++repeat) {
        Require(ggml_backend_dev_supports_op(device, node) == supported, "device capability mismatch");
    }
    ++checks;
}
void Capabilities(ggml_backend_dev_t device, const std::filesystem::path &catalog) {
    // This test receives its own copied directory. Registration has consumed
    // only metadata. Rename artifacts to prove subsequent queries perform no IO.
    std::vector<std::filesystem::path> paths;
    for (const auto &entry : std::filesystem::directory_iterator(catalog)) paths.push_back(entry.path());
    for (const auto &path : paths) {
        std::filesystem::rename(path, path.string() + ".retained");
    }
    for (int operation = 0; operation < 3; ++operation) {
        Fixture f(operation, 1);
        Query(device, f.out, true, true);
        const auto saved = *f.out;
        const int arity = operation == 0 ? 1 : 2;
        for (int slot = 0; slot <= arity; ++slot) {
            auto *tensor = slot == arity ? f.out : f.out->src[slot];
            const auto original = *tensor;
            tensor->type = GGML_TYPE_F16;
            Query(device, f.out, false, false);
            *tensor = original;
            tensor->nb[1] += sizeof(float);
            Query(device, f.out, false, false);
            *tensor = original;
            tensor->nb[3] += sizeof(float);
            Query(device, f.out, false, false);
            *tensor = original;
            tensor->view_src = f.b;
            Query(device, f.out, false, false);
            *tensor = original;
            tensor->view_offs = 4;
            Query(device, f.out, false, false);
            *tensor = original;
            tensor->ne[2] = 2;
            Query(device, f.out, false, false);
            *tensor = original;
        }
        for (std::size_t word = 0; word < GGML_MAX_OP_PARAMS / sizeof(std::int32_t); ++word) {
            f.out->op_params[word] = 123456;
            Query(device, f.out, false, false);
            *f.out = saved;
        }
        f.out->src[arity] = f.b;
        Query(device, f.out, false, false);
        *f.out = saved;
        f.out->src[0] = nullptr;
        Query(device, f.out, false, false);
        *f.out = saved;
        f.a->ne[0] = 1;
        Query(device, f.out, false, false);
        Fixture absent(operation, 2);
        Query(device, absent.out, true, false);
    }
    Require(adapter::GetStats().runtime.module_loads == 0, "query loaded a module");
}
double Execute(ggml_backend_dev_t device) {
    Backend coral(ggml_backend_dev_init(device, nullptr), ggml_backend_free);
    Backend cpu(ggml_backend_cpu_init(), ggml_backend_free);
    Require(coral && cpu, "backend initialization");
    double maximum = 0;
    for (int rows : {1, 5}) {
        for (int operation = 0; operation < 3; ++operation) {
            Fixture f(operation, rows);
            Buffer storage(ggml_backend_alloc_ctx_tensors(f.context.get(), cpu.get()), ggml_backend_buffer_free);
            Require(storage != nullptr, "host storage allocation");
            Query(device, f.out, true, true);
            const auto count = static_cast<std::size_t>(172 * rows);
            std::vector<float> a(count), b(count), reference(count), actual(count);
            for (int repeat = 0; repeat < 3; ++repeat) {
                for (std::size_t i = 0; i < count; ++i) {
                    a[i] = static_cast<float>(static_cast<int>((i + repeat * 11) % 31) - 15) / 3;
                    b[i] = static_cast<float>(static_cast<int>((i * 7 + repeat) % 41) - 20) / 7;
                }
                ggml_backend_tensor_set(f.a, a.data(), 0, count * sizeof(float));
                ggml_backend_tensor_set(f.b, b.data(), 0, count * sizeof(float));
                Require(ggml_backend_graph_compute(cpu.get(), f.graph) == GGML_STATUS_SUCCESS, "CPU execute");
                ggml_backend_tensor_get(f.out, reference.data(), 0, count * sizeof(float));
                ggml_backend_tensor_memset(f.out, 0, 0, count * sizeof(float));
                Require(ggml_backend_graph_compute(coral.get(), f.graph) == GGML_STATUS_SUCCESS, "Coral execute");
                ggml_backend_tensor_get(f.out, actual.data(), 0, count * sizeof(float));
                for (std::size_t i = 0; i < count; ++i) {
                    const double error = std::abs(static_cast<double>(actual[i]) - reference[i]);
                    maximum = std::max(maximum, error);
                    Require(std::isfinite(actual[i]) && error <= 2e-6 + 1e-5 * std::abs(reference[i]),
                            "backend numerical mismatch");
                }
                ++checks;
            }
        }
    }
    Require(adapter::GetStats().pointwise_calls == 18, "incomplete standalone members");
    return maximum;
}
void Preflight(ggml_backend_dev_t device) {
    Backend coral(ggml_backend_dev_init(device, nullptr), ggml_backend_free);
    Backend cpu(ggml_backend_cpu_init(), ggml_backend_free);
    Fixture f(2, 1);
    Buffer storage(ggml_backend_alloc_ctx_tensors(f.context.get(), cpu.get()), ggml_backend_buffer_free);
    Require(coral && cpu && storage, "preflight setup");
    std::vector<float> marker(172, 123.0f), actual(172);
    ggml_backend_tensor_set(f.out, marker.data(), 0, marker.size() * sizeof(float));
    f.out->op_params[2] = 1;
    Require(ggml_backend_graph_compute(coral.get(), f.graph) == GGML_STATUS_FAILED,
            "unsupported operation executed");
    ggml_backend_tensor_get(f.out, actual.data(), 0, actual.size() * sizeof(float));
    Require(actual == marker, "preflight failure changed output");
    Require(adapter::GetStats().runtime.calls_attempted == 0, "preflight dispatched work");
    checks += 3;
}
} // namespace
int main(int argc, char **argv) {
    try {
        if (argc != 3) throw std::invalid_argument("Usage: member-adapter-test MODE CATALOG");
        adapter::Options options;
        options.artifact_directory = argv[2];
        adapter::Register(options);
        const auto device = ggml_backend_dev_by_name("CORAL");
        const std::string mode(argv[1]);
        double error = 0;
        if (mode == "capabilities") Capabilities(device, argv[2]);
        else if (mode == "execute") error = Execute(device);
        else if (mode == "preflight") Preflight(device);
        else if (mode != "catalog") throw std::invalid_argument("unknown test mode");
        const auto stats = adapter::GetStats();
        Require(stats.failures == (mode == "preflight" ? 1u : 0u) &&
                stats.runtime.host_executable_load_attempts == 0, "runtime failures");
        std::cout << std::setprecision(12)
                  << "{\"status\":\"passed\",\"checks\":" << checks << ",\"max_abs_error\":" << error
                  << ",\"pointwise_calls\":" << stats.pointwise_calls
                  << ",\"dispatches\":" << stats.runtime.coral_dispatches
                  << ",\"instructions\":" << stats.runtime.kernel_instructions << "}\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
