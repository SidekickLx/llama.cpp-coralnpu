// SPDX-License-Identifier: MIT
#include "ggml_coral.h"
#include "planner/ggml_fusion.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace coral::ggml_backend {
namespace {
using runtime::DeviceContext;
using runtime::LoadedModule;
using runtime::MatmulShape;
using runtime::PreparedWeight;
using runtime::RuntimeStats;
using runtime::TensorBinding;
using ShapeKey = std::tuple<std::size_t, std::size_t, std::size_t>;
using PointwiseKey = std::tuple<contracts::PointwiseKind, std::size_t, std::size_t>;
using WeightKey = std::tuple<std::uintptr_t, std::uintptr_t, std::size_t, std::size_t>;

struct Registration {
    Options options;
    std::map<ShapeKey, std::string> artifacts;
    std::map<PointwiseKey, std::string> pointwise_artifacts;
    Stats stats;
    std::string error;
    bool registered = false;
    ggml_backend_reg reg{};
    ggml_backend_device device{};
};
Registration registration;

PointwiseKey Key(contracts::PointwiseSpec spec) { return {spec.kind, spec.rows, spec.cols}; }

ShapeKey Key(MatmulShape shape) { return {shape.m, shape.k, shape.n}; }

bool DenseMatrix(const ggml_tensor *tensor) noexcept {
    if (!tensor || tensor->type != GGML_TYPE_F32 || tensor->ne[0] <= 0 ||
        tensor->ne[1] <= 0 || tensor->ne[2] != 1 || tensor->ne[3] != 1 ||
        tensor->extra != nullptr) {
        return false;
    }
    constexpr std::uint64_t max_elements = 2147483647 / sizeof(float);
    const auto cols = static_cast<std::uint64_t>(tensor->ne[0]);
    const auto rows = static_cast<std::uint64_t>(tensor->ne[1]);
    return cols <= max_elements && rows <= max_elements / cols &&
           ggml_is_contiguous(tensor);
}

bool NoCompute(ggml_op op) noexcept {
    return op == GGML_OP_NONE || op == GGML_OP_VIEW || op == GGML_OP_RESHAPE ||
           op == GGML_OP_TRANSPOSE || op == GGML_OP_PERMUTE;
}

ggml_backend_buffer_t Buffer(const ggml_tensor *tensor) {
    return tensor->buffer ? tensor->buffer
                          : (tensor->view_src ? tensor->view_src->buffer : nullptr);
}

bool HostStorage(const ggml_tensor *tensor) {
    auto *buffer = Buffer(tensor);
    return tensor->data && buffer && ggml_backend_buffer_is_host(buffer) &&
           reinterpret_cast<std::uintptr_t>(tensor->data) % alignof(float) == 0;
}

void Accumulate(const RuntimeStats &before, const RuntimeStats &after) {
    auto &sum = registration.stats.runtime;
    sum.module_loads += after.module_loads - before.module_loads;
    sum.calls_attempted += after.calls_attempted - before.calls_attempted;
    sum.calls_completed += after.calls_completed - before.calls_completed;
    sum.coral_dispatches += after.coral_dispatches - before.coral_dispatches;
    sum.failed_dispatches += after.failed_dispatches - before.failed_dispatches;
    sum.dispatches_with_instruction_count +=
        after.dispatches_with_instruction_count - before.dispatches_with_instruction_count;
    sum.kernel_instructions += after.kernel_instructions - before.kernel_instructions;
    sum.observer_errors += after.observer_errors - before.observer_errors;
    sum.host_executable_load_attempts +=
        after.host_executable_load_attempts - before.host_executable_load_attempts;
    sum.binding_allocations += after.binding_allocations - before.binding_allocations;
    sum.upload_bytes += after.upload_bytes - before.upload_bytes;
    sum.download_bytes += after.download_bytes - before.download_bytes;
    sum.module_load_ms += after.module_load_ms - before.module_load_ms;
    sum.invoke_ms += after.invoke_ms - before.invoke_ms;
}

void RecordError(const char *message) noexcept {
    ++registration.stats.failures;
    // Even error reporting must not throw through a GGML C callback.
    try {
        registration.error = message;
    } catch (...) {
    }
}

struct BackendContext {
    DeviceContext device;
    std::map<ShapeKey, std::unique_ptr<LoadedModule>> modules;
    std::map<WeightKey, std::unique_ptr<PreparedWeight>> weights;
    std::map<PointwiseKey, std::unique_ptr<runtime::PointwiseModule>> pointwise_modules;
    bool failed = false;
    std::uint64_t optimization_generation = 0;

    void ComputePointwise(ggml_tensor *node, contracts::PointwiseSpec spec) {
        if (!HostStorage(node)) {
            throw std::runtime_error("pointwise output must have host-visible storage");
        }
        std::vector<TensorBinding> bindings;
        for (std::size_t i = 0; i < spec.Arity(); ++i) {
            const auto *source = node->src[i];
            if (!HostStorage(source)) {
                throw std::runtime_error("pointwise input must have host-visible storage");
            }
            bindings.push_back({static_cast<const float *>(source->data), spec.Elements(),
                                spec.rows, spec.cols, source->nb[1]});
        }
        const auto key = Key(spec);
        const auto artifact = registration.pointwise_artifacts.find(key);
        if (artifact == registration.pointwise_artifacts.end()) {
            throw std::runtime_error("missing pointwise artifact");
        }
        auto module = pointwise_modules.find(key);
        if (module == pointwise_modules.end()) {
            auto loaded = std::make_unique<runtime::PointwiseModule>(device, artifact->second, spec);
            module = pointwise_modules.emplace(key, std::move(loaded)).first;
        }
        const auto before = device.stats();
        const auto output = module->second->Run(bindings);
        std::memcpy(node->data, output.data(), output.size() * sizeof(float));
        ++registration.stats.pointwise_calls;
        if (registration.options.trace) {
            const auto after = device.stats();
            const NodeEvent event{node, {0, 0, 0},
                after.coral_dispatches - before.coral_dispatches,
                after.kernel_instructions - before.kernel_instructions, false,
                registration.stats.graph_calls};
            registration.options.trace(event, registration.options.trace_user_data);
        }
    }

    void Compute(ggml_tensor *node) {
        contracts::PointwiseSpec pointwise;
        if (CompatiblePointwise(node, &pointwise)) {
            ComputePointwise(node, pointwise);
            return;
        }
        MatmulShape shape;
        if (!Compatible(node, &shape) || !HostStorage(node) ||
            !HostStorage(node->src[0]) || !HostStorage(node->src[1])) {
            throw std::runtime_error("Coral graph contains an unsupported tensor binding");
        }
        const auto key = Key(shape);
        const auto artifact = registration.artifacts.find(key);
        if (artifact == registration.artifacts.end()) {
            throw std::runtime_error("Coral graph has no matching AOT artifact");
        }
        auto module = modules.find(key);
        if (module == modules.end()) {
            auto loaded = std::make_unique<LoadedModule>(device, artifact->second, shape);
            module = modules.emplace(key, std::move(loaded)).first;
        }
        const auto *w = node->src[0];
        const auto *x = node->src[1];
        TensorBinding weight_binding{static_cast<const float *>(w->data),
                                     shape.WeightElements(), shape.n, shape.k, w->nb[1]};
        TensorBinding input_binding{static_cast<const float *>(x->data),
                                    shape.InputElements(), shape.m, shape.k, x->nb[1]};

        // A WEIGHTS usage tag alone is not an immutability guarantee. The
        // application must opt in; ordinary graph inputs always get a new copy.
        const bool cacheable = registration.options.immutable_model_weights &&
                               w->op == GGML_OP_NONE && !w->view_src &&
                               ggml_backend_buffer_get_usage(Buffer(w)) ==
                                   GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
        const WeightKey weight_key{reinterpret_cast<std::uintptr_t>(Buffer(w)),
                                   reinterpret_cast<std::uintptr_t>(w->data), shape.n, shape.k};
        std::unique_ptr<PreparedWeight> temporary;
        PreparedWeight *prepared = nullptr;
        bool cache_hit = false;
        if (cacheable) {
            auto entry = weights.find(weight_key);
            cache_hit = entry != weights.end();
            if (!cache_hit) {
                auto value = std::make_unique<PreparedWeight>(device, weight_binding);
                entry = weights.emplace(weight_key, std::move(value)).first;
                ++registration.stats.weight_preparations;
            } else {
                ++registration.stats.weight_cache_hits;
            }
            prepared = entry->second.get();
        } else {
            temporary = std::make_unique<PreparedWeight>(device, weight_binding);
            prepared = temporary.get();
            ++registration.stats.weight_preparations;
        }

        const auto before = device.stats();
        auto output = module->second->Run(input_binding, *prepared);
        // Publish only this node's fully validated result. A later graph error
        // fails the whole graph; no CPU replay or success status is fabricated.
        std::memcpy(node->data, output.data(), output.size() * sizeof(float));
        ++registration.stats.matmuls;
        if (registration.options.trace) {
            const auto after = device.stats();
            const NodeEvent event{node, shape,
                                  after.coral_dispatches - before.coral_dispatches,
                                  after.kernel_instructions - before.kernel_instructions,
                                  cache_hit, registration.stats.graph_calls};
            registration.options.trace(event, registration.options.trace_user_data);
        }
    }
};

const char *BackendName(ggml_backend_t) { return "CORAL"; }

void FreeBackend(ggml_backend_t backend) {
    delete static_cast<BackendContext *>(backend->context);
    delete backend;
}

planner::MemberSupport QueryFFNMember(const ggml_tensor *node, void *);

ggml_status GraphCompute(ggml_backend_t backend, ggml_cgraph *graph) {
    auto &context = *static_cast<BackendContext *>(backend->context);
    const auto before = context.device.stats();
    ++registration.stats.graph_calls;
    try {
        if (context.failed) {
            throw std::runtime_error("Coral backend is unusable after a graph failure");
        }
        // Check the full split before starting numerical work.
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            auto *node = ggml_graph_node(graph, i);
            if (!(node->flags & GGML_TENSOR_FLAG_COMPUTE) || NoCompute(node->op)) {
                continue;
            }
            if (QueryFFNMember(node, nullptr) != planner::MemberSupport::Supported) {
                throw std::runtime_error("Coral graph contains an unsupported operation");
            }
        }
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            auto *node = ggml_graph_node(graph, i);
            if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) && !NoCompute(node->op)) {
                context.Compute(node);
            }
        }
        Accumulate(before, context.device.stats());
        return GGML_STATUS_SUCCESS;
    } catch (const std::exception &error) {
        RecordError(error.what());
    } catch (...) {
        RecordError("unknown Coral backend exception");
    }
    context.failed = true;
    Accumulate(before, context.device.stats());
    return GGML_STATUS_FAILED;
}

planner::MemberSupport QueryFFNMember(const ggml_tensor *node, void *) {
    MatmulShape shape;
    if (Compatible(node, &shape)) {
        return registration.artifacts.count(Key(shape))
            ? planner::MemberSupport::Supported : planner::MemberSupport::MissingArtifact;
    }
    contracts::PointwiseSpec pointwise;
    if (CompatiblePointwise(node, &pointwise)) {
        return registration.pointwise_artifacts.count(Key(pointwise))
            ? planner::MemberSupport::Supported : planner::MemberSupport::MissingArtifact;
    }
    return planner::MemberSupport::Unsupported;
}

void GraphOptimize(ggml_backend_t backend, ggml_cgraph *graph,
                   ggml_backend_graph_optimize_params *allocation) noexcept {
    auto &context = *static_cast<BackendContext *>(backend->context);
    ++registration.stats.graph_optimizations;
    try {
        if (context.failed) {
            return;
        }
        const planner::FFNPolicy policy{
            ++context.optimization_generation, registration.options.evaluation_callbacks,
            QueryFFNMember, nullptr};
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            auto result = planner::PlanFFN(*graph, i, policy, allocation);
            if (std::holds_alternative<std::monostate>(result)) {
                continue;
            }
            ++registration.stats.ffn_candidates;
            if (std::holds_alternative<planner::RegionRejection>(result)) {
                ++registration.stats.ffn_rejections;
            } else {
                ++registration.stats.ffn_preparation_plans;
                if (registration.options.plan_trace) {
                    registration.options.plan_trace(std::get<planner::GGMLFFNPlan>(result),
                                                    registration.options.trace_user_data);
                }
            }
        }
        // Admission only. The current backend has no full-FFN executable.
        // Keep every original node; M4 must prepare and revalidate before
        // skipping members or invoking a fused program.
    } catch (const std::exception &error) {
        context.failed = true;
        RecordError(error.what());
    } catch (...) {
        context.failed = true;
        RecordError("unknown FFN planning exception");
    }
}

ggml_backend_t InitBackend(ggml_backend_dev_t device, const char *params) {
    try {
        if (params && *params) {
            throw std::invalid_argument("Coral does not accept backend parameter strings");
        }
        auto context = std::make_unique<BackendContext>();
        auto backend = std::make_unique<::ggml_backend>();
        static ggml_guid guid = {0x47,0x63,0xa8,0x51,0x6e,0x29,0x43,0x31,
                                 0xb5,0x82,0x81,0x37,0x65,0xf0,0x21,0x3c};
        backend->guid = &guid;
        backend->iface.get_name = BackendName;
        backend->iface.free = FreeBackend;
        backend->iface.graph_compute = GraphCompute;
        backend->iface.graph_optimize = GraphOptimize;
        backend->device = device;
        backend->context = context.release();
        return backend.release();
    } catch (const std::exception &error) {
        RecordError(error.what());
    } catch (...) {
        RecordError("unknown Coral initialization exception");
    }
    return nullptr;
}

const char *DeviceName(ggml_backend_dev_t) { return "CORAL"; }
const char *Description(ggml_backend_dev_t) { return "Coral MPACT strict F32 AOT"; }
enum ggml_backend_dev_type DeviceType(ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_ACCEL; }

void Memory(ggml_backend_dev_t, std::size_t *free, std::size_t *total) {
    *free = 0;
    *total = 0;
}

void Properties(ggml_backend_dev_t device, ggml_backend_dev_props *props) {
    *props = {};
    props->name = DeviceName(device);
    props->description = Description(device);
    props->type = DeviceType(device);
    props->caps.buffer_from_host_ptr = true;
    props->caps.mmap_support = true;
}

ggml_backend_buffer_type_t BufferType(ggml_backend_dev_t) {
    return ggml_backend_cpu_buffer_type();
}

ggml_backend_buffer_t FromHost(ggml_backend_dev_t, void *ptr, std::size_t size, std::size_t) {
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);
}

bool SupportsBuffer(ggml_backend_dev_t, ggml_backend_buffer_type_t type) {
    return type && ggml_backend_buft_is_host(type);
}

bool SupportsOp(ggml_backend_dev_t, const ggml_tensor *node) {
    return QueryFFNMember(node, nullptr) == planner::MemberSupport::Supported;
}

const char *RegName(ggml_backend_reg_t) { return "CORAL"; }
std::size_t DeviceCount(ggml_backend_reg_t) { return 1; }
ggml_backend_dev_t GetDevice(ggml_backend_reg_t, std::size_t index) {
    return index == 0 ? &registration.device : nullptr;
}

std::map<ShapeKey, std::string> ReadArtifacts(const std::string &directory) {
    const std::filesystem::path root = std::filesystem::canonical(directory);
    std::ifstream input(root / "profiles.tsv");
    std::string line;
    if (!std::getline(input, line) || line != "coral-m2-v1") {
        throw std::runtime_error("missing or invalid M2 profiles.tsv header");
    }
    std::map<ShapeKey, std::string> result;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream fields(line);
        MatmulShape shape;
        std::string filename, extra;
        if (!(fields >> shape.m >> shape.k >> shape.n >> filename) || fields >> extra) {
            throw std::runtime_error("invalid M2 profile row");
        }
        shape.Validate();
        const std::filesystem::path path(filename);
        if (path.filename() != path || path.extension() != ".vmfb") {
            throw std::runtime_error("M2 profile must name a VMFB in its artifact directory");
        }
        const auto resolved = std::filesystem::canonical(root / path);
        if (resolved.parent_path() != root || !std::filesystem::is_regular_file(resolved)) {
            throw std::runtime_error("M2 artifact must be a regular file in its directory");
        }
        if (!result.emplace(Key(shape), resolved.string()).second) {
            throw std::runtime_error("duplicate M2 profile");
        }
    }
    if (result.empty() || !input.eof()) {
        throw std::runtime_error("empty or unreadable M2 profile index");
    }
    return result;
}

std::map<PointwiseKey, std::string> ReadPointwiseArtifacts(const std::string &directory) {
    const auto root = std::filesystem::canonical(directory);
    std::map<PointwiseKey, std::string> result;
    if (!std::filesystem::exists(root / "pointwise.tsv")) {
        return result; // A legacy M2 catalog grants matmul support only.
    }
    std::ifstream input(root / "pointwise.tsv");
    std::string line;
    if (!std::getline(input, line) || line != "coral-m3-pointwise-v1") {
        throw std::runtime_error("invalid pointwise catalog header");
    }
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        contracts::PointwiseSpec spec;
        std::string kind, filename, extra;
        std::istringstream fields(line);
        if (!(fields >> kind >> spec.rows >> spec.cols >> filename) || fields >> extra) {
            throw std::runtime_error("invalid pointwise catalog row");
        }
        if (kind == "silu") {
            spec.kind = contracts::PointwiseKind::Silu;
        } else if (kind == "multiply") {
            spec.kind = contracts::PointwiseKind::Multiply;
        } else if (kind == "swiglu") {
            spec.kind = contracts::PointwiseKind::SwiGLU;
        } else {
            throw std::runtime_error("unknown pointwise catalog operation");
        }
        (void)spec.Elements();
        const std::filesystem::path path(filename);
        if (path.filename() != path || path.extension() != ".vmfb") {
            throw std::runtime_error("pointwise profile must name a local VMFB");
        }
        const auto resolved = std::filesystem::canonical(root / path);
        if (resolved.parent_path() != root || !std::filesystem::is_regular_file(resolved)) {
            throw std::runtime_error("pointwise artifact must be a regular file in its directory");
        }
        if (!result.emplace(Key(spec), resolved.string()).second) {
            throw std::runtime_error("duplicate pointwise profile");
        }
    }
    if (result.empty() || !input.eof()) {
        throw std::runtime_error("empty or unreadable pointwise catalog");
    }
    return result;
}
} // namespace

bool Compatible(const ggml_tensor *op, MatmulShape *shape) noexcept {
    if (!op || op->op != GGML_OP_MUL_MAT || !DenseMatrix(op) ||
        !DenseMatrix(op->src[0]) || !DenseMatrix(op->src[1])) {
        return false;
    }
    const auto *w = op->src[0];
    const auto *x = op->src[1];
    if (w->ne[0] != x->ne[0] || op->ne[0] != w->ne[1] || op->ne[1] != x->ne[1]) {
        return false;
    }
    // Inspect every word so unknown hints or future parameter fields fail shut.
    for (std::size_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(std::int32_t); ++i) {
        const auto value = op->op_params[i];
        if (value != 0 && !((i == 0 || i == 3) && value == GGML_PREC_F32)) {
            return false;
        }
    }
    if (shape) {
        *shape = {static_cast<std::size_t>(x->ne[1]), static_cast<std::size_t>(x->ne[0]),
                  static_cast<std::size_t>(w->ne[1])};
    }
    return true;
}

bool CompatiblePointwise(const ggml_tensor *op, contracts::PointwiseSpec *spec) noexcept {
    if (!op) {
        return false;
    }
    contracts::PointwiseKind kind;
    std::size_t arity = 0;
    std::int32_t parameter = 0;
    if (op->op == GGML_OP_UNARY) {
        kind = contracts::PointwiseKind::Silu;
        arity = 1;
        parameter = GGML_UNARY_OP_SILU;
    } else if (op->op == GGML_OP_MUL) {
        kind = contracts::PointwiseKind::Multiply;
        arity = 2;
    } else if (op->op == GGML_OP_GLU) {
        kind = contracts::PointwiseKind::SwiGLU;
        arity = 2;
        parameter = GGML_GLU_OP_SWIGLU;
    } else {
        return false;
    }
    for (std::size_t word = 0; word < GGML_MAX_OP_PARAMS / sizeof(std::int32_t); ++word) {
        if (op->op_params[word] != (word == 0 ? parameter : 0)) {
            return false;
        }
    }
    for (std::size_t slot = arity; slot < GGML_MAX_SRC; ++slot) {
        if (op->src[slot]) {
            return false;
        }
    }
    for (std::size_t i = 0; i <= arity; ++i) {
        const auto *tensor = i == arity ? op : op->src[i];
        if (!DenseMatrix(tensor) || tensor->view_src || tensor->view_offs ||
            tensor->ne[0] != op->ne[0] || tensor->ne[1] != op->ne[1] ||
            tensor->nb[0] != sizeof(float) ||
            tensor->nb[1] != static_cast<std::size_t>(tensor->ne[0]) * sizeof(float) ||
            tensor->nb[2] != tensor->nb[1] * tensor->ne[1] || tensor->nb[3] != tensor->nb[2]) {
            return false;
        }
    }
    if (spec) {
        *spec = {kind, static_cast<std::size_t>(op->ne[1]), static_cast<std::size_t>(op->ne[0])};
    }
    return true;
}

void Register(const Options &options) {
    if (registration.registered) {
        throw std::logic_error("register Coral only once per process");
    }
    auto artifacts = ReadArtifacts(options.artifact_directory);
    auto pointwise = ReadPointwiseArtifacts(options.artifact_directory);
    registration.options = options;
    registration.artifacts = std::move(artifacts);
    registration.pointwise_artifacts = std::move(pointwise);
    auto &device = registration.device;
    device.reg = &registration.reg;
    device.iface.get_name = DeviceName;
    device.iface.get_description = Description;
    device.iface.get_memory = Memory;
    device.iface.get_type = DeviceType;
    device.iface.get_props = Properties;
    device.iface.init_backend = InitBackend;
    device.iface.get_buffer_type = BufferType;
    device.iface.buffer_from_host_ptr = FromHost;
    device.iface.supports_op = SupportsOp;
    device.iface.supports_buft = SupportsBuffer;
    device.iface.offload_op = SupportsOp;
    auto &reg = registration.reg;
    reg.api_version = GGML_BACKEND_API_VERSION;
    reg.iface.get_name = RegName;
    reg.iface.get_device_count = DeviceCount;
    reg.iface.get_device = GetDevice;
    registration.registered = true;
    ggml_backend_register(&reg);
}

Stats GetStats() { return registration.stats; }
const std::string &LastError() { return registration.error; }

} // namespace coral::ggml_backend
