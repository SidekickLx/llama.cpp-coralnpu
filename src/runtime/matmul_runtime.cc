// SPDX-License-Identifier: MIT
#include "matmul_runtime.h"
#include "ffn_runtime.h"
#include "artifacts/ffn_artifact.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/local_sync/sync_device.h"
#include "iree/modules/hal/module.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/module.h"
#include "metrics/runtime/stats_collector.h"
#include "runtime/driver/coralnpu_device.h"
#include "runtime/sim/simulator_backend.h"

namespace coral::runtime {
namespace {

using Clock = std::chrono::steady_clock;
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);

template <typename T, void (*Release)(T *)> class Handle {
  public:
    Handle() = default;
    ~Handle() { reset(); }
    Handle(const Handle &) = delete;
    Handle &operator=(const Handle &) = delete;
    Handle(Handle &&other) noexcept : value_(std::exchange(other.value_, nullptr)) {}
    Handle &operator=(Handle &&other) noexcept {
        if (this != &other) {
            reset();
            value_ = std::exchange(other.value_, nullptr);
        }
        return *this;
    }
    T *get() const { return value_; }
    T **put() {
        reset();
        return &value_;
    }
    void reset() {
        if (value_) {
            Release(value_);
            value_ = nullptr;
        }
    }

  private:
    T *value_ = nullptr;
};

using Instance = Handle<iree_vm_instance_t, iree_vm_instance_release>;
using Module = Handle<iree_vm_module_t, iree_vm_module_release>;
using Context = Handle<iree_vm_context_t, iree_vm_context_release>;
using List = Handle<iree_vm_list_t, iree_vm_list_release>;
using BufferView = Handle<iree_hal_buffer_view_t, iree_hal_buffer_view_release>;
using Device = Handle<iree_hal_device_t, iree_hal_device_release>;
using Allocator = Handle<iree_hal_allocator_t, iree_hal_allocator_release>;
using DeviceGroup = Handle<iree_hal_device_group_t, iree_hal_device_group_release>;
using Frontier = Handle<iree_async_frontier_tracker_t, iree_async_frontier_tracker_release>;
using ProactorPool = Handle<iree_async_proactor_pool_t, iree_async_proactor_pool_release>;

struct StatusOwner {
    iree_status_t value;
    ~StatusOwner() { iree_status_ignore(value); }
};

struct FreeIreeString {
    void operator()(char *value) const { iree_allocator_free(iree_allocator_system(), value); }
};

void Check(iree_status_t status, const char *operation) {
    if (iree_status_is_ok(status)) {
        return;
    }
    StatusOwner owner{status};
    auto allocator = iree_allocator_system();
    char *raw_message = nullptr;
    iree_host_size_t length = 0;
    const bool formatted = iree_status_to_string(status, &allocator, &raw_message, &length);
    std::unique_ptr<char, FreeIreeString> message(raw_message);
    const std::string detail = formatted ? std::string(message.get(), length)
                                         : iree_status_code_string(iree_status_code(status));
    throw std::runtime_error(std::string(operation) + ": " + detail);
}

std::size_t ElementCount(std::size_t rows, std::size_t cols) {
    // The pinned device target is RV32. M1 deliberately bounds every tensor
    // to signed 32-bit byte addressing as well as validating host arithmetic.
    constexpr auto max_bytes = std::size_t{2147483647};
    if (rows == 0 || cols == 0 || rows > max_bytes || cols > max_bytes ||
        rows > (max_bytes / sizeof(float)) / cols) {
        throw std::invalid_argument("invalid or oversized F32 matrix shape");
    }
    return rows * cols;
}

double ElapsedMs(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

std::string ToString(iree_string_view_t value) {
    return value.size == 0 ? std::string() : std::string(value.data, value.size);
}

std::vector<std::uint8_t> ReadModule(const std::string &path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("cannot open VMFB: " + path);
    }
    const auto length = input.tellg();
    constexpr std::streamoff max_module_bytes = 512 * 1024 * 1024;
    if (length <= 0 || length > max_module_bytes) {
        throw std::runtime_error("VMFB must be nonempty and at most 512 MiB");
    }
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(length));
    input.seekg(0);
    if (!input.read(reinterpret_cast<char *>(bytes.data()),
                    static_cast<std::streamsize>(bytes.size()))) {
        throw std::runtime_error("cannot read complete VMFB: " + path);
    }
    return bytes;
}

// Advertise the host executable format so mixed-target device selection works,
// but refuse every load. This permits host buffer services without providing
// any implementation capable of executing a host numerical kernel.
struct RejectHostLoader : iree_hal_executable_loader_t {
    std::uint64_t *attempts = nullptr;

    static void Destroy(iree_hal_executable_loader_t *base) {
        delete static_cast<RejectHostLoader *>(base);
    }
    static iree_status_t Infer(iree_hal_executable_loader_t *, iree_hal_executable_caching_mode_t,
                               iree_const_byte_span_t, iree_host_size_t, char *,
                               iree_host_size_t *size) {
        *size = 0;
        return iree_make_status(IREE_STATUS_UNAVAILABLE,
                                "M1 does not infer or load host executable code");
    }
    static bool Query(iree_hal_executable_loader_t *, iree_hal_executable_caching_mode_t,
                      iree_string_view_t format) {
        return iree_string_view_starts_with(format, IREE_SV("embedded-elf-" IREE_ARCH));
    }
    static iree_status_t Load(iree_hal_executable_loader_t *base,
                              const iree_hal_executable_params_t *, iree_host_size_t,
                              iree_hal_executable_t **output) {
        *output = nullptr;
        ++*static_cast<RejectHostLoader *>(base)->attempts;
        return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "M1 strict mode rejects host numerical executables");
    }
};

using Loader = Handle<iree_hal_executable_loader_t, iree_hal_executable_loader_release>;

struct GroupBuilder {
    iree_hal_device_group_builder_t value{};
    ~GroupBuilder() { iree_hal_device_group_builder_deinitialize(&value); }
};

struct ClearCallLists {
    iree_vm_list_t *inputs;
    iree_vm_list_t *outputs;
    ~ClearCallLists() {
        iree_vm_list_clear(inputs);
        iree_vm_list_clear(outputs);
    }
};

void PushBuffer(iree_vm_list_t *list, iree_hal_buffer_view_t *view) {
    auto ref = iree_hal_buffer_view_retain_ref(view);
    const auto status = iree_vm_list_push_ref_move(list, &ref);
    // push_ref_move clears ref on success; release the retained reference
    // ourselves if insertion failed.
    iree_vm_ref_release(&ref);
    Check(status, "bind input buffer");
}

} // namespace

namespace detail {

struct DeviceState {
    RuntimeStats counters;
    bool poisoned = false;
    // Declaration order matters: modules and device handles are released
    // before the group. The group owns the final device references/topology.
    Instance instance;
    ProactorPool pool;
    Allocator heap_allocator;
    Loader rejecting_host_loader;
    Frontier frontier;
    DeviceGroup group;
    Device host_device;
    Device coral_device;
    Module hal_module;

    void RequireHealthy() const {
        if (poisoned) {
            throw std::runtime_error(
                "context unusable after execution failure; create a new DeviceContext");
        }
    }

    static iree_status_t BeginDispatch(void *self, void *,
                                       const iree_hal_coralnpu_dispatch_info_t *,
                                       iree_hal_coralnpu_run_hooks_t *hooks) {
        auto &state = *static_cast<DeviceState *>(self);
        auto *collector = new (std::nothrow) iree_hal_coralnpu_stats_collector_t;
        if (!collector) {
            ++state.counters.observer_errors;
            return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                    "cannot allocate M1 dispatch collector");
        }
        iree_hal_coralnpu_stats_collector_initialize(0, nullptr, collector);
        *hooks = iree_hal_coralnpu_stats_collector_hooks(collector);
        return iree_ok_status();
    }

    static iree_status_t EndDispatch(void *self, void *, const iree_hal_coralnpu_dispatch_info_t *,
                                     const iree_hal_coralnpu_run_hooks_t *hooks,
                                     iree_status_code_t status) {
        auto &state = *static_cast<DeviceState *>(self);
        std::unique_ptr<iree_hal_coralnpu_stats_collector_t> collector(
            static_cast<iree_hal_coralnpu_stats_collector_t *>(hooks->self));
        ++state.counters.coral_dispatches;
        if (status != IREE_STATUS_OK) {
            ++state.counters.failed_dispatches;
        }
        if (collector->stats.valid_fields & IREE_HAL_CORALNPU_DISPATCH_STATS_FIELD_INSTRET_KERNEL) {
            ++state.counters.dispatches_with_instruction_count;
            state.counters.kernel_instructions += collector->stats.instret_kernel;
        }
        return iree_ok_status();
    }

    void Initialize() {
        // Set up the shared runtime infrastructure. LoadedModule will load a
        // VMFB and create its VM context later; no matmul runs in this method.
        // Check() throws on failure, and the owning Handle members release any
        // resources already created when DeviceState is destroyed.
        //
        // The system allocator manages runtime objects on the host. Tensor
        // storage is exposed through a separate HAL allocator below.
        const auto allocator = iree_allocator_system();

        // 1. Create the VM instance and register HAL reference types, such as
        // buffers and buffer views, so VM modules can exchange HAL objects.
        Check(iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT, allocator, instance.put()),
              "create VM instance");
        Check(iree_hal_module_register_all_types(instance.get()), "register HAL types");

        // 2. Provide the completion-polling infrastructure used by the devices.
        // A proactor services asynchronous operations and their completions.
        // The pool has one lazily created entry shared by our two devices;
        // this count is neither the number of devices nor matmul workers.
        // Default native options supply a poll thread when the entry is used.
        Check(iree_async_proactor_pool_create(
                  1, nullptr, iree_async_proactor_pool_options_default(), allocator, pool.put()),
              "create proactor pool");

        // Both devices use the same host-visible HAL heap in this simulator
        // setup. The two allocator arguments fund HAL metadata and buffer
        // contents, respectively. This does not allocate Coral TCM or promise
        // zero-copy access inside MPACT; its backend handles device staging.
        Check(iree_hal_allocator_create_heap(IREE_SV("m1-host"), allocator, allocator,
                                             heap_allocator.put()),
              "create host-visible allocator");

        // Pass the shared pool through the common device creation parameters.
        // Synchronous application calls still use this device infrastructure.
        auto create_params = iree_hal_device_create_params_default();
        create_params.proactor_pool = pool.get();

        // 3. Create the host device for the compiled module's local target and
        // for host buffer services. It runs on the host CPU, not on RISC-V.
        // Our strict policy forbids loading host numerical kernels: a matmul
        // accidentally assigned to the host must fail instead of passing as
        // successful Coral execution.
        iree_hal_sync_device_params_t host_params;
        iree_hal_sync_device_params_initialize(&host_params);

        // Query advertises the host executable format for device matching,
        // while Load rejects every executable. The static vtable must remain
        // valid until the loader's final reference is released.
        static const iree_hal_executable_loader_vtable_t rejecting_vtable = {
            RejectHostLoader::Destroy, RejectHostLoader::Infer, RejectHostLoader::Query,
            RejectHostLoader::Load};
        auto *rejecting = new RejectHostLoader{};
        iree_hal_executable_loader_initialize(
            &rejecting_vtable, iree_hal_executable_import_provider_null(), rejecting);
        rejecting->attempts = &counters.host_executable_load_attempts;

        // Adopt the loader's initial reference into an owning Handle. Its
        // counter pointer stays valid because the counters member outlives
        // the devices and loader, as specified by member declaration order.
        *rejecting_host_loader.put() = rejecting;
        auto *loader = rejecting_host_loader.get();
        Check(iree_hal_sync_device_create(IREE_SV("local-sync"), &host_params, &create_params, 1,
                                          &loader, heap_allocator.get(), allocator,
                                          host_device.put()),
              "create host device with rejecting executable loader");

        // 4. Create the Coral device with an explicit MPACT backend.
        // The observer wraps each dispatch to collect completion status and
        // simulated kernel instruction counts. Run() checks that evidence
        // before accepting an output; numerical agreement alone is not enough.
        iree_hal_coralnpu_device_params_t coral_params;
        iree_hal_coralnpu_device_params_initialize(&coral_params);
        coral_params.dispatch_observer.self = this;
        coral_params.dispatch_observer.begin_dispatch = BeginDispatch;
        coral_params.dispatch_observer.end_dispatch = EndDispatch;

        // Select MPACT directly rather than choosing a backend from the
        // environment. MPACT executes the compiled Coral RISC-V instructions.
        const auto backend = iree_hal_coralnpu_simulator_backend_make(
            coralnpu_simulator_mpact_create, IREE_SV("mpact"));
        Check(iree_hal_coralnpu_device_create(IREE_SV("coralnpu"), &coral_params, &create_params,
                                              &backend, heap_allocator.get(), allocator,
                                              coral_device.put()),
              "create Coral MPACT device");

        // 5. Assemble one device group containing both logical devices.
        // The frontier tracker records progress along execution timelines and
        // resolves waits once the required progress has been reached.
        Check(iree_async_frontier_tracker_create(iree_async_frontier_tracker_options_default(),
                                                 allocator, frontier.put()),
              "create device frontier tracker");

        // The builder retains the devices while assembling their topology.
        // Finalization transfers those references to the group; GroupBuilder
        // also releases temporary state if an earlier operation fails.
        GroupBuilder builder;
        iree_hal_device_group_builder_initialize(&builder.value, frontier.get());
        Check(iree_hal_device_group_builder_add_device(&builder.value, host_device.get()),
              "add host device");
        Check(iree_hal_device_group_builder_add_device(&builder.value, coral_device.get()),
              "add Coral device");
        Check(iree_hal_device_group_builder_finalize(&builder.value, allocator, group.put()),
              "finalize device group");

        // 6. Expose the group's HAL services as a native VM module. VMFB code
        // imports these functions for buffer management, dispatch and waits.
        // The HAL module retains the group and will be linked with each VMFB
        // by LoadedModule. SYNCHRONOUS makes HAL waits block the calling thread
        // instead of yielding the VM as a coroutine; it does not remove the
        // devices' internal asynchronous infrastructure. HAL debug output goes
        // to stderr so it stays separate from the runner's JSON on stdout.
        Check(iree_hal_module_create(instance.get(), iree_hal_module_device_policy_default(),
                                     group.get(), IREE_HAL_MODULE_FLAG_SYNCHRONOUS,
                                     iree_hal_module_debug_sink_stdio(stderr), allocator,
                                     hal_module.put()),
              "create multi-device HAL module");
    }

    BufferView Upload(TensorBinding binding) {
        BufferView result;
        const iree_hal_dim_t dims[] = {static_cast<iree_hal_dim_t>(binding.rows),
                                       static_cast<iree_hal_dim_t>(binding.cols)};
        iree_hal_buffer_params_t params{};
        params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
        params.usage = IREE_HAL_BUFFER_USAGE_DEFAULT;
        Check(iree_hal_buffer_view_allocate_buffer_copy(
                  host_device.get(), heap_allocator.get(), 2, dims, IREE_HAL_ELEMENT_TYPE_FLOAT_32,
                  IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, params,
                  iree_make_const_byte_span(binding.data, binding.element_count * sizeof(float)),
                  result.put()),
              "allocate and upload F32 tensor");
        ++counters.binding_allocations;
        counters.upload_bytes += binding.element_count * sizeof(float);
        return result;
    }
};

struct WeightState {
    std::shared_ptr<DeviceState> device;
    std::size_t rows = 0;
    std::size_t cols = 0;
    BufferView view;
};

struct FFNWeightState {
    contracts::FFNWeightLayout layout = contracts::FFNWeightLayout::Native;
    PreparedWeight gate;
    PreparedWeight up;
    PreparedWeight down;
    FFNWeightState(const DeviceContext &device, TensorBinding gate_binding,
                   TensorBinding up_binding, TensorBinding down_binding)
        : gate(device, gate_binding), up(device, up_binding), down(device, down_binding) {}
};

struct ModuleState {
    std::shared_ptr<DeviceState> device;
    MatmulShape shape;
    contracts::PointwiseSpec pointwise;
    contracts::FFNCompileSpec ffn;
    std::string program_sha256;
    std::vector<BufferView> arguments;
    std::size_t output_rows = 0, output_cols = 0;
    void Initialize(const std::string &path, const std::string &entry,
                    const char *abi_key, const std::string &abi, std::size_t arity,
                    std::vector<std::uint8_t> verified_bytecode = {});
    void UpdateActivation(TensorBinding binding);
    std::vector<float> Invoke();
    // The bytecode module borrows these bytes. Context/module handles must
    // be destroyed before the vector; vector storage is never changed.
    std::vector<std::uint8_t> bytecode;
    Module module;
    Context context;
    BufferView activation;
    List inputs;
    List outputs;
    iree_vm_function_t function{};
};

} // namespace detail

void MatmulShape::Validate() const {
    (void)InputElements();
    (void)WeightElements();
    (void)OutputElements();
}

std::size_t MatmulShape::InputElements() const { return ElementCount(m, k); }
std::size_t MatmulShape::WeightElements() const { return ElementCount(n, k); }
std::size_t MatmulShape::OutputElements() const { return ElementCount(m, n); }

std::string MatmulShape::AbiTag() const {
    Validate();
    return "v1;f32;row-major;X=" + std::to_string(m) + "x" + std::to_string(k) +
           ";W=" + std::to_string(n) + "x" + std::to_string(k) + ";Y=" + std::to_string(m) + "x" +
           std::to_string(n);
}

TensorBinding TensorBinding::Dense(const std::vector<float> &values, std::size_t rows,
                                   std::size_t cols) {
    (void)ElementCount(rows, cols);
    TensorBinding binding{values.data(), values.size(), rows, cols, cols * sizeof(float)};
    binding.Validate(rows, cols);
    return binding;
}

void TensorBinding::Validate(std::size_t expected_rows, std::size_t expected_cols) const {
    const auto count = ElementCount(expected_rows, expected_cols);
    if (!data || rows != expected_rows || cols != expected_cols || element_count != count ||
        row_stride_bytes != cols * sizeof(float)) {
        throw std::invalid_argument(
            "binding must be non-null, exact-shape, contiguous row-major F32");
    }
}

DeviceContext::DeviceContext() : state_(std::make_shared<detail::DeviceState>()) {
    state_->Initialize();
}
DeviceContext::~DeviceContext() = default;
void DeviceContext::Reset() {
    auto replacement = std::make_shared<detail::DeviceState>();
    replacement->Initialize();
    state_ = std::move(replacement);
}
RuntimeStats DeviceContext::stats() const { return state_->counters; }

PreparedWeight::PreparedWeight(const DeviceContext &device, TensorBinding weights)
    : state_(std::make_unique<detail::WeightState>()) {
    weights.Validate(weights.rows, weights.cols);
    state_->device = device.state_;
    state_->device->RequireHealthy();
    state_->rows = weights.rows;
    state_->cols = weights.cols;
    state_->view = state_->device->Upload(weights);
}
PreparedWeight::~PreparedWeight() = default;
PreparedWeight::PreparedWeight(PreparedWeight &&) noexcept = default;
PreparedWeight &PreparedWeight::operator=(PreparedWeight &&) noexcept = default;

LoadedModule::LoadedModule(const DeviceContext &device, const std::string &vmfb_path,
                           MatmulShape shape, const std::string &function)
    : state_(std::make_unique<detail::ModuleState>()) {
    shape.Validate();
    if (function.empty() || function.find('.') != std::string::npos) {
        throw std::invalid_argument("function must be an unqualified export name");
    }
    state_->device = device.state_;
    auto &owner = *state_->device;
    owner.RequireHealthy();
    state_->shape = shape;
    state_->output_rows = shape.m;
    state_->output_cols = shape.n;
    state_->Initialize(vmfb_path, function, "coral.m1.abi", shape.AbiTag(), 2);
}

void detail::ModuleState::Initialize(const std::string &path, const std::string &entry,
                                    const char *abi_key, const std::string &abi,
                                    std::size_t arity,
                                    std::vector<std::uint8_t> verified_bytecode) {
    auto &owner = *device;
    owner.RequireHealthy();
    const auto start = Clock::now();
    // Legacy matmul/pointwise callers supply a file. FFN callers supply the
    // exact owned snapshot returned by the artifact validator, never a path.
    bytecode = verified_bytecode.empty() ? ReadModule(path) : std::move(verified_bytecode);
    const auto allocator = iree_allocator_system();
    Check(iree_vm_bytecode_module_create(
              owner.instance.get(), IREE_VM_BYTECODE_MODULE_FLAG_NONE,
              iree_make_const_byte_span(bytecode.data(), bytecode.size()),
              iree_allocator_null(), allocator, module.put()),
          "load VMFB");
    Check(iree_vm_module_lookup_function_by_name(
              module.get(), IREE_VM_FUNCTION_LINKAGE_EXPORT,
              iree_make_cstring_view(entry.c_str()), &function),
          "find numerical program export");
    const auto actual_abi = ToString(iree_vm_function_lookup_attr_by_name(
        &function, iree_make_cstring_view(abi_key)));
    if (actual_abi != abi) {
        throw std::runtime_error("VMFB ABI mismatch: expected '" + abi + "', got '" + actual_abi + "'");
    }
    const auto signature = iree_vm_function_signature(&function);
    const std::string expected = "0" + std::string(arity, 'r') + "_r";
    if (ToString(signature.calling_convention) != expected) {
        throw std::runtime_error("unexpected synchronous tensor calling convention");
    }

    iree_vm_module_t *modules[] = {owner.hal_module.get(), module.get()};
    const auto status =
        iree_vm_context_create_with_modules(owner.instance.get(), IREE_VM_CONTEXT_FLAG_NONE, 2,
                                            modules, allocator, context.put());
    if (!iree_status_is_ok(status)) {
        owner.poisoned = true; // A module initializer may already have run.
        Check(status, "initialize VM context (host numerical kernels are disabled)");
    }
    Check(
        iree_vm_list_create(iree_vm_make_undefined_type_def(), arity, allocator, inputs.put()),
        "create input list");
    Check(
        iree_vm_list_create(iree_vm_make_undefined_type_def(), 1, allocator, outputs.put()),
        "create output list");
    ++owner.counters.module_loads;
    owner.counters.module_load_ms += ElapsedMs(start);
}

LoadedModule::~LoadedModule() = default;
LoadedModule::LoadedModule(LoadedModule &&) noexcept = default;
LoadedModule &LoadedModule::operator=(LoadedModule &&) noexcept = default;

std::vector<float> LoadedModule::Run(TensorBinding activation, const PreparedWeight &weights) {
    if (!state_ || !weights.state_) {
        throw std::invalid_argument("cannot use a moved-from module or weight");
    }
    auto &owner = *state_->device;
    owner.RequireHealthy();
    const auto shape = state_->shape;
    activation.Validate(shape.m, shape.k);
    if (weights.state_->device != state_->device || weights.state_->rows != shape.n ||
        weights.state_->cols != shape.k) {
        throw std::invalid_argument("weight belongs to another context or shape");
    }
    ClearCallLists clear{state_->inputs.get(), state_->outputs.get()};
    iree_vm_list_clear(clear.inputs);
    iree_vm_list_clear(clear.outputs);
    ++owner.counters.calls_attempted;
    try {
        state_->UpdateActivation(activation);
        PushBuffer(clear.inputs, state_->activation.get());
        PushBuffer(clear.inputs, weights.state_->view.get());

        return state_->Invoke();
    } catch (...) {
        // No partial output is returned and no implicit CPU replay is attempted.
        owner.poisoned = true;
        throw;
    }
}

void detail::ModuleState::UpdateActivation(TensorBinding binding) {
    auto &owner = *device;
    if (!activation.get()) {
        activation = owner.Upload(binding);
    } else {
        Check(iree_hal_device_transfer_h2d(
                  owner.host_device.get(), binding.data,
                  iree_hal_buffer_view_buffer(activation.get()), 0,
                  binding.element_count * sizeof(float),
                  IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()),
              "update activation buffer");
        owner.counters.upload_bytes += binding.element_count * sizeof(float);
    }
}

std::vector<float> detail::ModuleState::Invoke() {
    auto &owner = *device;
    std::vector<float> result(ElementCount(output_rows, output_cols));
    const auto before = owner.counters;
    const auto start = Clock::now();
    const auto status =
        iree_vm_invoke(context.get(), function, IREE_VM_INVOCATION_FLAG_NONE,
                       nullptr, inputs.get(), outputs.get(), iree_allocator_system());
    owner.counters.invoke_ms += ElapsedMs(start);
    Check(status, "invoke Coral program");
    const auto dispatches = owner.counters.coral_dispatches - before.coral_dispatches;
    const auto samples = owner.counters.dispatches_with_instruction_count -
                         before.dispatches_with_instruction_count;
    if (dispatches == 0 || samples != dispatches ||
        owner.counters.kernel_instructions == before.kernel_instructions ||
        owner.counters.observer_errors != before.observer_errors ||
        owner.counters.failed_dispatches != before.failed_dispatches) {
        throw std::runtime_error(
            "invocation lacks complete, successful Coral kernel execution evidence");
    }
    if (iree_vm_list_size(outputs.get()) != 1) {
        throw std::runtime_error("program must return exactly one tensor");
    }
    auto *output = iree_vm_list_get_buffer_view_assign(outputs.get(), 0);
    if (!output || iree_hal_buffer_view_shape_rank(output) != 2 ||
        iree_hal_buffer_view_shape_dim(output, 0) != static_cast<iree_hal_dim_t>(output_rows) ||
        iree_hal_buffer_view_shape_dim(output, 1) != static_cast<iree_hal_dim_t>(output_cols) ||
        iree_hal_buffer_view_element_type(output) != IREE_HAL_ELEMENT_TYPE_FLOAT_32 ||
        iree_hal_buffer_view_encoding_type(output) != IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR ||
        iree_hal_buffer_view_byte_length(output) != result.size() * sizeof(float)) {
        throw std::runtime_error("output violates the F32 [M,N] tensor ABI");
    }
    // Both HAL devices use the shared host-visible heap in M1. This copy
    // measures HAL -> caller traffic, not the simulator's internal staging.
    Check(iree_hal_device_transfer_d2h(
              owner.host_device.get(), iree_hal_buffer_view_buffer(output), 0, result.data(),
              result.size() * sizeof(float), IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
              iree_infinite_timeout()),
          "read back output");
    owner.counters.download_bytes += result.size() * sizeof(float);
    ++owner.counters.calls_completed;
    return result;
}

PointwiseModule::PointwiseModule(const DeviceContext &device, const std::string &path,
                                 contracts::PointwiseSpec spec)
    : state_(std::make_unique<detail::ModuleState>()) {
    (void)spec.Elements();
    state_->device = device.state_;
    state_->pointwise = spec;
    state_->output_rows = spec.rows;
    state_->output_cols = spec.cols;
    state_->Initialize(path, spec.EntryPoint(), contracts::kPointwiseAbiKey,
                       spec.AbiTag(), spec.Arity());
    state_->arguments.resize(spec.Arity());
}
PointwiseModule::~PointwiseModule() = default;
PointwiseModule::PointwiseModule(PointwiseModule &&) noexcept = default;
PointwiseModule &PointwiseModule::operator=(PointwiseModule &&) noexcept = default;

std::vector<float> PointwiseModule::Run(const std::vector<TensorBinding> &bindings) {
    if (!state_) {
        throw std::invalid_argument("cannot use a moved-from pointwise module");
    }
    auto &owner = *state_->device;
    owner.RequireHealthy();
    const auto &spec = state_->pointwise;
    if (bindings.size() != spec.Arity()) {
        throw std::invalid_argument("wrong number of pointwise inputs");
    }
    for (const auto &binding : bindings) {
        binding.Validate(spec.rows, spec.cols);
    }
    ClearCallLists clear{state_->inputs.get(), state_->outputs.get()};
    iree_vm_list_clear(clear.inputs);
    iree_vm_list_clear(clear.outputs);
    ++owner.counters.calls_attempted;
    try {
        for (std::size_t i = 0; i < bindings.size(); ++i) {
            const auto &binding = bindings[i];
            auto &view = state_->arguments[i];
            if (!view.get()) {
                view = owner.Upload(binding);
            } else {
                Check(iree_hal_device_transfer_h2d(
                          owner.host_device.get(), binding.data,
                          iree_hal_buffer_view_buffer(view.get()), 0,
                          binding.element_count * sizeof(float),
                          IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()),
                      "update pointwise input");
                owner.counters.upload_bytes += binding.element_count * sizeof(float);
            }
            PushBuffer(clear.inputs, view.get());
        }
        return state_->Invoke();
    } catch (...) {
        owner.poisoned = true;
        throw;
    }
}

PreparedFFNWeights::PreparedFFNWeights(const DeviceContext &device,
                                      const contracts::FFNCompileSpec &spec,
                                      TensorBinding gate, TensorBinding up, TensorBinding down) {
    contracts::ValidateFFNSpec(spec);
    // Reject the complete boundary before allocating any device snapshots.
    const auto &shape = spec.shape;
    gate.Validate(shape.f, shape.d);
    up.Validate(shape.f, shape.d);
    down.Validate(shape.d, shape.f);
    if (spec.weight_layout == contracts::FFNWeightLayout::Transposed) {
        std::array<TensorBinding, 3> source{gate, up, down};
        std::array<std::vector<float>, 3> transposed;
        for (std::size_t role = 0; role < source.size(); ++role) {
            const auto &input = source[role];
            auto &output = transposed[role];
            output.resize(input.element_count);
            for (std::size_t row = 0; row < input.rows; ++row) {
                for (std::size_t col = 0; col < input.cols; ++col) {
                    output[col * input.rows + row] = input.data[row * input.cols + col];
                }
            }
        }
        state_ = std::make_unique<detail::FFNWeightState>(
            device, TensorBinding::Dense(transposed[0], shape.d, shape.f),
            TensorBinding::Dense(transposed[1], shape.d, shape.f),
            TensorBinding::Dense(transposed[2], shape.f, shape.d));
    } else {
        state_ = std::make_unique<detail::FFNWeightState>(device, gate, up, down);
    }
    state_->layout = spec.weight_layout;
}
PreparedFFNWeights::~PreparedFFNWeights() = default;
PreparedFFNWeights::PreparedFFNWeights(PreparedFFNWeights &&) noexcept = default;
PreparedFFNWeights &PreparedFFNWeights::operator=(PreparedFFNWeights &&) noexcept = default;

FFNModule::FFNModule(const DeviceContext &device, const std::string &directory,
                     const contracts::FFNCompileSpec &spec)
    : state_(std::make_unique<detail::ModuleState>()) {
    contracts::ValidateFFNSpec(spec);
    state_->device = device.state_;
    state_->device->RequireHealthy();
    auto artifact = artifacts::LoadFFNArtifact(directory, spec);
    state_->ffn = spec;
    state_->program_sha256 = artifact.program_sha256;
    state_->output_rows = spec.shape.m;
    state_->output_cols = spec.shape.d;
    // Initialize compares actual VM function reflection and the four-reference
    // calling convention before creating a VM context or running initializers.
    state_->Initialize("", contracts::kFFNEntryPoint, contracts::kFFNReflectionKey,
                       artifact.reflection, 4, std::move(artifact.bytecode));
}
FFNModule::~FFNModule() = default;
std::string FFNModule::program_sha256() const {
    if (!state_) throw std::invalid_argument("cannot inspect a moved-from FFN module");
    return state_->program_sha256;
}
FFNModule::FFNModule(FFNModule &&) noexcept = default;
FFNModule &FFNModule::operator=(FFNModule &&) noexcept = default;

std::vector<float> FFNModule::Run(TensorBinding activation, const PreparedFFNWeights &weights) {
    if (!state_ || !weights.state_) {
        throw std::invalid_argument("cannot use a moved-from FFN module or weights");
    }
    auto &owner = *state_->device;
    owner.RequireHealthy();
    const auto &shape = state_->ffn.shape;
    activation.Validate(shape.m, shape.d);
    const auto &gate = *weights.state_->gate.state_;
    const auto &up = *weights.state_->up.state_;
    const auto &down = *weights.state_->down.state_;
    const bool transposed = state_->ffn.weight_layout == contracts::FFNWeightLayout::Transposed;
    if (weights.state_->layout != state_->ffn.weight_layout ||
        gate.device != state_->device || up.device != state_->device ||
        down.device != state_->device ||
        gate.rows != static_cast<std::size_t>(transposed ? shape.d : shape.f) ||
        gate.cols != static_cast<std::size_t>(transposed ? shape.f : shape.d) ||
        up.rows != gate.rows || up.cols != gate.cols ||
        down.rows != gate.cols || down.cols != gate.rows) {
        throw std::invalid_argument("FFN weights belong to another context or shape");
    }

    ClearCallLists clear{state_->inputs.get(), state_->outputs.get()};
    iree_vm_list_clear(clear.inputs);
    iree_vm_list_clear(clear.outputs);
    ++owner.counters.calls_attempted;
    try {
        state_->UpdateActivation(activation);
        PushBuffer(clear.inputs, state_->activation.get());
        PushBuffer(clear.inputs, gate.view.get());
        PushBuffer(clear.inputs, up.view.get());
        PushBuffer(clear.inputs, down.view.get());
        return state_->Invoke();
    } catch (...) {
        // An attempted dispatch may have changed device state. Do not retry
        // this context or expose a partially computed output.
        owner.poisoned = true;
        throw;
    }
}


std::vector<float> FFNModule::RunRows(TensorBinding activation,
                                    const PreparedFFNWeights &weights) {
    if (!state_ || !weights.state_)
        throw std::invalid_argument("cannot use a moved-from FFN module or weights");
    const auto &shape = state_->ffn.shape;
    if (!activation.rows || activation.rows > static_cast<std::size_t>(shape.m))
        throw std::invalid_argument("logical FFN rows exceed compiled capacity");
    activation.Validate(activation.rows, shape.d);
    if (activation.rows == static_cast<std::size_t>(shape.m)) return Run(activation, weights);
    const auto prepare_start = Clock::now();
    // A fresh zeroed tail is required for every invocation, including decreasing
    // M on a reused graph. Pooling this staging storage is a separate M6 task.
    std::vector<float> padded(static_cast<std::size_t>(shape.m * shape.d), 0.0f);
    std::memcpy(padded.data(), activation.data, activation.element_count * sizeof(float));
    const auto prepare_end = Clock::now();
    auto output = Run(TensorBinding::Dense(padded, shape.m, shape.d), weights);
    const auto crop_start = Clock::now();
    output.resize(activation.element_count);
    auto &stats = state_->device->counters;
    stats.ffn_padded_rows += shape.m - activation.rows;
    stats.ffn_cropped_rows += shape.m - activation.rows;
    stats.ffn_boundary_ms +=
        std::chrono::duration<double, std::milli>(prepare_end - prepare_start).count() +
        std::chrono::duration<double, std::milli>(Clock::now() - crop_start).count();
    return output;
}

} // namespace coral::runtime
