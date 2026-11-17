// SPDX-License-Identifier: MIT
#include "ggml_coral.h"
#include "support/json.h"
#include "llama.h"
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {
namespace fs = std::filesystem;
namespace adapter = coral::ggml_backend;
using coral::JsonString;
using coral::runtime::MatmulShape;

constexpr int kMaxPromptTokens = 32;
constexpr int kContextTokens = 128;

struct Arguments {
    std::string mode;
    fs::path model;
    fs::path output;
    fs::path artifacts;
    fs::path tokens;
    std::string prompt = "Once upon a time";
    int decode = 3;
};

void Usage() {
    std::cout
        << "coral-llama-runner --mode cpu|coral --model FILE --output NEW_DIRECTORY\n"
        << "  [--prompt TEXT --decode 2|3|4] | [--tokens FILE]\n"
        << "  --artifacts DIRECTORY   Required only in Coral mode (AOT profiles.tsv + VMFBs).\n"
        << "A replay file supplies both prompt and decode input tokens. No compiler is invoked.\n";
}

Arguments Parse(int argc, char **argv) {
    Arguments args;
    std::set<std::string> seen;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (!seen.insert(key).second) {
            throw std::invalid_argument("duplicate option: " + key);
        }
        if (i + 1 >= argc) {
            throw std::invalid_argument("duplicate option or missing option value: " + key);
        }
        const std::string value = argv[++i];
        if (value.empty()) {
            throw std::invalid_argument("empty option value: " + key);
        }
        if (key == "--mode") {
            args.mode = value;
        } else if (key == "--model") {
            args.model = value;
        } else if (key == "--output") {
            args.output = value;
        } else if (key == "--artifacts") {
            args.artifacts = value;
        } else if (key == "--tokens") {
            args.tokens = value;
        } else if (key == "--prompt") {
            args.prompt = value;
        } else if (key == "--decode") {
            const auto result = std::from_chars(value.data(), value.data() + value.size(), args.decode);
            if (result.ec != std::errc{} || result.ptr != value.data() + value.size()) {
                throw std::invalid_argument("--decode must be 2, 3 or 4");
            }
        } else {
            throw std::invalid_argument("unknown option: " + key);
        }
    }
    if (args.mode != "cpu" && args.mode != "coral") {
        throw std::invalid_argument("--mode must be cpu or coral");
    }
    if (!fs::is_regular_file(args.model) || args.output.empty()) {
        throw std::invalid_argument("an existing --model and a new --output directory are required");
    }
    if ((args.mode == "coral") != !args.artifacts.empty()) {
        throw std::invalid_argument("--artifacts is required in Coral mode and forbidden in CPU mode");
    }
    if (!args.tokens.empty() && (seen.count("--prompt") || seen.count("--decode"))) {
        throw std::invalid_argument("--tokens cannot be combined with --prompt or --decode");
    }
    if (args.decode < 2 || args.decode > 4 || args.prompt.size() > 4096) {
        throw std::invalid_argument("use 2-4 decode steps and a prompt of at most 4096 bytes");
    }
    return args;
}

std::ofstream OpenOutput(const fs::path &path, bool binary = false) {
    std::ofstream stream;
    stream.exceptions(std::ios::badbit | std::ios::failbit);
    stream.open(path, std::ios::out | (binary ? std::ios::binary : std::ios::openmode{}));
    return stream;
}

struct TokenInputs {
    int prefill = 0;
    int decode = 0;
    std::vector<llama_token> values;
};

TokenInputs ReadTokens(const fs::path &path) {
    std::ifstream input(path);
    std::string header;
    TokenInputs tokens;
    if (!std::getline(input, header) || header != "coral-m2-tokens-v1" ||
        !(input >> tokens.prefill >> tokens.decode) ||
        tokens.prefill < 1 || tokens.prefill > kMaxPromptTokens ||
        tokens.decode < 2 || tokens.decode > 4) {
        throw std::invalid_argument("invalid replay header or token counts");
    }
    tokens.values.resize(tokens.prefill + tokens.decode);
    for (auto &token : tokens.values) {
        if (!(input >> token) || token < 0) {
            throw std::invalid_argument("invalid or missing replay token");
        }
    }
    input >> std::ws;
    if (!input.eof() || input.bad()) {
        throw std::invalid_argument("extra data or read error in replay file");
    }
    return tokens;
}

TokenInputs Tokenize(const llama_vocab *vocab, const Arguments &args) {
    TokenInputs tokens;
    tokens.decode = args.decode;
    // A fixed-size prompt keeps the prefill in one microbatch and makes its M
    // dimension reproducible. Reject longer prompts instead of truncating them.
    tokens.values.resize(kMaxPromptTokens);
    const int count = llama_tokenize(vocab, args.prompt.data(), args.prompt.size(),
                                    tokens.values.data(), tokens.values.size(), true, false);
    if (count <= 0 || count > kMaxPromptTokens) {
        throw std::invalid_argument("prompt must tokenize to 1-32 tokens, including special tokens");
    }
    tokens.prefill = count;
    tokens.values.resize(count);
    return tokens;
}

void WriteTokens(const fs::path &path, const TokenInputs &tokens) {
    auto output = OpenOutput(path);
    output << "coral-m2-tokens-v1\n" << tokens.prefill << ' ' << tokens.decode << '\n';
    for (auto token : tokens.values) {
        output << token << '\n';
    }
    output.close();
}

void WriteStats(std::ostream &out, const adapter::Stats &after, const adapter::Stats &before = {}) {
    const auto &a = after.runtime;
    const auto &b = before.runtime;
    out << "{\"graph_calls\":" << after.graph_calls - before.graph_calls
        << ",\"graph_optimizations\":" << after.graph_optimizations - before.graph_optimizations
        << ",\"ffn_candidates\":" << after.ffn_candidates - before.ffn_candidates
        << ",\"ffn_preparation_plans\":" << after.ffn_preparation_plans - before.ffn_preparation_plans
        << ",\"ffn_rejections\":" << after.ffn_rejections - before.ffn_rejections
        << ",\"matmuls\":" << after.matmuls - before.matmuls
        << ",\"weight_preparations\":" << after.weight_preparations - before.weight_preparations
        << ",\"weight_cache_hits\":" << after.weight_cache_hits - before.weight_cache_hits
        << ",\"failures\":" << after.failures - before.failures
        << ",\"module_loads\":" << a.module_loads - b.module_loads
        << ",\"calls_completed\":" << a.calls_completed - b.calls_completed
        << ",\"coral_dispatches\":" << a.coral_dispatches - b.coral_dispatches
        << ",\"instruction_samples\":"
        << a.dispatches_with_instruction_count - b.dispatches_with_instruction_count
        << ",\"kernel_instructions\":" << a.kernel_instructions - b.kernel_instructions
        << ",\"host_executable_load_attempts\":"
        << a.host_executable_load_attempts - b.host_executable_load_attempts
        << ",\"failed_dispatches\":" << a.failed_dispatches - b.failed_dispatches
        << ",\"observer_errors\":" << a.observer_errors - b.observer_errors
        << ",\"binding_allocations\":" << a.binding_allocations - b.binding_allocations
        << ",\"logical_upload_bytes\":" << a.upload_bytes - b.upload_bytes
        << ",\"logical_download_bytes\":" << a.download_bytes - b.download_bytes
        << ",\"module_load_ms\":" << a.module_load_ms - b.module_load_ms
        << ",\"invoke_ms\":" << a.invoke_ms - b.invoke_ms << '}';
}

bool Numerical(const ggml_tensor *node) {
    return (node->flags & GGML_TENSOR_FLAG_COMPUTE) && node->op != GGML_OP_NONE &&
           node->op != GGML_OP_VIEW && node->op != GGML_OP_RESHAPE &&
           node->op != GGML_OP_TRANSPOSE && node->op != GGML_OP_PERMUTE;
}

struct Trace {
    std::ofstream output;
    std::set<std::tuple<std::size_t, std::size_t, std::size_t>> profiles;
    std::map<const ggml_tensor *, adapter::NodeEvent> pending;
    std::map<ggml_backend_buffer_t, std::size_t> buffers;
    std::exception_ptr callback_error;
    std::uint64_t cpu_nodes = 0;
    std::uint64_t coral_nodes = 0;
    std::uint64_t cpu_matmuls = 0;
    int step = -1;

    explicit Trace(const fs::path &path) : output(OpenOutput(path)) {}

    const char *Phase() const { return step == 0 ? "prefill" : "decode"; }

    void Begin(int index) {
        if (!pending.empty()) {
            throw std::runtime_error("unmatched Coral dispatch trace");
        }
        step = index;
        cpu_nodes = coral_nodes = cpu_matmuls = 0;
    }

    std::size_t BufferId(const ggml_tensor *tensor) {
        auto buffer = tensor->buffer;
        if (!buffer && tensor->view_src) {
            buffer = tensor->view_src->buffer;
        }
        if (!buffer) {
            return 0;
        }
        const auto existing = buffers.find(buffer);
        if (existing != buffers.end()) {
            return existing->second;
        }
        const std::size_t id = buffers.size() + 1;
        buffers.emplace(buffer, id);
        output << "{\"kind\":\"buffer\",\"id\":" << id
               << ",\"bytes\":" << ggml_backend_buffer_get_size(buffer)
               << ",\"type\":" << JsonString(ggml_backend_buft_name(ggml_backend_buffer_get_type(buffer)))
               << ",\"host\":" << (ggml_backend_buffer_is_host(buffer) ? "true" : "false")
               << ",\"usage\":" << int(ggml_backend_buffer_get_usage(buffer)) << "}\n";
        return id;
    }

    void Tensor(const ggml_tensor *tensor, std::size_t buffer_id) {
        if (!tensor) {
            output << "null";
            return;
        }
        output << "{\"dtype\":" << JsonString(ggml_type_name(tensor->type))
               << ",\"ne\":[";
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            if (i) output << ',';
            output << tensor->ne[i];
        }
        output << "],\"nb\":[";
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            if (i) output << ',';
            output << tensor->nb[i];
        }
        output << "],\"buffer_id\":" << buffer_id << '}';
    }

    void Observe(ggml_tensor *node) {
        MatmulShape shape;
        const bool compatible = adapter::Compatible(node, &shape);
        if (compatible) {
            profiles.emplace(shape.m, shape.k, shape.n);
        }
        const auto event = pending.find(node);
        const bool coral = event != pending.end();
        if (coral) {
            ++coral_nodes;
        } else {
            ++cpu_nodes;
            if (node->op == GGML_OP_MUL_MAT) ++cpu_matmuls;
        }
        // Register buffers before beginning the node record: each buffer is its
        // own JSON line. IDs are local observations, not allocator call counts.
        const auto dst = BufferId(node);
        const auto w = node->src[0] ? BufferId(node->src[0]) : 0;
        const auto x = node->src[1] ? BufferId(node->src[1]) : 0;
        output << "{\"kind\":\"node\",\"step\":" << step
               << ",\"phase\":" << JsonString(Phase())
               << ",\"name\":" << JsonString(ggml_get_name(node))
               << ",\"op\":" << JsonString(ggml_op_name(node->op))
               << ",\"placement\":" << JsonString(coral ? "CORAL" : "CPU")
               << ",\"coral_compatible\":" << (compatible ? "true" : "false")
               << ",\"output\":";
        Tensor(node, dst);
        output << ",\"src0\":";
        Tensor(node->src[0], w);
        output << ",\"src1\":";
        Tensor(node->src[1], x);
        if (compatible) {
            output << ",\"m\":" << shape.m << ",\"k\":" << shape.k << ",\"n\":" << shape.n;
        }
        if (node->op == GGML_OP_MUL_MAT) {
            output << ",\"op_params\":[";
            for (std::size_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(std::int32_t); ++i) {
                if (i) output << ',';
                output << node->op_params[i];
            }
            output << ']';
        }
        if (coral) {
            output << ",\"dispatches\":" << event->second.dispatches
                   << ",\"instructions\":" << event->second.instructions
                   << ",\"weight_cache_hit\":" << (event->second.weight_cache_hit ? "true" : "false");
            pending.erase(event);
        }
        output << "}\n";
    }

    static void OnCoral(const adapter::NodeEvent &event, void *user) {
        auto &trace = *static_cast<Trace *>(user);
        if (trace.step < 0 || !event.dispatches || !event.instructions ||
            !trace.pending.emplace(event.node, event).second) {
            throw std::runtime_error("invalid or duplicate Coral node event");
        }
    }

    static bool OnEval(ggml_tensor *node, bool ask, void *user) noexcept {
        auto &trace = *static_cast<Trace *>(user);
        // Observe every numerical node after execution. This diagnostic mode
        // intentionally limits scheduler batching; it is not a throughput test.
        if (ask) {
            return Numerical(node);
        }
        if (trace.callback_error) {
            return false;
        }
        try {
            trace.Observe(node);
            return true;
        } catch (...) {
            trace.callback_error = std::current_exception();
            return false;
        }
    }

    void Check() {
        // The pinned scheduler can return success after a callback requests an
        // abort. Explicitly propagate observer errors before publishing logits.
        if (callback_error) {
            std::rethrow_exception(callback_error);
        }
        if (!pending.empty()) {
            throw std::runtime_error("Coral dispatch missing a completed GGML node observation");
        }
        output.flush();
    }

    void WriteProfiles(const fs::path &path) const {
        auto file = OpenOutput(path);
        file << "coral-m2-v1\n";
        for (const auto &[m, k, n] : profiles) {
            file << m << '\t' << k << '\t' << n << "\tm" << m << "_k" << k << "_n" << n << ".vmfb\n";
        }
        file.close();
    }
};

void CheckBackends(bool coral) {
    // With static CPU registration, llama_backend_init will not call the plugin
    // loader. Reject any unexpected backend rather than mislabel its work CPU.
    const auto count = ggml_backend_dev_count();
    if (count != (coral ? 2u : 1u)) {
        throw std::runtime_error("runner requires exactly CPU, plus CORAL only in Coral mode");
    }
    for (std::size_t i = 0; i < count; ++i) {
        const auto device = ggml_backend_dev_get(i);
        const std::string name = ggml_backend_dev_name(device);
        const auto type = ggml_backend_dev_type(device);
        if (name == "CPU" && type == GGML_BACKEND_DEVICE_TYPE_CPU) {
            continue;
        }
        if (coral && name == "CORAL" && type == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            continue;
        }
        throw std::runtime_error("unexpected registered backend: " + name);
    }
}

struct BackendLifetime {
    BackendLifetime() { llama_backend_init(); }
    ~BackendLifetime() { llama_backend_free(); }
};

struct Batch {
    llama_batch value = llama_batch_init(kMaxPromptTokens, 0, 1);
    ~Batch() { llama_batch_free(value); }

    void Set(const std::vector<llama_token> &tokens, int position) {
        value.n_tokens = static_cast<int32_t>(tokens.size());
        for (int i = 0; i < value.n_tokens; ++i) {
            value.token[i] = tokens[i];
            value.pos[i] = position + i;
            value.n_seq_id[i] = 1;
            value.seq_id[i][0] = 0;
            value.logits[i] = i == value.n_tokens - 1;
        }
    }
};

llama_token SaveLogits(llama_context *context, int vocab_size, std::ofstream &output) {
    const float *logits = llama_get_logits_ith(context, -1);
    if (!logits) {
        throw std::runtime_error("llama.cpp returned no logits");
    }
    for (int i = 0; i < vocab_size; ++i) {
        if (!std::isfinite(logits[i])) {
            throw std::runtime_error("non-finite model logits");
        }
    }
    output.write(reinterpret_cast<const char *>(logits), vocab_size * sizeof(float));
    return static_cast<llama_token>(std::max_element(logits, logits + vocab_size) - logits);
}

void Run(const Arguments &args) {
    TokenInputs tokens;
    const bool replay = !args.tokens.empty();
    if (replay) {
        tokens = ReadTokens(args.tokens);
    }
    if (!args.output.parent_path().empty()) {
        fs::create_directories(args.output.parent_path());
    }
    if (!fs::create_directory(args.output)) {
        throw std::invalid_argument("--output must be a new directory; previous results are preserved");
    }
    const bool coral = args.mode == "coral";
    Trace trace(args.output / "trace.jsonl");
    auto steps = OpenOutput(args.output / "steps.jsonl");
    auto logits = OpenOutput(args.output / "logits.f32", true);
    CheckBackends(false);
    BackendLifetime lifetime;
    if (coral) {
        adapter::Options options;
        options.artifact_directory = args.artifacts.string();
        // The model is read-only, and its owner below outlives the context and
        // all cached weight snapshots. No LoRA or weight mutation is permitted.
        options.immutable_model_weights = true;
        options.trace = Trace::OnCoral;
        options.trace_user_data = &trace;
        adapter::Register(options);
    }
    CheckBackends(coral);

    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    model_params.use_extra_bufts = false;
    model_params.load_mode = LLAMA_LOAD_MODE_MMAP;
    model_params.lazy_mode = LLAMA_LAZY_MODE_OFF;
    model_params.check_tensors = true;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(args.model.c_str(), model_params), llama_model_free);
    if (!model) {
        throw std::runtime_error("failed to load GGUF model");
    }
    const auto *vocab = llama_model_get_vocab(model.get());
    const int vocab_size = llama_vocab_n_tokens(vocab);
    if (vocab_size <= 0) {
        throw std::runtime_error("model vocabulary is empty");
    }
    if (!replay) {
        tokens = Tokenize(vocab, args);
    }
    for (auto token : tokens.values) {
        if (token < 0 || token >= vocab_size) {
            throw std::invalid_argument("replay token is outside this model's vocabulary");
        }
    }

    auto context_params = llama_context_default_params();
    context_params.n_ctx = kContextTokens;
    context_params.n_batch = kMaxPromptTokens;
    context_params.n_ubatch = kMaxPromptTokens;
    context_params.n_threads = 1;
    context_params.n_threads_batch = 1;
    context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    context_params.type_k = GGML_TYPE_F32;
    context_params.type_v = GGML_TYPE_F32;
    context_params.offload_kqv = false;
    context_params.op_offload = coral;
    context_params.cb_eval = Trace::OnEval;
    context_params.cb_eval_user_data = &trace;
    std::unique_ptr<llama_context, decltype(&llama_free)> context(
        llama_init_from_model(model.get(), context_params), llama_free);
    if (!context) {
        throw std::runtime_error("failed to create llama context: " + adapter::LastError());
    }

    Batch batch;
    int position = 0;
    llama_token next = 0;
    for (int step = 0; step <= tokens.decode; ++step) {
        std::vector<llama_token> inputs;
        if (step == 0) {
            inputs.assign(tokens.values.begin(), tokens.values.begin() + tokens.prefill);
        } else {
            if (!replay) {
                tokens.values.push_back(next);
            }
            inputs.push_back(tokens.values[tokens.prefill + step - 1]);
        }
        trace.Begin(step);
        batch.Set(inputs, position);
        const auto before = adapter::GetStats();
        const auto start = std::chrono::steady_clock::now();
        const int status = llama_decode(context.get(), batch.value);
        trace.Check();
        if (status != 0) {
            throw std::runtime_error("llama_decode failed (" + std::to_string(status) +
                                     "): " + adapter::LastError());
        }
        const auto after = adapter::GetStats();
        if (coral && (!trace.coral_nodes ||
                      after.runtime.coral_dispatches == before.runtime.coral_dispatches)) {
            throw std::runtime_error("Coral mode requires actual NPU dispatch in every prefill/decode step");
        }
        next = SaveLogits(context.get(), vocab_size, logits);
        const double elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        steps << "{\"step\":" << step << ",\"phase\":" << JsonString(trace.Phase())
              << ",\"input_tokens\":" << inputs.size()
              << ",\"position\":" << position << ",\"logit_row\":" << step
              << ",\"argmax_token\":" << next << ",\"cpu_nodes\":" << trace.cpu_nodes
              << ",\"cpu_matmuls\":" << trace.cpu_matmuls
              << ",\"coral_nodes\":" << trace.coral_nodes << ",\"wall_ms\":" << elapsed
              << ",\"stats\":";
        WriteStats(steps, after, before);
        steps << "}\n";
        steps.flush();
        std::cout << trace.Phase() << " step=" << step << " CPU nodes=" << trace.cpu_nodes
                  << " Coral matmuls=" << trace.coral_nodes
                  << " dispatches=" << after.runtime.coral_dispatches - before.runtime.coral_dispatches
                  << " argmax=" << next << '\n';
        position += static_cast<int>(inputs.size());
    }
    trace.WriteProfiles(args.output / "profiles.tsv");
    WriteTokens(args.output / "tokens.txt", tokens);
    logits.close();
    steps.close();
    trace.output.close();

    // The summary is the completion marker. Partial runs keep their diagnostic
    // files, but never receive a successful summary.
    auto summary = OpenOutput(args.output / "summary.pending.json");
    summary << "{\"schema\":\"coral-m2-run-v1\",\"status\":\"completed\",\"mode\":"
            << JsonString(args.mode) << ",\"coral_registered\":" << (coral ? "true" : "false")
            << ",\"registered_devices\":[\"CPU\"" << (coral ? ",\"CORAL\"" : "") << ']'
            << ",\"prefill_tokens\":" << tokens.prefill << ",\"decode_steps\":" << tokens.decode
            << ",\"vocab_size\":" << vocab_size << ",\"logit_rows\":" << tokens.decode + 1
            << ",\"logits_format\":\"little-endian-f32-row-major\""
            << ",\"input_mode\":" << JsonString(replay ? "fixed-token-replay" : "greedy")
            << ",\"model_file\":" << JsonString(args.model.filename().string())
            << ",\"observed_profiles\":" << trace.profiles.size()
            << ",\"observed_buffer_handles\":" << trace.buffers.size() << ",\"stats\":";
    WriteStats(summary, adapter::GetStats());
    summary << "}\n";
    summary.close();
    fs::rename(args.output / "summary.pending.json", args.output / "summary.json");
}
} // namespace

int main(int argc, char **argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        Usage();
        return 0;
    }
    try {
        Run(Parse(argc, argv));
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "coral-llama-runner: " << error.what() << '\n';
        return 1;
    }
}
