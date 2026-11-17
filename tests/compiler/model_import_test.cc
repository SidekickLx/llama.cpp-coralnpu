// SPDX-License-Identifier: MIT
#include "tests/compiler/ggml_fixture.h"
#include "planner/ggml_ffn_pattern.h"
#include "support/json.h"
#include <array>
#include <cstring>
#include "ggml-backend.h"
#include "llama.h"
#include <dlfcn.h>
#include <algorithm>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace fs = std::filesystem;
void Require(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct Snapshot {
    ggml_tensor node{};
    std::vector<ggml_tensor> inputs;
    std::vector<std::vector<float>> input_values;
    std::vector<float> output_values;

    void Before(const ggml_tensor *source) {
        node = *source;
        inputs.clear();
        input_values.clear();
        inputs.reserve(GGML_MAX_SRC);
        input_values.reserve(GGML_MAX_SRC);
        std::map<const ggml_tensor *, std::size_t> positions;
        for (int slot = 0; slot < GGML_MAX_SRC; ++slot) {
            auto *input = source->src[slot];
            if (!input) {
                continue;
            }
            auto found = positions.find(input);
            if (found == positions.end()) {
                Require(input->type == GGML_TYPE_F32 && ggml_is_contiguous(input),
                        "model probe input must be dense F32");
                const auto index = inputs.size();
                inputs.push_back(*input);
                input_values.emplace_back(ggml_nelements(input));
                ggml_backend_tensor_get(input, input_values.back().data(), 0, ggml_nbytes(input));
                inputs.back().data = input_values.back().data();
                found = positions.emplace(input, index).first;
            }
            node.src[slot] = &inputs[found->second];
        }
    }

    void After(const ggml_tensor *source) {
        Require(source->type == GGML_TYPE_F32 && ggml_is_contiguous(source),
                "model probe output must be dense F32");
        output_values.resize(ggml_nelements(source));
        ggml_backend_tensor_get(source, output_values.data(), 0, ggml_nbytes(source));
        node.data = output_values.data();
    }
};

// Each node owns its pre-execution inputs and post-execution result. The
// allocator may recycle live intermediates before the full FFN completes.
struct CompletedNode {
    Snapshot snapshot;
    std::array<const ggml_tensor *, GGML_MAX_SRC> sources{};
    std::size_t order = 0;
};
struct FFNCapture {
    CoralCheckGGMLImport check;
    fs::path output;
    std::map<const ggml_tensor *, std::unique_ptr<CompletedNode>> completed;
    std::unique_ptr<CompletedNode> pending;
    const ggml_tensor *pending_source = nullptr;
    int step = 0, regions = 0;

    FFNCapture(CoralCheckGGMLImport callback, const fs::path &directory)
        : check(callback), output(directory) {}

    static bool Dense(const ggml_tensor *tensor) {
        return tensor && tensor->type == GGML_TYPE_F32 && tensor->ne[0] > 0 &&
               tensor->ne[1] > 0 && tensor->ne[2] == 1 && tensor->ne[3] == 1 &&
               ggml_is_contiguous(tensor);
    }
    static bool Selected(const ggml_tensor *node) {
        if (!Dense(node) ||
            (node->op != GGML_OP_MUL_MAT && node->op != GGML_OP_GLU &&
             node->op != GGML_OP_UNARY && node->op != GGML_OP_MUL)) {
            return false;
        }
        for (auto *source : node->src) {
            if (source && !Dense(source)) {
                return false;
            }
        }
        return true;
    }
    void Begin(int index) {
        Require(!pending, "unfinished node snapshot");
        completed.clear();
        step = index;
        regions = 0;
    }
    void Before(const ggml_tensor *node) {
        Require(!pending, "callbacks did not provide a per-node boundary");
        if (!Selected(node)) {
            return;
        }
        pending = std::make_unique<CompletedNode>();
        pending->snapshot.Before(node);
        std::copy(std::begin(node->src), std::end(node->src), pending->sources.begin());
        pending->order = completed.size();
        pending_source = node;
    }
    const CompletedNode &Get(const ggml_tensor *node) const {
        const auto found = completed.find(node);
        Require(found != completed.end(), "FFN producer was not captured in this step");
        return *found->second;
    }
    void Import(const coral::planner::GGMLFFNMatch &match) {
        std::map<std::size_t, const ggml_tensor *> ordered;
        for (auto *node : {match.gate, match.up, match.silu, match.hidden, match.down}) {
            if (node) {
                Require(ordered.emplace(Get(node).order, node).second, "repeated FFN member");
            }
        }
        // Boundary payloads come from snapshots at the original consuming edges.
        const ggml_tensor *identities[] = {
            match.x, match.gate->src[0], match.up->src[0], match.down->src[0]};
        const ggml_tensor *values[] = {
            Get(match.gate).snapshot.node.src[1], Get(match.gate).snapshot.node.src[0],
            Get(match.up).snapshot.node.src[0], Get(match.down).snapshot.node.src[0]};
        const auto *up_x = Get(match.up).snapshot.node.src[1];
        Require(ggml_nbytes(up_x) == ggml_nbytes(values[0]) &&
                std::memcmp(up_x->data, values[0]->data, ggml_nbytes(up_x)) == 0,
                "gate and up consumed different activation values");
        std::vector<std::unique_ptr<ggml_tensor>> storage;
        std::map<const ggml_tensor *, ggml_tensor *> mapping;
        std::vector<const ggml_tensor *> inputs, operations;
        for (std::size_t role = 0; role < 4; ++role) {
            auto found = mapping.find(identities[role]);
            if (found == mapping.end()) {
                auto tensor = std::make_unique<ggml_tensor>(*values[role]);
                tensor->buffer = nullptr;
                found = mapping.emplace(identities[role], tensor.get()).first;
                storage.push_back(std::move(tensor));
            } else {
                Require(ggml_nbytes(found->second) == ggml_nbytes(values[role]) &&
                        std::memcmp(found->second->data, values[role]->data,
                                    ggml_nbytes(values[role])) == 0,
                        "shared input changed between consuming edges");
            }
            inputs.push_back(found->second);
        }
        for (const auto &[order, original] : ordered) {
            (void)order;
            const auto &capture = Get(original);
            auto tensor = std::make_unique<ggml_tensor>(capture.snapshot.node);
            tensor->buffer = nullptr;
            for (int slot = 0; slot < GGML_MAX_SRC; ++slot) {
                if (!capture.sources[slot]) {
                    tensor->src[slot] = nullptr;
                    continue;
                }
                const auto source = mapping.find(capture.sources[slot]);
                Require(source != mapping.end(), "incomplete or unordered FFN boundary");
                tensor->src[slot] = source->second;
            }
            Require(mapping.emplace(original, tensor.get()).second, "duplicated FFN snapshot");
            operations.push_back(tensor.get());
            storage.push_back(std::move(tensor));
        }
        const ggml_tensor *result = mapping.at(match.down);
        const std::string label = "step-" + std::to_string(step) + "-ffn-" + std::to_string(regions);
        const auto directory = output / label;
        Require(fs::create_directory(directory), "region directory already exists");
        const CoralGGMLFixture fixture{
            1, label.c_str(), inputs.data(), inputs.size(),
            operations.data(), operations.size(), &result, 1};
        // The interpreter receives only the four boundary values. Captured
        // intermediates provide metadata, never intermediate evaluation inputs.
        Require(check(&fixture, directory.c_str()) == 0,
                "full real FFN import/reference comparison failed");
        std::ofstream origin(directory / "capture.json");
        origin << "{\"step\":" << step << ",\"region\":" << regions
               << ",\"anchor\":" << coral::JsonString(ggml_get_name(match.down))
               << ",\"members\":[";
        for (std::size_t i = 0; i < operations.size(); ++i) {
            if (i) {
                origin << ',';
            }
            origin << coral::JsonString(ggml_get_name(operations[i]));
        }
        origin << "],\"input_capture\":\"before-consumer\",\"output_capture\":\"after-producer\"}\n";
        origin.close();
        Require(!origin.fail(), "cannot save region capture metadata");
        ++regions;
        std::cout << "Verified " << label << '\n' << std::flush;
    }
    void After(const ggml_tensor *node) {
        if (pending_source != node) {
            return;
        }
        pending->snapshot.After(node);
        Require(completed.emplace(node, std::move(pending)).second, "node executed twice in one step");
        pending_source = nullptr;
        const auto match = coral::planner::MatchFFNPattern(node);
        if (match) {
            Import(*match);
        }
    }
};

struct Observer {
    CoralCheckGGMLImport check;
    fs::path output;
    std::exception_ptr error;
    Snapshot pending;
    const ggml_tensor *pending_source = nullptr;
    bool matmul = false, glu = false;
    int step = 0;
    int captures = 0;
    int expected_layers = 0;
    std::unique_ptr<FFNCapture> regions;

    Observer(CoralCheckGGMLImport callback, const fs::path &directory, bool full_regions)
        : check(callback), output(directory) {
        if (full_regions) {
            regions = std::make_unique<FFNCapture>(callback, directory);
        }
    }

    bool Selected(const ggml_tensor *node) const {
        return (node->op == GGML_OP_MUL_MAT && !matmul) ||
               (node->op == GGML_OP_GLU && !glu);
    }

    static bool Callback(ggml_tensor *node, bool ask, void *user) noexcept {
        auto &self = *static_cast<Observer *>(user);
        if (self.error) {
            return ask; // Request a boundary, then stop at its completed callback.
        }
        try {
            if (self.regions) {
                if (ask) {
                    self.regions->Before(node);
                } else {
                    self.regions->After(node);
                }
                return true;
            }
            if (ask) {
                if (self.Selected(node)) {
                    self.pending.Before(node);
                    self.pending_source = node;
                }
                // Ask for EVERY node. Otherwise GGML may ask about a later
                // node before executing its producers, making input capture
                // premature. Each completed callback is a synchronization point.
                return true;
            }
            if (self.pending_source == node) {
                self.pending.After(node);
                std::vector<const ggml_tensor *> inputs;
                for (auto &input : self.pending.inputs) {
                    inputs.push_back(&input);
                }
                const ggml_tensor *operation = &self.pending.node;
                const std::string label = "step-" + std::to_string(self.step) +
                    (node->op == GGML_OP_GLU ? "-glu" : "-matmul");
                const auto directory = self.output / label;
                Require(fs::create_directory(directory), "capture directory already exists");
                const CoralGGMLFixture fixture{
                    1, label.c_str(), inputs.data(), inputs.size(),
                    &operation, 1, &operation, 1};
                Require(self.check(&fixture, directory.c_str()) == 0,
                        "real model node import/reference comparison failed");
                if (node->op == GGML_OP_GLU) {
                    self.glu = true;
                } else {
                    self.matmul = true;
                }
                ++self.captures;
                self.pending_source = nullptr;
            }
            return true;
        } catch (...) {
            self.error = std::current_exception();
            return false;
        }
    }

    void Begin(int index) {
        step = index;
        if (regions) {
            regions->Begin(index);
        }
        matmul = glu = false;
        pending_source = nullptr;
    }

    void Finish() {
        if (error) {
            std::rethrow_exception(error);
        }
        if (regions) {
            Require(regions->regions == expected_layers, "not every real model FFN was captured");
            captures += regions->regions;
        } else {
            Require(matmul && glu, "expected one real matmul and one real split GLU per step");
        }
    }
};

struct BackendLifetime {
    BackendLifetime() { llama_backend_init(); }
    ~BackendLifetime() { llama_backend_free(); }
};

struct Batch {
    llama_batch value = llama_batch_init(32, 0, 1);
    ~Batch() { llama_batch_free(value); }
};
} // namespace

int main(int argc, char **argv) {
    try {
        Require(argc == 4 || argc == 5,
                "Usage: coral-model-import-test MODEL CHECK_LIBRARY NEW_OUTPUT_DIR [--ffn-regions]");
        const bool full_regions = argc == 5;
        Require(!full_regions || std::string(argv[4]) == "--ffn-regions", "unknown mode");
        fs::path output(argv[3]);
        if (!output.parent_path().empty()) {
            fs::create_directories(output.parent_path());
        }
        Require(fs::create_directory(output), "output directory must be new");
        // Keep the test library loaded until process exit: MLIR/LLVM can
        // retain process-wide registrations owned by the shared library.
        void *library = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
        Require(library != nullptr, "cannot load compiler test library");
        const char *entry = full_regions ? "coral_check_ggml_ffn_import" : "coral_check_ggml_import";
        auto check = reinterpret_cast<CoralCheckGGMLImport>(dlsym(library, entry));
        Require(check != nullptr, "missing compiler test entry point");
        Observer observer{check, output, full_regions};
        BackendLifetime lifetime;
        Require(ggml_backend_dev_count() == 1 &&
                ggml_backend_dev_type(ggml_backend_dev_get(0)) == GGML_BACKEND_DEVICE_TYPE_CPU,
                "model test requires exactly one CPU backend");
        auto model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0;
        model_params.use_extra_bufts = false;
        model_params.load_mode = LLAMA_LOAD_MODE_MMAP;
        model_params.lazy_mode = LLAMA_LAZY_MODE_OFF;
        model_params.check_tensors = true;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[1], model_params), llama_model_free);
        Require(model != nullptr, "failed to load model");
        observer.expected_layers = llama_model_n_layer(model.get());
        Require(observer.expected_layers > 0, "model has no layers");
        auto parameters = llama_context_default_params();
        parameters.n_ctx = 128;
        parameters.n_batch = parameters.n_ubatch = 32;
        parameters.n_threads = parameters.n_threads_batch = 1;
        parameters.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        parameters.type_k = parameters.type_v = GGML_TYPE_F32;
        parameters.offload_kqv = parameters.op_offload = false;
        parameters.cb_eval = Observer::Callback;
        parameters.cb_eval_user_data = &observer;
        std::unique_ptr<llama_context, decltype(&llama_free)> context(
            llama_init_from_model(model.get(), parameters), llama_free);
        Require(context != nullptr, "failed to create model context");
        const auto *vocab = llama_model_get_vocab(model.get());
        const std::string prompt = "Once upon a time";
        std::vector<llama_token> tokens(32);
        const int count = llama_tokenize(vocab, prompt.data(), prompt.size(),
                                        tokens.data(), tokens.size(), true, false);
        Require(count > 0 && count <= 32, "unexpected prompt token count");
        tokens.resize(count);
        Batch batch;
        int position = 0;
        const int steps = full_regions ? 4 : 2;
        for (int step = 0; step < steps; ++step) {
            observer.Begin(step);
            batch.value.n_tokens = tokens.size();
            for (std::size_t i = 0; i < tokens.size(); ++i) {
                batch.value.token[i] = tokens[i];
                batch.value.pos[i] = position + i;
                batch.value.n_seq_id[i] = 1;
                batch.value.seq_id[i][0] = 0;
                batch.value.logits[i] = i + 1 == tokens.size();
            }
            const int status = llama_decode(context.get(), batch.value);
            observer.Finish();
            Require(status == 0, "model decode failed");
            const auto *logits = llama_get_logits_ith(context.get(), -1);
            Require(logits != nullptr, "missing model logits");
            const int vocab_size = llama_vocab_n_tokens(vocab);
            const llama_token next = std::max_element(logits, logits + vocab_size) - logits;
            position += tokens.size();
            tokens = {next};
        }
        std::ofstream summary(output / "summary.json");
        summary << "{\"status\":\"passed\",\"steps\":" << steps
                << ",\"layers\":" << observer.expected_layers
                << ",\"mode\":" << coral::JsonString(full_regions ? "ffn-regions" : "nodes")
                << ",\"captures\":" << observer.captures
                << ",\"input_capture\":\"before-node\",\"output_capture\":\"after-node\"}\n";
        summary.close();
        Require(!summary.fail(), "cannot write model test summary");
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "coral-model-import-test: " << error.what() << '\n';
        return 1;
    }
}
