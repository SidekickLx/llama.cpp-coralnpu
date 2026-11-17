// SPDX-License-Identifier: MIT
#pragma once

#include "ggml-backend.h"
#include "runtime/matmul_runtime.h"
#include <cstdint>
#include <string>

namespace coral::ggml_backend {

struct NodeEvent {
    const ggml_tensor *node;
    runtime::MatmulShape shape;
    std::uint64_t dispatches;
    std::uint64_t instructions;
    bool weight_cache_hit;
};

struct Options {
    std::string artifact_directory;
    // Opt-in promise: WEIGHTS leaf tensors remain immutable and their storage
    // stays alive until every backend using this registration has been freed.
    bool immutable_model_weights = false;
    void (*trace)(const NodeEvent &, void *) = nullptr;
    void *trace_user_data = nullptr;
};

struct Stats {
    runtime::RuntimeStats runtime;
    std::uint64_t graph_calls = 0;
    std::uint64_t matmuls = 0;
    std::uint64_t weight_cache_hits = 0;
    std::uint64_t weight_preparations = 0;
    std::uint64_t failures = 0;
};

// Pure structural check; no file access, device creation or compilation.
// Only default/F32 precision and no hint are accepted by this first version.
bool Compatible(const ggml_tensor *op, runtime::MatmulShape *shape = nullptr) noexcept;

// Register once, before model/context creation. Configuration and execution
// are confined to one host thread. All GGML C callbacks contain exceptions.
// Reads profiles.tsv eagerly; supports_op only queries its in-memory index.
void Register(const Options &options);
Stats GetStats();
const std::string &LastError();

} // namespace coral::ggml_backend
