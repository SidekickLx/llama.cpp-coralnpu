// SPDX-License-Identifier: MIT
#pragma once

#include "ggml-backend.h"
#include "runtime/matmul_runtime.h"
#include <cstdint>
#include <string>

namespace coral::planner { struct GGMLFFNPlan; }

namespace coral::ggml_backend {

struct NodeEvent {
    const ggml_tensor *node;
    runtime::MatmulShape shape; // Valid for MUL_MAT events only.
    std::uint64_t dispatches;
    std::uint64_t instructions;
    bool weight_cache_hit;
    std::uint64_t compute_call = 0; // One received graph_compute range.
};

// One completed region invocation. Member tensors are borrowed only for this
// synchronous callback; their intermediate data was not materialized.
struct FFNEvent {
    const planner::GGMLFFNPlan &plan;
    std::string program_sha256;
    std::uint64_t dispatches;
    std::uint64_t instructions;
    bool weight_cache_hit;
    std::uint64_t compute_call;
    double invoke_ms;
};

struct Options {
    std::string artifact_directory;
    // Opt-in promise: WEIGHTS leaf tensors remain immutable and their storage
    // stays alive until every backend using this registration has been freed.
    bool immutable_model_weights = false;
    void (*trace)(const NodeEvent &, void *) = nullptr;
    void *trace_user_data = nullptr;
    // Synchronous, borrowed plan inspection inside our graph_optimize callback.
    // Reservation can invoke it too. Never retain tensor pointers from this call.
    void (*plan_trace)(const planner::GGMLFFNPlan &, void *) = nullptr;
    // The application owns scheduler callbacks. Keep the conservative default
    // unless it guarantees that per-node evaluation callbacks are absent.
    bool evaluation_callbacks = true;
    // Optional ffn.tsv entries enable complete regions. Keeping this false
    // provides the same member path for diagnostics and numerical comparison.
    bool enable_ffn = true;
    void (*ffn_trace)(const FFNEvent &, void *) = nullptr;
};

struct Stats {
    runtime::RuntimeStats runtime;
    std::uint64_t graph_calls = 0;
    std::uint64_t graph_optimizations = 0;
    std::uint64_t ffn_candidates = 0;
    std::uint64_t ffn_preparation_plans = 0;
    std::uint64_t ffn_rejections = 0;
    std::uint64_t ffn_calls = 0;
    std::uint64_t ffn_members_skipped = 0;
    std::uint64_t ffn_missing_artifacts = 0;
    std::uint64_t ffn_guard_rejections = 0;
    std::uint64_t ffn_weight_preparations = 0;
    std::uint64_t ffn_weight_cache_hits = 0;
    std::uint64_t matmuls = 0;
    std::uint64_t pointwise_calls = 0;
    std::uint64_t weight_cache_hits = 0;
    std::uint64_t weight_preparations = 0;
    std::uint64_t failures = 0;
};

// Pure structural check; no file access, device creation or compilation.
// Only default/F32 precision and no hint are accepted by this first version.
bool Compatible(const ggml_tensor *op, runtime::MatmulShape *shape = nullptr) noexcept;

bool CompatiblePointwise(const ggml_tensor *op,
                         contracts::PointwiseSpec *spec = nullptr) noexcept;

// Register once, before model/context creation. Configuration and execution
// are confined to one host thread. All GGML C callbacks contain exceptions.
// Reads profiles.tsv and optional pointwise.tsv/ffn.tsv indexes eagerly; supports_op only
// queries their in-memory indexes. Missing members stay on another backend.
void Register(const Options &options);
Stats GetStats();
const std::string &LastError();

} // namespace coral::ggml_backend
