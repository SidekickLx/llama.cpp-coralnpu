// SPDX-License-Identifier: MIT
#pragma once
#include "ggml-backend.h"
#include <filesystem>
#include <memory>
struct llama_context;
namespace coral::ggml_backend {
// Metadata only; neither changes placement nor reads tensor payloads.
// The scheduler/context must outlive this object. Attach before first decode.
class GraphExporter {
public:
    explicit GraphExporter(const std::filesystem::path &output);
    ~GraphExporter();
    GraphExporter(const GraphExporter &) = delete;
    GraphExporter &operator=(const GraphExporter &) = delete;
    void Attach(llama_context *context);
    void Attach(ggml_backend_sched_t scheduler);
    void Begin(int step);
    void Finish();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace coral::ggml_backend
