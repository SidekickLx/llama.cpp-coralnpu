// SPDX-License-Identifier: MIT
#pragma once
#include <cstdint>
#include <memory>
#include <string>
namespace coral::planner { struct GGMLFFNPlan; }
namespace coral::jit {
struct Stats {
    std::uint64_t requests = 0, worker_launches = 0, compilations = 0;
    std::uint64_t disk_hits = 0, recoveries = 0;
    double elapsed_ms = 0;
};
struct Artifact {
    std::string directory;
    std::string program_sha256;
};
// Single host thread. The device/VM stay with the caller; this client owns no
// compiler library. A worker is launched only after a persistent lookup misses.
class FFNJitClient {
  public:
    explicit FFNJitClient(const std::string &configuration);
    ~FFNJitClient();
    Artifact Resolve(const planner::GGMLFFNPlan &, Stats &);
    FFNJitClient(const FFNJitClient &) = delete;
    FFNJitClient &operator=(const FFNJitClient &) = delete;
  private:
    struct State;
    std::unique_ptr<State> state_;
};
} // namespace coral::jit
