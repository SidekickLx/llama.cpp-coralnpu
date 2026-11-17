// SPDX-License-Identifier: MIT
#pragma once

#include "contracts/ffn_contract.h"

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace coral::planner {

// The two accepted GGML spellings lower to the same FFN function semantics.
enum class FFNPattern {
    SplitSwiGlu,
    SiluMultiply,
};

struct FFNRegionCandidate {
    contracts::GraphSnapshotId graph;
    contracts::FFNCompileSpec compile_spec;
    contracts::FFNTensorIds tensors;
    FFNPattern pattern = FFNPattern::SplitSwiGlu;

    // Complete member-node results in graph execution order. A candidate is
    // descriptive data; it is not permission to skip or execute any node.
    std::vector<contracts::TensorId> members;
};

struct ScheduleSnapshotId {
    // Nonzero process-local generation. Splitting, allocation, aliasing, or
    // observation-policy changes invalidate existing plans.
    std::uint64_t value = 0;
};

struct BackendSplitId {
    // Nonzero ID scoped to a ScheduleSnapshotId, never a backend pointer.
    std::uint64_t value = 0;
};

struct FFNRegionPlan {
    FFNRegionCandidate candidate;
    ScheduleSnapshotId schedule;
    BackendSplitId split;

    // This aggregate records a checked plan; construction alone proves nothing.
    // Only the planner may authorize it after the checks in m3-contracts.md.
    // Execution is anchored at candidate.tensors.output (the down projection).
    // Runtime bindings and loaded artifact/device ownership live elsewhere.
};

enum class RegionRejectionCode {
    UnsupportedPattern,
    UnsupportedDtype,
    UnsupportedLayout,
    UnsupportedParameters,
    ShapeMismatch,
    ExternalIntermediateUse,
    AliasOrView,
    DebugObservation,
    SplitMismatch,
    InputsNotReady,
    InputLifetimeNotProven,
    StaleGraphOrSchedule,
    MissingMemberCapability,
    MissingUseInformation,
    MissingMemberArtifact,
    MissingRegionArtifact,
    UnsupportedTarget,
};

struct RegionRejection {
    RegionRejectionCode code;
    std::optional<contracts::TensorId> tensor;
    std::string message;
};

// A rejection is an expected planning result, not a device-execution error.
// Neither alternative changes GGML placement. Actual preparation/invocation
// failures must be reported separately; they must not silently trigger replay.
using FFNPlanningResult = std::variant<FFNRegionPlan, RegionRejection>;

} // namespace coral::planner
