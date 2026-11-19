// SPDX-License-Identifier: MIT
#pragma once
#include "contracts/ffn_artifact.h"
#include <cstdint>
#include <string>
#include <vector>

namespace coral::artifacts {
// Returned bytes are the exact owned snapshot that passed integrity checks.
// Runtime loads this snapshot, never reopens a path after validation.
struct FFNArtifact {
    std::vector<std::uint8_t> bytecode;
    std::string reflection;
    std::string program_sha256;
};
std::string RuntimeCompatibilityJSON();
std::string DescribeFFNJSON(const contracts::FFNCompileSpec &spec);
std::string SHA256(const std::vector<std::uint8_t> &bytes);
FFNArtifact LoadFFNArtifact(const std::string &directory,
                            const contracts::FFNCompileSpec &expected);
} // namespace coral::artifacts
