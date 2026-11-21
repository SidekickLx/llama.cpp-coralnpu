// SPDX-License-Identifier: MIT
#pragma once
#include "contracts/ffn_artifact.h"
#include "nlohmann/json.hpp"
#include <array>
#include <string>
#include <vector>

namespace coral::validation {
using JSON = nlohmann::json;
struct FFNInputs {
    contracts::FFNCompileSpec spec;
    std::array<std::vector<float>, 4> values;
    std::string source_sha256;
};
struct FFNRunnerOptions {
    std::string artifact;
    contracts::FFNWeightLayout weight_layout = contracts::FFNWeightLayout::Native;
    std::vector<std::string> inputs;
    std::string output;
    std::size_t capacity_m = 0; // Zero keeps the exact input profile.
    std::size_t repeat = 3;
    std::size_t recreate = 1;
    bool help = false;
};
FFNRunnerOptions ParseFFNOptions(int argc, char **argv, bool coral);
FFNInputs ReadFFNInputs(const std::string &path);
std::vector<FFNInputs> ReadFFNCases(const std::vector<std::string> &paths);
JSON FFNOutput(const contracts::FFNCompileSpec &, const std::vector<float> &);
void WriteNewJSON(const std::string &path, const JSON &);
void Require(bool condition, const std::string &message);
} // namespace coral::validation
