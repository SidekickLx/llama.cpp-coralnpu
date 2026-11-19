// SPDX-License-Identifier: MIT
#include "artifacts/ffn_artifact.h"
#include <iostream>
#include <stdexcept>
#include <string>

std::int64_t Dimension(const char *text) {
    const std::string value(text);
    if (value.empty()) throw std::invalid_argument("empty dimension");
    for (char c : value) {
        if (c < '0' || c > '9') throw std::invalid_argument("dimension must be decimal");
    }
    return std::stoll(value);
}
int main(int argc, char **argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--compatibility") {
            std::cout << coral::artifacts::RuntimeCompatibilityJSON() << '\n';
            return 0;
        }
        if (argc != 5) throw std::invalid_argument("Usage: artifact-check DIRECTORY|--describe M D F");
        coral::contracts::FFNCompileSpec spec;
        spec.shape = {Dimension(argv[2]), Dimension(argv[3]), Dimension(argv[4])};
        if (std::string(argv[1]) == "--describe") {
            std::cout << coral::artifacts::DescribeFFNJSON(spec) << '\n';
        } else {
            const auto artifact = coral::artifacts::LoadFFNArtifact(argv[1], spec);
            std::cout << "{\"status\":\"passed\",\"vmfb_bytes\":" << artifact.bytecode.size()
                      << ",\"program_sha256\":\"" << artifact.program_sha256 << "\"}\n";
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
