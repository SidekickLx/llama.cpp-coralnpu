// SPDX-License-Identifier: MIT
#include "support/ffn_io.h"
#include "ggml-cpu.h"
#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>

namespace {
using namespace coral::validation;
using Clock = std::chrono::steady_clock;
double Milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
int Run(int argc, char **argv) {
    const auto options = ParseFFNOptions(argc, argv, false);
    if (options.help) {
        std::cout << "coral-ffn-cpu-reference --inputs JSON [--inputs JSON ...]\n"
                     "                        --output NEW_JSON [--repeat 3]\n"
                     "Evaluate the positional FFN with upstream GGML CPU, one thread.\n";
        return 0;
    }
    const auto inputs = ReadFFNCases(options.inputs);
    JSON cases = JSON::array();
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        const auto setup_start = Clock::now();
        const auto &input = inputs[i];
        const auto &s = input.spec.shape;
        // Input parsing bounds every matrix and intermediate. This arena holds
        // all graph tensors plus generous graph/object alignment overhead.
        const auto bytes = static_cast<std::size_t>(2 * s.m * s.d + 3 * s.d * s.f +
                                                    3 * s.m * s.f) * sizeof(float);
        std::unique_ptr<ggml_context, decltype(&ggml_free)> context(
            ggml_init({bytes + 4 * 1024 * 1024, nullptr, false}), ggml_free);
        Require(context != nullptr, "GGML reference allocation failed");
        auto *x = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, s.d, s.m);
        auto *gate = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, s.d, s.f);
        auto *up = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, s.d, s.f);
        auto *down = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, s.f, s.d);
        ggml_tensor *arguments[] = {x, gate, up, down};
        for (std::size_t role = 0; role < 4; ++role) {
            std::memcpy(arguments[role]->data, input.values[role].data(),
                        input.values[role].size() * sizeof(float));
        }
        auto *g = ggml_mul_mat(context.get(), gate, x);
        auto *u = ggml_mul_mat(context.get(), up, x);
        auto *hidden = ggml_swiglu_split(context.get(), g, u);
        auto *output = ggml_mul_mat(context.get(), down, hidden);
        auto *graph = ggml_new_graph(context.get());
        ggml_build_forward_expand(graph, output);
        const double setup_ms = Milliseconds(setup_start);
        std::vector<float> baseline;
        JSON times = JSON::array();
        JSON tensor;
        for (std::size_t repeat = 0; repeat < options.repeat; ++repeat) {
            const auto start = Clock::now();
            Require(ggml_graph_compute_with_ctx(context.get(), graph, 1) == GGML_STATUS_SUCCESS,
                    "GGML CPU reference execution failed");
            times.push_back(Milliseconds(start));
            const auto *data = static_cast<const float *>(output->data);
            std::vector<float> actual(data, data + s.m * s.d);
            tensor = FFNOutput(input.spec, actual);
            if (repeat == 0) baseline = std::move(actual);
            else Require(actual == baseline, "GGML repeated execution changed the output");
        }
        cases.push_back({{"index", i}, {"input_sha256", input.source_sha256},
                         {"setup_ms", setup_ms}, {"evaluation_ms", std::move(times)},
                         {"outputs", JSON::array({std::move(tensor)})}});
    }
    WriteNewJSON(options.output, {{"schema", "coral.ffn.cpu-reference.v1"}, {"status", "passed"},
                 {"engine", "ggml-cpu"}, {"threads", 1}, {"cases", std::move(cases)}});
    std::cout << JSON({{"status", "passed"}, {"cases", inputs.size()}}).dump() << '\n';
    return 0;
}
} // namespace
int main(int argc, char **argv) {
    try {
        return Run(argc, argv);
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
