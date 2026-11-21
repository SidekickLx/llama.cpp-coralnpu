// SPDX-License-Identifier: MIT
#include "runtime/ffn_runtime.h"
#include "support/ffn_io.h"
#include <chrono>
#include <iostream>

namespace {
using namespace coral::runtime;
using namespace coral::validation;
using Clock = std::chrono::steady_clock;

double Milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
JSON Stats(const RuntimeStats &s) {
    return {{"module_loads", s.module_loads}, {"calls_attempted", s.calls_attempted},
            {"calls_completed", s.calls_completed}, {"coral_dispatches", s.coral_dispatches},
            {"instruction_samples", s.dispatches_with_instruction_count},
            {"kernel_instructions", s.kernel_instructions},
            {"host_executable_load_attempts", s.host_executable_load_attempts},
            {"failed_dispatches", s.failed_dispatches}, {"observer_errors", s.observer_errors},
            {"binding_allocations", s.binding_allocations}, {"upload_bytes", s.upload_bytes},
            {"download_bytes", s.download_bytes}, {"module_load_ms", s.module_load_ms},
            {"ffn_padded_rows", s.ffn_padded_rows}, {"ffn_cropped_rows", s.ffn_cropped_rows},
            {"ffn_boundary_ms", s.ffn_boundary_ms},
            {"invoke_ms", s.invoke_ms}};
}
int Run(int argc, char **argv) {
    const auto options = ParseFFNOptions(argc, argv, true);
    if (options.help) {
        std::cout << "coral-ffn-runner --artifact DIRECTORY --inputs JSON [--inputs JSON ...]\n"
                     "                 --output NEW_JSON [--repeat 3] [--recreate 1]\n"
                     "                 [--weight-layout native|transposed] [--capacity-m B]\n"
                     "Inputs: positional M3 F32 tensor JSON (X, gate, up, down).\n"
                     "All cases must share an exact profile. Each gets independent weights.\n"
                     "Output is published only after every Coral execution check succeeds.\n"
                     "Numerical reference comparison is performed by the external verifier.\n";
        return 0;
    }
    const auto input_start = Clock::now();
    const auto inputs = ReadFFNCases(options.inputs);
    const double input_ms = Milliseconds(input_start);
    auto spec = inputs.front().spec;
    spec.weight_layout = options.weight_layout;
    const auto &shape = inputs.front().spec.shape;
    if (options.capacity_m) {
        Require(options.capacity_m >= static_cast<std::size_t>(shape.m),
                "capacity cannot truncate logical rows");
        spec.shape.m = options.capacity_m;
    }
    std::vector<std::vector<float>> baselines(inputs.size());
    JSON cases = JSON::array();
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        cases.push_back({{"index", i}, {"input_sha256", inputs[i].source_sha256},
                         {"calls", JSON::array()}, {"weight_preparations_ms", JSON::array()}});
    }
    JSON cycles = JSON::array();
    std::string program;
    for (std::size_t cycle = 0; cycle < options.recreate; ++cycle) {
        auto start = Clock::now();
        DeviceContext device;
        const double device_ms = Milliseconds(start);
        start = Clock::now();
        FFNModule module(device, options.artifact, spec);
        const double module_ms = Milliseconds(start);
        if (cycle == 0) program = module.program_sha256();
        Require(program == module.program_sha256(), "artifact changed between device lifetimes");
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            const auto &values = inputs[i].values;
            start = Clock::now();
            PreparedFFNWeights weights(
                device, spec,
                TensorBinding::Dense(values[1], shape.f, shape.d),
                TensorBinding::Dense(values[2], shape.f, shape.d),
                TensorBinding::Dense(values[3], shape.d, shape.f));
            cases[i]["weight_preparations_ms"].push_back(Milliseconds(start));
            const auto binding = TensorBinding::Dense(values[0], shape.m, shape.d);
            for (std::size_t repeat = 0; repeat < options.repeat; ++repeat) {
                const auto before = device.stats();
                start = Clock::now();
                auto output = module.RunRows(binding, weights);
                const double call_ms = Milliseconds(start);
                const auto after = device.stats();
                // Finite checking is a runner policy for diagnostic fixtures;
                // the reusable runtime's tensor ABI remains general F32.
                auto tensor = FFNOutput(inputs[i].spec, output);
                if (cycle == 0 && repeat == 0) {
                    baselines[i] = std::move(output);
                    cases[i]["outputs"] = JSON::array({std::move(tensor)});
                } else {
                    Require(output == baselines[i], "repeated execution changed the output");
                }
                cases[i]["calls"].push_back({
                    {"cycle", cycle}, {"repeat", repeat},
                    {"first_call_in_context", before.calls_attempted == 0},
                    {"wall_ms", call_ms}, {"invoke_ms", after.invoke_ms - before.invoke_ms},
                    {"coral_dispatches", after.coral_dispatches - before.coral_dispatches},
                    {"instruction_samples", after.dispatches_with_instruction_count -
                                            before.dispatches_with_instruction_count},
                    {"kernel_instructions", after.kernel_instructions - before.kernel_instructions}});
            }
        }
        const auto stats = device.stats();
        Require(stats.module_loads == 1 && stats.calls_attempted == inputs.size() * options.repeat &&
                stats.calls_completed == stats.calls_attempted &&
                stats.host_executable_load_attempts == 0 &&
                stats.failed_dispatches == 0 && stats.observer_errors == 0 &&
                stats.binding_allocations == inputs.size() * 3 + 1,
                "incomplete execution or module/buffer reuse evidence");
        cycles.push_back({{"index", cycle}, {"device_create_ms", device_ms},
                          {"module_constructor_ms", module_ms}, {"stats", Stats(stats)}});
    }
    JSON report{{"schema", "coral.ffn.runner.v1"}, {"status", "passed"},
                {"engine", "coral-mpact"}, {"program_sha256", program},
                {"abi", coral::contracts::FFNAbi(spec)}, {"capacity_m", spec.shape.m},
                {"shape", {shape.m, shape.d, shape.f}}, {"repeat", options.repeat},
                {"recreate", options.recreate}, {"input_read_ms", input_ms},
                {"cases", std::move(cases)}, {"cycles", std::move(cycles)}};
    WriteNewJSON(options.output, report);
    std::cout << JSON({{"status", "passed"}, {"cases", inputs.size()},
                       {"calls", inputs.size() * options.repeat * options.recreate},
                       {"program_sha256", program}}).dump() << '\n';
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
