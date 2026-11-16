// SPDX-License-Identifier: MIT
#include "matmul_runtime.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using coral::runtime::DeviceContext;
using coral::runtime::LoadedModule;
using coral::runtime::MatmulShape;
using coral::runtime::PreparedWeight;
using coral::runtime::RuntimeStats;
using coral::runtime::TensorBinding;

constexpr double kRtol = 1e-5;
constexpr double kAtol = 1e-6;

struct Options {
    std::string module;
    std::string function = "matmul";
    std::string input;
    std::string weights;
    std::string output;
    MatmulShape shape;
    std::size_t repeat = 3;
    std::size_t recreate = 1;
    bool self_test = false;
    bool help = false;
};

void Usage() {
    std::cout << "coral-matmul-runner --module FILE [options]\n"
              << "  --m M --k K --n N       Static shape (default: 1,16,8)\n"
              << "  --function NAME        Unqualified VMFB export (default: matmul)\n"
              << "  --input FILE           Dense little-endian F32 X[M,K]\n"
              << "  --weights FILE         Dense little-endian F32 W[N,K]\n"
              << "                         Supply both, or use deterministic fixtures.\n"
              << "  --output FILE          Write baseline Y only after all checks pass\n"
              << "  --repeat COUNT         Calls per loaded module (default: 3)\n"
              << "  --recreate COUNT       Recreate context/module/weights (default: 1)\n"
              << "  --self-test            Changed X, alternate W, restore W, zero X,\n"
              << "                         weight snapshot and invalid binding checks\n"
              << "  --help                 Show this help without creating a device\n"
              << "Every successful call requires real MPACT execution and a CPU reference\n"
              << "match (rtol=1e-5, atol=1e-6). Host numerical kernels are disabled.\n";
}

std::size_t PositiveInteger(const std::string &text) {
    std::size_t value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || value == 0) {
        throw std::invalid_argument("expected a positive integer, got: " + text);
    }
    return value;
}

Options ParseOptions(int argc, char **argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--help") {
            options.help = true;
            continue;
        }
        if (flag == "--self-test") {
            options.self_test = true;
            continue;
        }
        if (i + 1 >= argc) {
            throw std::invalid_argument("missing value for " + flag);
        }
        const std::string value = argv[++i];
        if (flag == "--module") {
            options.module = value;
        } else if (flag == "--function") {
            options.function = value;
        } else if (flag == "--input") {
            options.input = value;
        } else if (flag == "--weights") {
            options.weights = value;
        } else if (flag == "--output") {
            options.output = value;
        } else if (flag == "--m") {
            options.shape.m = PositiveInteger(value);
        } else if (flag == "--k") {
            options.shape.k = PositiveInteger(value);
        } else if (flag == "--n") {
            options.shape.n = PositiveInteger(value);
        } else if (flag == "--repeat") {
            options.repeat = PositiveInteger(value);
        } else if (flag == "--recreate") {
            options.recreate = PositiveInteger(value);
        } else {
            throw std::invalid_argument("unknown option: " + flag);
        }
    }
    if (!options.help) {
        if (options.module.empty()) {
            throw std::invalid_argument("--module is required");
        }
        if (options.input.empty() != options.weights.empty()) {
            throw std::invalid_argument("--input and --weights must be supplied together");
        }
        if (!options.output.empty() &&
            (options.output == options.input || options.output == options.weights ||
             options.output == options.module)) {
            throw std::invalid_argument("output must not overwrite an input or VMFB");
        }
        options.shape.Validate();
    }
    return options;
}

std::vector<float> ReadF32(const std::string &path, std::size_t count) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    const auto bytes = count * sizeof(float);
    if (!input || input.tellg() != static_cast<std::streamoff>(bytes)) {
        throw std::runtime_error("input file has wrong size or cannot be read: " + path);
    }
    input.seekg(0);
    std::vector<float> values(count);
    if (!input.read(reinterpret_cast<char *>(values.data()), static_cast<std::streamsize>(bytes))) {
        throw std::runtime_error("incomplete F32 input: " + path);
    }
    for (const auto value : values) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("M1 fixtures require finite F32 values: " + path);
        }
    }
    return values;
}

std::vector<float> Fixture(std::size_t count, std::size_t factor) {
    std::vector<float> values(count);
    for (std::size_t i = 0; i < count; ++i) {
        values[i] = static_cast<float>(static_cast<int>((i * factor + 3) % 19) - 9) / 8.0F;
    }
    return values;
}

// Independent row-major reference with double accumulation, rounded to F32.
// It is only a validation oracle; its output is never used for device fallback.
std::vector<float> Reference(MatmulShape shape, const std::vector<float> &x,
                             const std::vector<float> &w) {
    std::vector<float> result(shape.OutputElements());
    for (std::size_t m = 0; m < shape.m; ++m) {
        for (std::size_t n = 0; n < shape.n; ++n) {
            double value = 0;
            for (std::size_t k = 0; k < shape.k; ++k) {
                value += static_cast<double>(x[m * shape.k + k]) *
                         static_cast<double>(w[n * shape.k + k]);
            }
            result[m * shape.n + n] = static_cast<float>(value);
        }
    }
    return result;
}

std::vector<float> CheckCase(const char *name, LoadedModule &module, const PreparedWeight &prepared,
                             MatmulShape shape, const std::vector<float> &x,
                             const std::vector<float> &w) {
    const auto expected = Reference(shape, x, w);
    auto result = module.Run(TensorBinding::Dense(x, shape.m, shape.k), prepared);
    if (result.size() != expected.size()) {
        throw std::runtime_error("output element count differs from reference");
    }
    std::size_t mismatches = 0;
    double max_absolute_error = 0;
    for (std::size_t i = 0; i < result.size(); ++i) {
        if (!std::isfinite(result[i]) || !std::isfinite(expected[i])) {
            throw std::runtime_error("non-finite device result or reference");
        }
        const double error = std::abs(static_cast<double>(result[i]) - expected[i]);
        max_absolute_error = std::max(max_absolute_error, error);
        if (error > kAtol + kRtol * std::abs(static_cast<double>(expected[i]))) {
            ++mismatches;
        }
    }
    std::cout << "{\"case\":\"" << name << "\",\"max_abs_error\":" << max_absolute_error
              << ",\"mismatches\":" << mismatches
              << ",\"passed\":" << (mismatches == 0 ? "true" : "false") << "}\n";
    if (mismatches != 0) {
        throw std::runtime_error(std::string(name) + ": device/reference mismatch");
    }
    return result;
}

void ExpectInvalidBinding(LoadedModule &module, TensorBinding binding,
                          const PreparedWeight &weights) {
    try {
        module.Run(binding, weights);
    } catch (const std::invalid_argument &) {
        return;
    }
    throw std::runtime_error("invalid tensor binding was not rejected");
}

void PrintStats(std::size_t cycle, MatmulShape shape, const RuntimeStats &stats) {
    std::cout << "{\"cycle\":" << cycle << ",\"m\":" << shape.m << ",\"k\":" << shape.k
              << ",\"n\":" << shape.n << ",\"module_loads\":" << stats.module_loads
              << ",\"calls_attempted\":" << stats.calls_attempted
              << ",\"calls_completed\":" << stats.calls_completed
              << ",\"coral_dispatches\":" << stats.coral_dispatches
              << ",\"failed_dispatches\":" << stats.failed_dispatches
              << ",\"instruction_samples\":" << stats.dispatches_with_instruction_count
              << ",\"kernel_instructions\":" << stats.kernel_instructions
              << ",\"observer_errors\":" << stats.observer_errors
              << ",\"host_executable_load_attempts\":" << stats.host_executable_load_attempts
              << ",\"host_numerical_execution\":\"disabled\""
              << ",\"binding_allocations\":" << stats.binding_allocations
              << ",\"upload_bytes\":" << stats.upload_bytes
              << ",\"download_bytes\":" << stats.download_bytes
              << ",\"module_load_ms\":" << stats.module_load_ms
              << ",\"invoke_ms\":" << stats.invoke_ms << "}\n";
}

int Run(const Options &options) {
    const std::uint16_t endian_probe = 1;
    if (*reinterpret_cast<const unsigned char *>(&endian_probe) != 1) {
        throw std::runtime_error("M1 raw F32 I/O currently requires a little-endian host");
    }
    const auto shape = options.shape;
    const auto x = options.input.empty() ? Fixture(shape.InputElements(), 7)
                                         : ReadF32(options.input, shape.InputElements());
    const auto w = options.weights.empty() ? Fixture(shape.WeightElements(), 11)
                                           : ReadF32(options.weights, shape.WeightElements());
    std::vector<float> baseline;
    for (std::size_t cycle = 0; cycle < options.recreate; ++cycle) {
        DeviceContext device;
        try {
            LoadedModule module(device, options.module, shape, options.function);
            auto source_weights = w;
            PreparedWeight prepared(device, TensorBinding::Dense(source_weights, shape.n, shape.k));
            if (options.self_test) {
                // Snapshot ownership: changing the original host vector must
                // not change the already prepared weights.
                std::fill(source_weights.begin(), source_weights.end(), 123.0F);
            }
            for (std::size_t i = 0; i < options.repeat; ++i) {
                baseline = CheckCase("baseline", module, prepared, shape, x, w);
            }
            if (options.self_test) {
                auto changed_x = x;
                for (auto &value : changed_x) {
                    value = value * 0.5F + 0.25F;
                }
                CheckCase("changed_activation", module, prepared, shape, changed_x, w);

                // Ones make the +0.25 weight change observable even when the
                // original activation was zero or sums to zero.
                const std::vector<float> probe(shape.InputElements(), 1.0F);
                auto alternate_w = w;
                for (auto &value : alternate_w) {
                    value += 0.25F;
                }
                PreparedWeight alternate(device,
                                         TensorBinding::Dense(alternate_w, shape.n, shape.k));
                CheckCase("alternate_weight", module, alternate, shape, probe, alternate_w);
                CheckCase("restore_weight", module, prepared, shape, probe, w);
                CheckCase("zero_activation", module, prepared, shape,
                          std::vector<float>(shape.InputElements(), 0.0F), w);

                const auto before = device.stats().calls_attempted;
                auto bad_stride = TensorBinding::Dense(x, shape.m, shape.k);
                bad_stride.row_stride_bytes += sizeof(float);
                ExpectInvalidBinding(module, bad_stride, prepared);

                auto bad_size = TensorBinding::Dense(x, shape.m, shape.k);
                --bad_size.element_count;
                ExpectInvalidBinding(module, bad_size, prepared);

                if (device.stats().calls_attempted != before) {
                    throw std::runtime_error("invalid binding reached device execution");
                }
                CheckCase("after_rejected_binding", module, prepared, shape, x, w);
            }
            PrintStats(cycle, shape, device.stats());
        } catch (...) {
            PrintStats(cycle, shape, device.stats());
            throw;
        }
    }
    if (!options.output.empty()) {
        // Exclusive creation on the supported Linux host: no overwrite and no
        // check/open race. No output file is opened until every check passes.
        std::unique_ptr<std::FILE, decltype(&std::fclose)> output(
            std::fopen(options.output.c_str(), "wbx"), &std::fclose);
        if (!output) {
            throw std::runtime_error("output exists or cannot be created: " + options.output);
        }
        if (std::fwrite(baseline.data(), sizeof(float), baseline.size(), output.get()) !=
            baseline.size()) {
            throw std::runtime_error("cannot write complete output: " + options.output);
        }
        if (std::fclose(output.release()) != 0) {
            throw std::runtime_error("cannot finish output: " + options.output);
        }
    }
    return 0;
}
} // namespace

int main(int argc, char **argv) {
    try {
        const auto options = ParseOptions(argc, argv);
        if (options.help) {
            Usage();
            return 0;
        }
        std::cout << std::setprecision(9);
        return Run(options);
    } catch (const std::invalid_argument &error) {
        std::cerr << "Invalid argument: " << error.what() << '\n';
        return 2;
    } catch (const std::exception &error) {
        std::cerr << "M1 runner failed: " << error.what() << '\n';
        return 1;
    }
}
