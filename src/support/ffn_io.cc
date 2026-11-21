// SPDX-License-Identifier: MIT
#include "support/ffn_io.h"
#include "artifacts/ffn_artifact.h"
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <fcntl.h>
#include <filesystem>
#include <functional>
#include <set>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

namespace coral::validation {
void Require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}
namespace {
constexpr std::size_t kMaxElements = 1 << 20;
constexpr std::size_t kMaxJSONBytes = 16 << 20;
struct File {
    int fd;
    explicit File(int value) : fd(value) { Require(fd >= 0, "cannot open input/output file"); }
    ~File() { close(fd); }
    File(const File &) = delete;
    File &operator=(const File &) = delete;
};
struct UniqueKeys {
    std::vector<std::set<std::string>> objects;
    bool operator()(int depth, JSON::parse_event_t event, JSON &value) {
        Require(depth <= 16, "tensor JSON nesting limit exceeded");
        if (event == JSON::parse_event_t::object_start) objects.emplace_back();
        if (event == JSON::parse_event_t::key) {
            Require(!objects.empty() && objects.back().insert(value.get<std::string>()).second,
                    "duplicate tensor JSON key");
        }
        if (event == JSON::parse_event_t::object_end) objects.pop_back();
        return true;
    }
};
std::size_t Count(const std::string &text, std::size_t maximum) {
    std::size_t count = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), count);
    Require(result.ec == std::errc{} && result.ptr == text.data() + text.size() &&
            count > 0 && count <= maximum, "invalid or excessive repetition count");
    return count;
}
std::int64_t Dimension(const JSON &value) {
    Require(value.is_number_integer() && value > 0 && value <= kMaxElements,
            "tensor dimensions must be bounded positive integers");
    return value.get<std::int64_t>();
}
} // namespace

FFNRunnerOptions ParseFFNOptions(int argc, char **argv, bool coral) {
    FFNRunnerOptions result;
    std::set<std::string> seen;
    for (int i = 1; i < argc; ++i) {
        const std::string flag(argv[i]);
        if (flag == "--help") {
            result.help = true;
            continue;
        }
        Require(i + 1 < argc, "missing option value: " + flag);
        Require(flag == "--inputs" || seen.insert(flag).second, "duplicate option: " + flag);
        const std::string value(argv[++i]);
        if (flag == "--inputs") result.inputs.push_back(value);
        else if (flag == "--output") result.output = value;
        else if (flag == "--artifact" && coral) result.artifact = value;
        else if (flag == "--weight-layout" && coral) {
            Require(value == "native" || value == "transposed", "unknown FFN weight layout");
            if (value == "transposed") result.weight_layout = contracts::FFNWeightLayout::Transposed;
        }
        else if (flag == "--capacity-m" && coral) result.capacity_m = Count(value, kMaxElements);
        else if (flag == "--repeat") result.repeat = Count(value, 100);
        else if (flag == "--recreate" && coral) result.recreate = Count(value, 16);
        else throw std::invalid_argument("unknown option: " + flag);
    }
    if (result.help) return result;
    Require(!result.inputs.empty() && result.inputs.size() <= 64 && !result.output.empty() &&
            (!coral || !result.artifact.empty()), "inputs, new output and (for Coral) artifact are required");
    Require(result.repeat * result.recreate * result.inputs.size() <= 1024,
            "runner invocation budget exceeded");
    // Check early for usability; atomic publication below also handles races.
    Require(!std::filesystem::exists(std::filesystem::symlink_status(result.output)),
            "output already exists");
    return result;
}

FFNInputs ReadFFNInputs(const std::string &path) {
    File file(open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK));
    struct stat info{};
    Require(fstat(file.fd, &info) == 0 && S_ISREG(info.st_mode) &&
            info.st_size > 0 && info.st_size <= static_cast<off_t>(kMaxJSONBytes),
            "input must be a nonempty regular JSON file of at most 16 MiB");
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(info.st_size));
    std::size_t offset = 0;
    while (offset < bytes.size()) {
        const auto count = read(file.fd, bytes.data() + offset, bytes.size() - offset);
        if (count < 0 && errno == EINTR) continue;
        Require(count > 0, "incomplete tensor JSON read");
        offset += count;
    }
    UniqueKeys keys;
    const auto document = JSON::parse(bytes.begin(), bytes.end(), std::ref(keys));
    Require(document.is_object() && document.size() == 1 && document.contains("inputs") &&
            document.at("inputs").is_array() && document.at("inputs").size() == 4,
            "expected exactly four positional FFN inputs");
    FFNInputs result;
    result.source_sha256 = artifacts::SHA256(bytes);
    std::array<std::array<std::int64_t, 2>, 4> shapes;
    std::size_t elements = 0;
    for (std::size_t i = 0; i < 4; ++i) {
        const auto &tensor = document.at("inputs").at(i);
        Require(tensor.is_object() && tensor.size() == 3 && tensor.contains("dtype") &&
                tensor.contains("shape") && tensor.contains("values") &&
                tensor.at("dtype") == "f32" && tensor.at("shape").is_array() &&
                tensor.at("shape").size() == 2 && tensor.at("values").is_array(),
                "each tensor requires dtype=f32, rank-two shape and flat values");
        shapes[i] = {Dimension(tensor.at("shape").at(0)), Dimension(tensor.at("shape").at(1))};
        const auto count = static_cast<std::size_t>(shapes[i][0] * shapes[i][1]);
        Require(count <= kMaxElements && elements <= kMaxElements - count,
                "FFN input element budget exceeded");
        elements += count;
        Require(tensor.at("values").size() == count, "tensor value count differs from its shape");
        result.values[i].reserve(count);
        for (const auto &number : tensor.at("values")) {
            Require(number.is_number(), "tensor values must be numbers");
            const double value = number.get<double>();
            Require(std::isfinite(value) && std::isfinite(static_cast<float>(value)),
                    "tensor values must be finite F32 numbers");
            result.values[i].push_back(static_cast<float>(value));
        }
    }
    result.spec.shape = {shapes[0][0], shapes[0][1], shapes[1][0]};
    contracts::ValidateFFNSpec(result.spec);
    const auto &s = result.spec.shape;
    Require(shapes[1] == std::array<std::int64_t, 2>{s.f, s.d} &&
            shapes[2] == shapes[1] && shapes[3] == std::array<std::int64_t, 2>{s.d, s.f},
            "FFN role shapes must be X[M,D], gate[F,D], up[F,D], down[D,F]");
    Require(static_cast<std::size_t>(s.m * s.f) <= kMaxElements, "FFN intermediate element budget exceeded");
    return result;
}
std::vector<FFNInputs> ReadFFNCases(const std::vector<std::string> &paths) {
    std::vector<FFNInputs> inputs;
    std::size_t elements = 0;
    for (const auto &path : paths) {
        auto input = ReadFFNInputs(path);
        for (const auto &values : input.values) elements += values.size();
        Require(elements <= 4 * kMaxElements, "runner total input element budget exceeded");
        if (!inputs.empty()) {
            const auto &a = inputs.front().spec.shape;
            const auto &b = input.spec.shape;
            Require(a.m == b.m && a.d == b.d && a.f == b.f,
                    "all cases in a runner process must share one exact FFN profile");
        }
        inputs.push_back(std::move(input));
    }
    return inputs;
}
JSON FFNOutput(const contracts::FFNCompileSpec &spec, const std::vector<float> &values) {
    Require(values.size() == static_cast<std::size_t>(spec.shape.m * spec.shape.d),
            "FFN output value count mismatch");
    for (float value : values) Require(std::isfinite(value), "non-finite FFN output");
    return {{"dtype", "f32"}, {"shape", {spec.shape.m, spec.shape.d}}, {"values", values}};
}
void WriteNewJSON(const std::string &path, const JSON &value) {
    const auto text = value.dump(2) + "\n";
    // The temporary file is in the destination directory. link() publishes
    // the complete regular file atomically and refuses any existing target.
    std::string pattern = path + ".tmp-XXXXXX";
    std::vector<char> name(pattern.begin(), pattern.end());
    name.push_back('\0');
    File file(mkstemp(name.data()));
    try {
        std::size_t offset = 0;
        while (offset < text.size()) {
            const auto count = write(file.fd, text.data() + offset, text.size() - offset);
            if (count < 0 && errno == EINTR) continue;
            Require(count > 0, "cannot write result JSON");
            offset += count;
        }
        Require(fsync(file.fd) == 0, "cannot flush result JSON");
        Require(link(name.data(), path.c_str()) == 0, "cannot publish result without overwriting");
    } catch (...) {
        unlink(name.data());
        throw;
    }
    unlink(name.data());
}
} // namespace coral::validation
