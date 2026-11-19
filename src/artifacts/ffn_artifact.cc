// SPDX-License-Identifier: MIT
#include "artifacts/ffn_artifact.h"
#include "build_identity.h"
#include "nlohmann/json.hpp"
extern "C" {
#include "sha256/sha256.h"
}
#include <cerrno>
#include <fcntl.h>
#include <functional>
#include <map>
#include <set>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

namespace coral::artifacts {
namespace {
using JSON = nlohmann::json;
void Require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error("FFN artifact: " + message);
}
struct File {
    int fd;
    explicit File(int value) : fd(value) {
        Require(fd >= 0, "cannot open regular artifact file or directory");
    }
    ~File() { close(fd); }
    File(const File &) = delete;
    File &operator=(const File &) = delete;
};
std::vector<std::uint8_t> Read(int directory, const char *name, std::size_t limit) {
    File file(openat(directory, name, O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK));
    struct stat info{};
    Require(fstat(file.fd, &info) == 0 && S_ISREG(info.st_mode) &&
            info.st_size > 0 && static_cast<std::uint64_t>(info.st_size) <= limit,
            std::string(name) + ": not a bounded nonempty regular file");
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(info.st_size));
    std::size_t offset = 0;
    while (offset < bytes.size()) {
        const auto count = read(file.fd, bytes.data() + offset, bytes.size() - offset);
        if (count < 0 && errno == EINTR) continue;
        Require(count > 0, std::string(name) + ": incomplete read");
        offset += static_cast<std::size_t>(count);
    }
    return bytes;
}
struct UniqueKeys {
    std::vector<std::set<std::string>> objects;
    bool operator()(int depth, JSON::parse_event_t event, JSON &value) {
        Require(depth <= 32, "JSON nesting limit exceeded");
        if (event == JSON::parse_event_t::object_start) objects.emplace_back();
        if (event == JSON::parse_event_t::key) {
            Require(!objects.empty() && objects.back().insert(value.get<std::string>()).second,
                    "duplicate JSON key");
        }
        if (event == JSON::parse_event_t::object_end) objects.pop_back();
        return true;
    }
};
JSON Parse(const std::vector<std::uint8_t> &bytes) {
    UniqueKeys keys;
    return JSON::parse(bytes.begin(), bytes.end(), std::ref(keys));
}
void Keys(const JSON &object, std::initializer_list<const char *> expected) {
    Require(object.is_object() && object.size() == expected.size(), "unexpected object fields");
    for (auto key : expected) Require(object.contains(key), std::string("missing field: ") + key);
}
void Same(const JSON &actual, const JSON &expected, const char *label) {
    // JSON numeric equality alone accepts 1.0 as 1. Preserve wire types.
    if (expected.is_number_integer()) {
        Require(actual.is_number_integer(), std::string(label) + ": integer required");
    } else {
        Require(actual.type() == expected.type(), std::string(label) + ": incorrect type");
    }
    Require(actual == expected, std::string(label) + ": incompatible value");
    if (expected.is_structured()) {
        for (auto it = expected.begin(); it != expected.end(); ++it) {
            if (expected.is_object()) Same(actual.at(it.key()), it.value(), label);
            else Same(actual.at(it - expected.begin()), it.value(), label);
        }
    }
}
std::string Digest(const JSON &value) {
    Require(value.is_string(), "digest must be a string");
    const auto text = value.get<std::string>();
    Require(text.size() == 64, "digest must have 64 lowercase hexadecimal characters");
    for (char c : text) Require((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'), "invalid digest");
    return text;
}
std::string HashText(const std::string &text) {
    return SHA256(std::vector<std::uint8_t>(text.begin(), text.end()));
}
JSON Compatibility() {
    const auto lock = JSON::parse(kFFNDependencyLock);
    const auto &target = lock.at("target_profile");
    return JSON{
        {"host", "linux-x86_64"}, {"simulator", "mpact"},
        {"dependency_lock_sha256", kFFNDependencyLockHash},
        {"repositories", lock.at("repositories")}, {"patches", lock.at("upstream_patch_sha256")},
        {"runtime_sources", JSON::parse(kFFNRuntimeSources)},
        {"target", {{"triple", "riscv32-unknown-elf"}, {"abi", target.at("abi")},
                    {"features", target.at("default_features")},
                    {"itcm_bytes", target.at("itcm_bytes")}, {"dtcm_bytes", target.at("dtcm_bytes")},
                    {"highmem", false}, {"stack_bytes", 4096}, {"dtcm_reserve_bytes", 4096}}}};
}
JSON Binding(const char *role, std::int64_t rows, std::int64_t cols) {
    return {{"role", role}, {"dtype", "f32"}, {"layout", "row-major"}, {"shape", {rows, cols}}};
}
JSON Contract(const contracts::FFNCompileSpec &spec) {
    contracts::ValidateFFNSpec(spec);
    const auto &s = spec.shape;
    return {{"abi", contracts::kFFNAbi}, {"entry", contracts::kFFNEntryPoint},
            {"numeric_policy", contracts::kFFNNumericPolicy},
            {"shape", {{"m", s.m}, {"d", s.d}, {"f", s.f}}},
            {"inputs", JSON::array({Binding("activation", s.m, s.d),
                Binding("gate_weight", s.f, s.d), Binding("up_weight", s.f, s.d),
                Binding("down_weight", s.d, s.f)})},
            {"output", Binding("output", s.m, s.d)}};
}
JSON CompilerOptions() {
    const auto target = Compatibility().at("target");
    return JSON::array({
        "--iree-input-type=stablehlo", "--iree-execution-model=async-internal",
        "--iree-hal-target-device=local", "--iree-hal-local-target-device-backends=llvm-cpu",
        "--iree-llvmcpu-target-cpu=generic", "--iree-hal-target-device=coralnpu",
        "--coralnpu-fast-math=false", "--coralnpu-roofline-speedup-threshold=0",
        "--coralnpu-target-abi=" + target.at("abi").get<std::string>(),
        "--coralnpu-target-cpu-features=" + target.at("features").get<std::string>(),
        "--coralnpu-dtcm-size-kb=32", "--coralnpu-stack-size-kb=4",
        "--coralnpu-dtcm-reserve-kb=4", "--coralnpu-dtcm-buffers-per-operand=1",
        "--coralnpu-link-executables=false", "--coralnpu-max-loop-unrolling=32"});
}
} // namespace

std::string SHA256(const std::vector<std::uint8_t> &bytes) {
    unsigned char digest[SHA256_DIGEST_SIZE];
    sha256_hash(digest, bytes.data(), bytes.size());
    constexpr char hex[] = "0123456789abcdef";
    std::string result;
    for (auto byte : digest) {
        result.push_back(hex[byte >> 4]);
        result.push_back(hex[byte & 15]);
    }
    return result;
}
std::string RuntimeCompatibilityJSON() { return Compatibility().dump(); }
std::string DescribeFFNJSON(const contracts::FFNCompileSpec &spec) {
    return JSON{{"contract", Contract(spec)}, {"compatibility", Compatibility()},
                {"compiler_options", CompilerOptions()}, {"binding_tag", contracts::FFNBindingTag(spec)},
                {"compatibility_sha256", HashText(Compatibility().dump())}}.dump();
}
FFNArtifact LoadFFNArtifact(const std::string &directory,
                            const contracts::FFNCompileSpec &expected) {
    contracts::ValidateFFNSpec(expected);
    File root(open(directory.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC));
    const auto manifest = Parse(Read(root.fd, "manifest.json", 1024 * 1024));
    Keys(manifest, {"schema", "state", "contract", "compatibility", "program",
                    "compiler", "files", "placement", "reflection"});
    Same(manifest.at("schema"), contracts::kFFNArtifactSchema, "schema");
    Same(manifest.at("state"), "complete", "publication");
    Same(manifest.at("contract"), Contract(expected), "contract");
    Same(manifest.at("compatibility"), Compatibility(), "runtime compatibility");
    const auto &program = manifest.at("program");
    Keys(program, {"pipeline", "semantic_sha256"});
    Same(program.at("pipeline"), contracts::kFFNPipeline, "pipeline");
    const auto semantic = Digest(program.at("semantic_sha256"));
    const auto &compiler = manifest.at("compiler");
    Keys(compiler, {"binary_sha256", "support_sha256", "options"});
    Digest(compiler.at("binary_sha256"));
    Keys(compiler.at("support_sha256"), {"linker", "linker_script", "iree_archive", "crt_archive"});
    for (const auto &value : compiler.at("support_sha256")) Digest(value);
    Same(compiler.at("options"), CompilerOptions(), "compiler options");
    const auto &placement = manifest.at("placement");
    Keys(placement, {"numerical_target", "host_dispatches", "coral_dispatches"});
    Same(placement.at("numerical_target"), "coralnpu", "numerical placement");
    Same(placement.at("host_dispatches"), 0, "host dispatches");
    Require(placement.at("coral_dispatches").is_number_integer() &&
            placement.at("coral_dispatches") > 0 &&
            placement.at("coral_dispatches") <= 1024, "invalid Coral dispatch count");
    const auto &files = manifest.at("files");
    Keys(files, {"region.vmfb", "region.stablehlo.mlir", "region.metadata.json", "placement.json"});
    FFNArtifact artifact;
    artifact.program_sha256 = semantic;
    for (const auto *name : {"region.vmfb", "region.stablehlo.mlir",
                             "region.metadata.json", "placement.json"}) {
        const auto limit = std::string(name) == "region.vmfb" ? 512u * 1024 * 1024 : 16u * 1024 * 1024;
        auto bytes = Read(root.fd, name, limit);
        const auto digest = SHA256(bytes);
        Require(digest == Digest(files.at(name)), std::string(name) + ": SHA-256 mismatch");
        if (std::string(name) == "region.stablehlo.mlir") {
            Require(digest == semantic, "semantic IR identity mismatch");
        } else if (std::string(name) == "region.vmfb") {
            artifact.bytecode = std::move(bytes);
        } else if (std::string(name) == "placement.json") {
            // The producer supplies detailed dispatch evidence; the summary
            // cannot disagree with its hashed report.
            const auto evidence = Parse(bytes);
            Keys(evidence, {"summary", "dispatches"});
            Same(evidence.at("summary"), placement, "placement summary");
            Require(evidence.at("dispatches").is_array() &&
                    evidence.at("dispatches").size() == placement.at("coral_dispatches"),
                    "incomplete placement evidence");
            for (const auto &dispatch : evidence.at("dispatches")) {
                Keys(dispatch, {"name", "target"});
                Require(dispatch.at("name").is_string() && !dispatch.at("name").get<std::string>().empty(),
                        "dispatch name missing");
                Same(dispatch.at("target"), "coralnpu", "dispatch target");
            }
        } else {
            Require(Parse(bytes).is_object(), "source metadata must be an object");
        }
    }
    artifact.reflection = contracts::FFNBindingTag(expected) +
        ";policy=" + contracts::kFFNNumericPolicy + ";target=" + HashText(Compatibility().dump()) +
        ";program=" + semantic + ";pipeline=" + contracts::kFFNPipeline;
    Same(manifest.at("reflection"), artifact.reflection, "reflection declaration");
    return artifact;
}
} // namespace coral::artifacts
