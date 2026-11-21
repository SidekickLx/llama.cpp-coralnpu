// SPDX-License-Identifier: MIT
#include "jit/ffn_jit.h"
#include "artifacts/ffn_artifact.h"
#include "planner/ggml_fusion.h"
#include "support/strict_json.h"
extern "C" {
#include "sha256/sha256.h"
}
#include <array>
#include <chrono>
#include <csignal>
#include <cstring>
#include <filesystem>
#include <map>
#include <spawn.h>
#include <sys/wait.h>
#include <thread>

namespace coral::jit {
namespace {
namespace fs = std::filesystem;
using JSON = json::Value;
using json::Require;
using Clock = std::chrono::steady_clock;
std::string Hash(const JSON &value) {
    const auto text = json::Canonical(value);
    return artifacts::SHA256({text.begin(), text.end()});
}
std::string Hex(const unsigned char *bytes) {
    constexpr char hex[] = "0123456789abcdef";
    std::string text;
    for (std::size_t i = 0; i < SHA256_DIGEST_SIZE; ++i) {
        text.push_back(hex[bytes[i] >> 4]);
        text.push_back(hex[bytes[i] & 15]);
    }
    return text;
}
std::string HashFile(const std::string &path) {
    const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK);
    Require(fd >= 0, "missing JIT toolchain file: " + path);
    sha256_t state;
    sha256_init(&state);
    try {
        struct stat info{};
        Require(fstat(fd, &info) == 0 && S_ISREG(info.st_mode), "invalid JIT toolchain file");
        std::array<unsigned char, 64 * 1024> buffer;
        for (;;) {
            const auto count = read(fd, buffer.data(), buffer.size());
            if (count < 0 && errno == EINTR) continue;
            Require(count >= 0, "cannot hash JIT toolchain file");
            if (count == 0) break;
            sha256_update(&state, buffer.data(), count);
        }
    } catch (...) {
        close(fd);
        throw;
    }
    close(fd);
    unsigned char bytes[SHA256_DIGEST_SIZE];
    sha256_final(&state, bytes);
    return Hex(bytes);
}
std::string Digest(const JSON &value) {
    Require(value.is_string(), "digest must be a string");
    const auto text = value.get<std::string>();
    Require(text.size() == 64 && text.find_first_not_of("0123456789abcdef") == std::string::npos,
            "invalid SHA-256 digest");
    return text;
}
void Write(const fs::path &path, const JSON &value) {
    const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    Require(fd >= 0, "cannot create JIT evidence file");
    try {
        const auto text = json::Canonical(value) + "\n";
        std::size_t offset = 0;
        while (offset < text.size()) {
            const auto count = write(fd, text.data() + offset, text.size() - offset);
            if (count < 0 && errno == EINTR) continue;
            Require(count > 0, "cannot write JIT evidence");
            offset += count;
        }
    } catch (...) {
        close(fd);
        throw;
    }
    close(fd);
}
JSON Snapshot(const planner::GGMLFFNPlan &plan) {
    std::map<const ggml_tensor *, std::size_t> ids;
    std::vector<const ggml_tensor *> tensors;
    JSON inputs = JSON::array(), operations = JSON::array(), nodes = JSON::array();
    for (auto *tensor : plan.inputs) {
        auto inserted = ids.emplace(tensor, tensors.size() + 1);
        if (inserted.second) tensors.push_back(tensor);
        inputs.push_back(inserted.first->second);
    }
    const auto boundary = tensors.size();
    for (auto *tensor : plan.members) {
        Require(ids.emplace(tensor, tensors.size() + 1).second, "duplicate FFN snapshot member");
        tensors.push_back(tensor);
        operations.push_back(tensors.size());
    }
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        const auto &tensor = *tensors[i];
        JSON ne = JSON::array(), nb = JSON::array(), params = JSON::array(), sources = JSON::array();
        for (int axis = 0; axis < GGML_MAX_DIMS; ++axis) {
            ne.push_back(tensor.ne[axis]);
            nb.push_back(tensor.nb[axis]);
        }
        // Boundary inputs become function arguments. Their external producers
        // and payloads do not cross the compiler process boundary.
        for (const auto word : tensor.op_params) params.push_back(i < boundary ? 0 : word);
        for (const auto *source : tensor.src) {
            sources.push_back(i < boundary || !source ? 0 : ids.at(source));
        }
        nodes.push_back({{"type", tensor.type}, {"ne", ne}, {"nb", nb},
                         {"op", i < boundary ? GGML_OP_NONE : tensor.op},
                         {"op_params", params}, {"src", sources}});
    }
    return {{"schema", "coral.ggml.ffn-snapshot.v1"},
            {"shape", {plan.shape.m, plan.shape.d, plan.shape.f}},
            {"inputs", inputs}, {"nodes", nodes}, {"operations", operations},
            {"output", ids.at(plan.members.back())}};
}
struct Elapsed {
    Stats &stats;
    Clock::time_point start = Clock::now();
    ~Elapsed() {
        stats.elapsed_ms += std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    }
};
struct Spawn {
    posix_spawn_file_actions_t actions;
    posix_spawnattr_t attributes;
    Spawn() {
        Require(posix_spawn_file_actions_init(&actions) == 0, "spawn file actions initialization");
        if (posix_spawnattr_init(&attributes) != 0) {
            posix_spawn_file_actions_destroy(&actions);
            throw std::runtime_error("spawn attribute initialization");
        }
    }
    ~Spawn() {
        posix_spawnattr_destroy(&attributes);
        posix_spawn_file_actions_destroy(&actions);
    }
};
void Launch(const JSON &config, const fs::path &work) {
    const auto &paths = config.at("paths");
    std::vector<std::string> words{
        paths.at("python"), paths.at("jit_worker.py"), "compile",
        "--config", (work / "config.json").string(),
        "--request", (work / "request.json").string()};
    std::vector<char *> argv;
    for (auto &word : words) argv.push_back(word.data());
    argv.push_back(nullptr);
    std::vector<std::string> environment{
        "PATH=/usr/bin:/bin", "LANG=C", "LC_ALL=C",
        "PYTHONNOUSERSITE=1", "PYTHONDONTWRITEBYTECODE=1"};
    std::vector<char *> envp;
    for (auto &word : environment) envp.push_back(word.data());
    envp.push_back(nullptr);
    Spawn spawn;
    Require(posix_spawn_file_actions_addopen(&spawn.actions, 0, "/dev/null", O_RDONLY, 0) == 0 &&
            posix_spawn_file_actions_addopen(&spawn.actions, 1, (work / "worker.log").c_str(),
                                             O_WRONLY | O_CREAT | O_EXCL, 0600) == 0 &&
            posix_spawn_file_actions_adddup2(&spawn.actions, 1, 2) == 0 &&
            posix_spawn_file_actions_addclosefrom_np(&spawn.actions, 3) == 0 &&
            posix_spawnattr_setflags(&spawn.attributes, POSIX_SPAWN_SETPGROUP) == 0 &&
            posix_spawnattr_setpgroup(&spawn.attributes, 0) == 0, "cannot configure worker process");
    pid_t pid = 0;
    const int error = posix_spawn(&pid, argv[0], &spawn.actions, &spawn.attributes, argv.data(), envp.data());
    Require(error == 0, "cannot launch compiler worker: " + std::string(std::strerror(error)));
    const auto deadline = Clock::now() + std::chrono::seconds(config.at("timeout_seconds").get<int>());
    int status = 0;
    for (;;) {
        const auto result = waitpid(pid, &status, WNOHANG);
        if (result == pid) break;
        if (result < 0 && errno == EINTR) continue;
        if (result < 0 || Clock::now() >= deadline) {
            kill(-pid, SIGKILL); // Kill compiler descendants as well as the worker.
            while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
            throw std::runtime_error("compiler worker wait/timeout failure; retained evidence: " + work.string());
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    Require(WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "compiler worker failed; retained evidence: " + work.string());
}
} // namespace

struct FFNJitClient::State {
    JSON config;
    fs::path root;
    explicit State(const std::string &path) : config(json::Read(path)) {
        json::Keys(config, {"schema", "cache", "environment", "paths", "environment_sha256", "timeout_seconds", "optimization"});
        Require(config.at("schema") == "coral.ffn.jit-config.v1", "unknown JIT configuration");
        Require(Hash(config.at("environment")) == Digest(config.at("environment_sha256")),
                "JIT environment digest mismatch");
        const auto &env = config.at("environment");
        Require(env.at("schema") == "coral.compile-environment.v1" &&
                env.at("compatibility") == JSON::parse(artifacts::RuntimeCompatibilityJSON()),
                "JIT environment/runtime compatibility mismatch");
        contracts::FFNCompileSpec tiny;
        tiny.shape = {1, 1, 1};
        Require(env.at("compiler_options") ==
                JSON::parse(artifacts::DescribeFFNJSON(tiny)).at("compiler_options"),
                "JIT compiler options mismatch");
        json::Integer(config.at("timeout_seconds"), 1, 600);
        json::Keys(config.at("optimization"), {"weight_layout", "m_buckets"});
        const auto &buckets = config.at("optimization").at("m_buckets");
        Require(buckets.is_array() && buckets.size() <= 16, "invalid M bucket policy");
        std::int64_t previous = 0;
        for (const auto &value : buckets) {
            const auto capacity = json::Integer(value, 1, 2147483647 / 4);
            Require(capacity > previous, "M buckets must be strictly increasing");
            previous = capacity;
        }
        const auto layout = config.at("optimization").at("weight_layout");
        Require(layout == "native" || layout == "transposed", "unknown JIT weight layout");
        const auto &paths = config.at("paths");
        const auto &files = env.at("files");
        Require(paths.is_object() && files.is_object() && paths.size() == files.size(),
                "incomplete JIT toolchain identity");
        for (const auto *role : {"python", "jit_worker.py", "tool", "checker", "compiler",
                                 "region_service.py", "region_cache.py", "ffn_artifact.py",
                                 "linker", "linker_script", "iree_archive", "crt_archive"}) {
            Require(files.contains(role), std::string("missing JIT tool identity: ") + role);
        }
        // Validate the pinned environment once per client, including the actual
        // compiler shared library. No child process or compiler is invoked here.
        for (auto entry = files.begin(); entry != files.end(); ++entry) {
            const auto file = paths.at(entry.key()).get<std::string>();
            Require(fs::path(file).is_absolute() && HashFile(file) == Digest(entry.value()),
                    "stale JIT configuration: regenerate after a toolchain/service change");
        }
        root = config.at("cache").get<std::string>();
        Require(root.is_absolute() && !fs::is_symlink(root), "invalid JIT cache root");
        fs::create_directories(root);
        for (const auto *name : {"entries", "requests", "staging", "quarantine"}) {
            fs::create_directories(root / name);
            Require(!fs::is_symlink(root / name), "symlink cache directory");
        }
    }

    Artifact Validate(const std::string &key, const contracts::FFNCompileSpec &spec) {
        const auto directory = root / "entries" / key;
        Require(!fs::is_symlink(directory), "symlink cache entry");
        const auto identity = json::Read((directory / "cache.json").string());
        json::Keys(identity, {"schema", "pipeline", "semantic_sha256", "profile",
                              "target", "numeric_policy", "environment"});
        Require(Hash(identity) == key && identity.at("schema") == "coral.ffn.cache.v1" &&
                identity.at("pipeline") == contracts::kFFNPipeline &&
                identity.at("numeric_policy") == contracts::kFFNNumericPolicy &&
                identity.at("environment") == config.at("environment") &&
                identity.at("target") == config.at("environment").at("compatibility"),
                "incompatible JIT cache identity");
        const auto &s = spec.shape;
        const JSON profile{{"m", s.m}, {"d", s.d}, {"f", s.f}, {"dtype", "f32"},
                           {"layout", "row-major"}, {"abi", contracts::FFNAbi(spec)}};
        Require(json::Canonical(identity.at("profile")) == json::Canonical(profile),
                "JIT cache profile mismatch");
        const auto manifest = json::Read((directory / "manifest.json").string());
        const auto &files = config.at("environment").at("files");
        Require(manifest.at("compiler").at("binary_sha256") == files.at("compiler"),
                "JIT compiler provenance mismatch");
        for (const auto *name : {"linker", "linker_script", "iree_archive", "crt_archive"}) {
            Require(manifest.at("compiler").at("support_sha256").at(name) == files.at(name),
                    "JIT compiler support provenance mismatch");
        }
        const auto artifact = artifacts::LoadFFNArtifact(directory.string(), spec);
        Require(artifact.program_sha256 == Digest(identity.at("semantic_sha256")),
                "JIT program digest mismatch");
        return {directory.string(), artifact.program_sha256};
    }
};

FFNJitClient::FFNJitClient(const std::string &configuration)
    : state_(std::make_unique<State>(configuration)) {}
FFNJitClient::~FFNJitClient() = default;

contracts::FFNCompileSpec FFNJitClient::Specification(contracts::FFNShape shape) const {
    contracts::FFNCompileSpec spec;
    spec.shape = shape;
    for (const auto &value : state_->config.at("optimization").at("m_buckets")) {
        const auto capacity = value.get<std::int64_t>();
        if (shape.m <= capacity) {
            spec.shape.m = capacity;
            break;
        }
    }
    if (state_->config.at("optimization").at("weight_layout") == "transposed")
        spec.weight_layout = contracts::FFNWeightLayout::Transposed;
    contracts::ValidateFFNSpec(spec);
    return spec;
}

Artifact FFNJitClient::Resolve(const planner::GGMLFFNPlan &plan, Stats &stats) {
    Elapsed elapsed{stats};
    ++stats.requests;
    const auto spec = Specification(plan.shape);
    const JSON request{{"schema", "coral.ffn.jit-request.v1"},
                       {"environment_sha256", state_->config.at("environment_sha256")},
                       {"snapshot", Snapshot(plan)},
                       {"optimization", state_->config.at("optimization")}};
    const auto request_key = Hash(request);
    std::string invalid;
    try {
        const auto index = json::Read((state_->root / "requests" / (request_key + ".json")).string());
        json::Keys(index, {"schema", "request_sha256", "key"});
        Require(index.at("schema") == "coral.ffn.request-index.v1" &&
                index.at("request_sha256") == request_key, "JIT request index mismatch");
        auto artifact = state_->Validate(Digest(index.at("key")), spec);
        ++stats.disk_hits;
        return artifact;
    } catch (const std::exception &error) {
        invalid = error.what();
    }
    std::string pattern = (state_->root / "staging/worker-XXXXXX").string();
    std::vector<char> name(pattern.begin(), pattern.end());
    name.push_back('\0');
    Require(mkdtemp(name.data()) != nullptr, "cannot create worker evidence directory");
    const fs::path work(name.data());
    Write(work / "request.json", request);
    Write(work / "config.json", state_->config); // Owned snapshot, not a mutable config path.
    Write(work / "lookup.json", {{"reason", invalid}});
    ++stats.worker_launches;
    Launch(state_->config, work);
    const auto response = json::Read((work / "response.json").string());
    json::Keys(response, {"schema", "status", "request_sha256", "key", "state", "recovered", "compile_ms"});
    Require(response.at("schema") == "coral.ffn.jit-response.v1" &&
            response.at("status") == "passed" && response.at("request_sha256") == request_key &&
            (response.at("state") == "compiled" || response.at("state") == "hit") &&
            response.at("recovered").is_boolean(), "invalid compiler worker response");
    auto artifact = state_->Validate(Digest(response.at("key")), spec);
    if (response.at("state") == "compiled") ++stats.compilations;
    else ++stats.disk_hits;
    if (response.at("recovered").get<bool>()) ++stats.recoveries;
    return artifact;
}
} // namespace coral::jit
