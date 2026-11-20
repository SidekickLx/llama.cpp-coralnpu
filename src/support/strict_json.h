// SPDX-License-Identifier: MIT
#pragma once
#include "nlohmann/json.hpp"
#include <cerrno>
#include <fcntl.h>
#include <functional>
#include <set>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace coral::json {
using Value = nlohmann::json;
inline void Require(bool value, const std::string &message) {
    if (!value) throw std::runtime_error(message);
}
struct UniqueKeys {
    std::vector<std::set<std::string>> objects;
    bool operator()(int depth, Value::parse_event_t event, Value &value) {
        Require(depth <= 32, "JSON nesting limit exceeded");
        if (event == Value::parse_event_t::object_start) objects.emplace_back();
        if (event == Value::parse_event_t::key) {
            Require(!objects.empty() && objects.back().insert(value.get<std::string>()).second,
                    "duplicate JSON field");
        }
        if (event == Value::parse_event_t::object_end) objects.pop_back();
        return true;
    }
};
inline Value Read(const std::string &path, std::size_t limit = 1024 * 1024) {
    const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
    Require(fd >= 0, "cannot open JSON: " + path);
    std::vector<char> bytes;
    try {
        struct stat info{};
        Require(fstat(fd, &info) == 0 && S_ISREG(info.st_mode) && info.st_size > 0 &&
                    static_cast<std::uint64_t>(info.st_size) <= limit, "invalid JSON file: " + path);
        bytes.resize(info.st_size);
        std::size_t offset = 0;
        while (offset < bytes.size()) {
            const auto count = read(fd, bytes.data() + offset, bytes.size() - offset);
            if (count < 0 && errno == EINTR) continue;
            Require(count > 0, "incomplete JSON read");
            offset += count;
        }
    } catch (...) {
        close(fd);
        throw;
    }
    close(fd);
    UniqueKeys keys;
    return Value::parse(bytes.begin(), bytes.end(), std::ref(keys));
}
inline void Keys(const Value &value, std::initializer_list<const char *> names) {
    Require(value.is_object() && value.size() == names.size(), "unexpected JSON object fields");
    for (const auto *name : names) Require(value.contains(name), std::string("missing field: ") + name);
}
inline std::int64_t Integer(const Value &value, std::int64_t low, std::int64_t high) {
    Require(value.is_number_integer() && value >= low && value <= high, "invalid JSON integer");
    return value.get<std::int64_t>();
}
inline std::string Canonical(const Value &value) { return value.dump(-1, ' ', true); }
} // namespace coral::json
