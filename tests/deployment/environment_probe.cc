// SPDX-License-Identifier: MIT
// Test-only executable: prove that the deployed filesystem cannot launch any
// compiler entry point. It is not linked into a production runner.
#include "support/json.h"
#include <cerrno>
#include <cstring>
#include <fstream>
#include <ifaddrs.h>
#include <iostream>
#include <linux/capability.h>
#include <net/if.h>
#include <stdexcept>
#include <string>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {
void Require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}
void RejectExecutable(const std::string &path) {
    struct stat state{};
    errno = 0;
    Require(stat(path.c_str(), &state) == -1 && errno == ENOENT,
            "compiler entry point exists inside the runtime root");
    int result_pipe[2];
    Require(pipe(result_pipe) == 0, "pipe failed");
    const auto child = fork();
    Require(child >= 0, "fork failed");
    if (child == 0) {
        close(result_pipe[0]);
        const char *args[] = {path.c_str(), "--version", nullptr};
        execv(path.c_str(), const_cast<char *const *>(args));
        const int failure = errno;
        const auto bytes = write(result_pipe[1], &failure, sizeof(failure));
        _exit(bytes == sizeof(failure) && failure == ENOENT ? 0 : 1);
    }
    close(result_pipe[1]);
    int failure = 0, status = 0;
    const auto bytes = read(result_pipe[0], &failure, sizeof(failure));
    close(result_pipe[0]);
    Require(waitpid(child, &status, 0) == child && WIFEXITED(status) &&
                WEXITSTATUS(status) == 0 && bytes == sizeof(failure) && failure == ENOENT,
            "compiler launch did not fail with ENOENT");
    std::cout << "{\"kind\":\"absent_executable\",\"path\":" << coral::JsonString(path)
              << ",\"stat_errno\":2,\"exec_errno\":2}\n";
}
} // namespace
int main(int argc, char **argv) {
    try {
        Require(argc == 2, "Usage: environment-probe ABSENT_PATHS");
        Require(getuid() == 65534 && geteuid() == 65534 && getgid() == 65534,
                "runtime did not drop root privileges");
        Require(getpid() == 1, "runtime is not PID 1 in a fresh PID namespace");
        Require(prctl(PR_GET_NO_NEW_PRIVS, 0, 0, 0, 0) == 1, "no_new_privs missing");
        __user_cap_header_struct header{};
        header.version = _LINUX_CAPABILITY_VERSION_3;
        __user_cap_data_struct caps[2]{};
        Require(syscall(SYS_capget, &header, caps) == 0, "capability query failed");
        for (const auto &cap : caps) {
            Require(cap.effective == 0 && cap.permitted == 0 && cap.inheritable == 0,
                    "runtime retained capabilities");
        }
        for (const auto *path : {"/proc", "/sys", "/home", "/mnt", "/root"}) {
            struct stat state{};
            errno = 0;
            Require(stat(path, &state) == -1 && errno == ENOENT,
                    "host filesystem entry is reachable");
        }
        ifaddrs *interfaces = nullptr;
        Require(getifaddrs(&interfaces) == 0, "network interface query failed");
        for (const auto *entry = interfaces; entry; entry = entry->ifa_next) {
            Require(!(entry->ifa_flags & IFF_UP), "an active network interface is present");
        }
        freeifaddrs(interfaces);
        std::ifstream paths(argv[1]);
        Require(paths.good(), "cannot open compiler control list");
        std::string path;
        std::size_t count = 0;
        while (std::getline(paths, path)) {
            Require(!path.empty() && path[0] == '/', "invalid control path");
            RejectExecutable(path);
            ++count;
        }
        Require(paths.eof() && count >= 5, "incomplete compiler absence checks");
        std::cout << "{\"status\":\"passed\",\"uid\":" << getuid()
                  << ",\"pid\":" << getpid() << ",\"capabilities\":0,\"no_new_privs\":true"
                  << ",\"active_network_interfaces\":0,\"absent_executables\":" << count << "}\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "environment probe: " << error.what() << '\n';
        return 1;
    }
}
