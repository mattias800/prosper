#include "host/platform/process_replace.hpp"

#include <cerrno>
#include <limits>
#include <thread>
#ifdef _WIN32
#include <cstdlib>
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#ifdef __linux__
#include <sys/syscall.h>
#endif
#ifdef __APPLE__
#include <spawn.h>
extern char** environ;
#endif
#endif

namespace prosper::host {
int process_replace_isolate_descriptors() {
#if defined(__linux__) && defined(SYS_close_range)
    constexpr unsigned kCloseRangeUnshare = 1u << 1;
    constexpr unsigned kCloseRangeCloexec = 1u << 2;
    return ::syscall(SYS_close_range, 3u, std::numeric_limits<unsigned>::max(),
                     kCloseRangeUnshare | kCloseRangeCloexec)
               ? errno
               : 0;
#else
    return ENOTSUP;
#endif
}
#ifdef _WIN32
namespace {
std::string quote_argument(const std::string& arg) {
    std::string quoted = "\"";
    size_t slashes = 0;
    for (const char c : arg) {
        if (c == '\\') {
            ++slashes;
            continue;
        }
        quoted.append(c == '"' ? 2 * slashes + 1 : slashes, '\\');
        slashes = 0;
        quoted += c;
    }
    quoted.append(2 * slashes, '\\');
    return quoted + '"';
}
}   // namespace
#endif
int process_replace_preflight(const std::string& executable) {
    if (executable.empty()) return EINVAL;
#ifdef _WIN32
    DWORD kind = 0;
    if (!GetBinaryTypeA(executable.c_str(), &kind)) return ENOEXEC;
    return kind == SCS_64BIT_BINARY ? 0 : ENOEXEC;
#else
    struct stat info{};
    if (::stat(executable.c_str(), &info)) return errno;
    if (!S_ISREG(info.st_mode)) return EACCES;
    if (::access(executable.c_str(), X_OK)) return errno;
#ifdef __linux__
    // A disposable host thread probes isolation without changing the guest/frontend shared FD
    // table. Unsupported kernels/seccomp policies are rejected before irreversible shutdown.
    int error = 0;
    try {
        std::thread probe([&] { error = process_replace_isolate_descriptors(); });
        probe.join();
    } catch (...) { return EAGAIN; }
    return error;
#else
    return 0;
#endif
#endif
}

int replace_process(const std::vector<std::string>& argv) {
    if (argv.empty()) return EINVAL;
    if (const int error = process_replace_preflight(argv[0])) return error;
#ifdef _WIN32
    std::string command;
    for (const auto& arg : argv) {
        if (!command.empty()) command += ' ';
        command += quote_argument(arg);
    }
    if (command.size() >= 32767) return E2BIG;
    STARTUPINFOA startup{};
    startup.cb = sizeof startup;
    PROCESS_INFORMATION child{};
    // No inherited guest descriptors or host graphics/display handles. Windows has no same-PID
    // exec: create the fresh process successfully, then retire every thread of this one.
    if (!CreateProcessA(argv[0].c_str(), command.data(), nullptr, nullptr, FALSE, 0, nullptr,
                        nullptr, &startup, &child))
        return EIO;
    CloseHandle(child.hThread);
    CloseHandle(child.hProcess);
    std::_Exit(0);
#else
    std::vector<char*> pointers;
    for (const auto& arg : argv) pointers.push_back(const_cast<char*>(arg.c_str()));
    pointers.push_back(nullptr);
#ifdef __APPLE__
    // Darwin's SETEXEC replaces this process and applies close-by-default in the kernel; only
    // explicit stdio file actions survive. No new process or shared-table marking interval.
    posix_spawnattr_t attributes;
    int error = ::posix_spawnattr_init(&attributes);
    if (error) return error;
    posix_spawn_file_actions_t actions;
    error = ::posix_spawn_file_actions_init(&actions);
    if (error) {
        ::posix_spawnattr_destroy(&attributes);
        return error;
    }
    error =
        ::posix_spawnattr_setflags(&attributes, POSIX_SPAWN_SETEXEC | POSIX_SPAWN_CLOEXEC_DEFAULT);
    for (int fd = 0; fd < 3 && !error; ++fd)
        error = ::posix_spawn_file_actions_adddup2(&actions, fd, fd);
    if (!error)
        error =
            ::posix_spawn(nullptr, pointers[0], &actions, &attributes, pointers.data(), environ);
    ::posix_spawn_file_actions_destroy(&actions);
    ::posix_spawnattr_destroy(&attributes);
    return error ? error : EIO;
#else
    // Workers can no longer insert descriptors into this thread's table between CLOEXEC and exec.
    if (const int error = process_replace_isolate_descriptors()) return error;
    ::execv(pointers[0], pointers.data());
#endif
#endif
    return errno ? errno : EIO;
}
} // namespace prosper::host
