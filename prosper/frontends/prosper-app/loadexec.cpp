#include "loadexec.hpp"
#include "hle/service/hle_loadexec.hpp"
#include "host/image/guest_executable.hpp"
#include "host/platform/lifecycle.hpp"
#include "host/platform/process_replace.hpp"

#include <charconv>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

namespace prosper::frontend {
namespace {
std::mutex pending_mutex;
std::optional<std::vector<std::string>> pending_command;
std::string selected_executable = "/app0/eboot.bin";
constexpr uint64_t kInternal = 0x80A10001;

bool read_count(const char* text, size_t& count) {
    const char* end = text + std::strlen(text);
    const auto parsed = std::from_chars(text, end, count);
    return parsed.ec == std::errc{} && parsed.ptr == end && count > 0 && count <= 256;
}
}   // namespace

bool parse_loadexec_option(int& i, int argc, char** argv, std::string* error) {
    const std::string flag = argv[i];
    const auto fail = [&] {
        if (error) *error = "invalid executable/argv process-replacement option";
        return false;
    };
    if (flag == "--guest-executable") {
        if (++i >= argc || !*argv[i]) return fail();
        selected_executable = argv[i];
        return true;
    }
    if (flag != "--guest-argv" || ++i >= argc) return fail();
    size_t count = 0;
    if (!read_count(argv[i], count) || count > static_cast<size_t>(argc - i - 1)) return fail();
    std::vector<std::string> args;
    for (size_t n = 0; n < count; ++n) args.emplace_back(argv[++i]);
    set_guest_entry_arguments(args);
    return true;
}

std::string selected_guest_executable() {
    return selected_executable;
}

std::vector<std::string> loadexec_host_arguments(int argc, char** argv) {
    std::vector<std::string> result;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--guest-executable") {
            if (++i >= argc) return {};
        } else if (flag == "--guest-argv") {
            size_t count = 0;
            if (++i >= argc || !read_count(argv[i], count) ||
                count > static_cast<size_t>(argc - i - 1))
                return {};
            i += static_cast<int>(count);
        } else
            result.push_back(flag);
    }
    return result;
}

LoadExecBackend make_loadexec_frontend_backend(int argc, char** argv,
                                               const std::string& executable) {
    std::vector<std::string> original = loadexec_host_arguments(argc, argv);
    const std::thread::id frontend_thread = std::this_thread::get_id();
    return [original = std::move(original), executable,
            frontend_thread](const LoadExecRequest& request) {
        // Module init runs synchronously on the frontend before its loop exists. Parking that
        // thread prevents request consumption. Explicit lifecycle cap until bootstrap can hand off:
        // return an error with the old guest intact, never queue a request that cannot finish.
        if (std::this_thread::get_id() == frontend_thread) {
            std::fprintf(stderr,
                         "[loadexec] replacement from frontend/bootstrap thread unsupported\n");
            return kInternal;
        }
        // Check the native launch target before requesting irreversible guest shutdown. There is
        // still a replacement race (permissions/file contents can change afterwards); that residual
        // failure is fatal and logged, never advertised as a resumable old-guest error or SCE_OK.
        if (host::process_replace_preflight(executable)) return kInternal;
        std::vector<std::string> command{executable};
        command.insert(command.end(), original.begin(), original.end());
        command.insert(command.end(),
                       {"--dump", request.dump_root, "--guest-executable", request.executable,
                        "--guest-argv", std::to_string(request.argv.size())});
        command.insert(command.end(), request.argv.begin(), request.argv.end());
        {
            std::lock_guard lock(pending_mutex);
            if (pending_command) return kInternal;
            pending_command = std::move(command);
        }
        std::fprintf(stderr, "[loadexec] replacing guest with %s (%zu arguments)\n",
                     request.executable.c_str(), request.argv.size());
        prosper_request_stop();
        // Native guest threads cannot be reset in-process. Do not resume this thread in the old
        // image while the frontend drains GPU work and exec replaces every native thread/mapping.
        std::mutex park_mutex;
        std::condition_variable park;
        std::unique_lock lock(park_mutex);
        park.wait(lock, [] { return false; });
        return kInternal;
    };
}

void install_loadexec_frontend(int argc, char** argv, const std::string& executable) {
    set_loadexec_backend(make_loadexec_frontend_backend(argc, argv, executable));
}

void finish_loadexec_replacement() {
    std::optional<std::vector<std::string>> command;
    {
        std::lock_guard lock(pending_mutex);
        command = pending_command;
    }
    if (!command) return;
    std::fflush(nullptr);
    const int error = host::replace_process(*command);
    std::fprintf(stderr, "[loadexec] native process replacement failed after shutdown: %s\n",
                 std::strerror(error));
    std::fflush(nullptr);
    std::_Exit(1);
}
}   // namespace prosper::frontend
