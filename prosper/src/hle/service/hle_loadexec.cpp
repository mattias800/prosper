// LoadExec replaces the native guest process, rather than reporting success to its launcher while
// continuing to execute the old image. Own traces supply executable paths and complete argv; the
// two-argument ABI is confirmed at both launcher call sites and by the PS5 3.20 JoBqSQt1yyA NID.
#include "hle/service/hle_loadexec.hpp"

#include "hle/dispatch/dispatch.hpp"
#include "host/image/guest_executable.hpp"
#include "host/memory/guest_memory_copy.hpp"
#include "self/module.hpp"

#include <array>
#include <algorithm>
#include <cstdio>
#include <limits>
#include <mutex>

namespace prosper {
namespace {
// CONFIDENCE: MED — existing SystemService family parameter/internal encodings; the launchers
// print the result generically. Exact LoadExec-specific errno translation is not measured.
constexpr uint64_t kParameter = 0x80A10003;
constexpr uint64_t kInternal = 0x80A10001;
// CONFIDENCE: LOW — conservative marshalling caps, not measured SDK limits. The observed requests
// have 13 short arguments. Refuse oversized input explicitly rather than truncating or splitting it.
constexpr size_t kPathBytes = 1024;
constexpr size_t kArgumentBytes = 65536;
constexpr size_t kArgumentCount = 256;
std::mutex backend_mutex;
std::string app0;
LoadExecBackend replacement;

bool read_string(uint64_t address, size_t limit, std::string& result) {
    result.clear();
    std::array<char, 256> bytes;
    while (result.size() < limit) {
        const size_t count = std::min(bytes.size(), limit - result.size());
        if (address > std::numeric_limits<uint64_t>::max() - result.size()) return false;
        const size_t copied = host::guest_read_prefix(address + result.size(), bytes.data(), count);
        for (size_t i = 0; i < copied; ++i) {
            if (!bytes[i]) return true;
            result.push_back(bytes[i]);
        }
        if (copied < count) return false;
    }
    return false;
}

static PROSPER_SYSV_ABI uint64_t loadexec(uint64_t path, uint64_t argv, uint64_t, uint64_t,
                                          uint64_t, uint64_t) {
    LoadExecRequest request;
    LoadExecBackend backend;
    {
        std::lock_guard lock(backend_mutex);
        request.dump_root = app0;
        backend = replacement;
    }
    if (!read_string(path, kPathBytes, request.executable) || request.executable.empty())
        return kParameter;
    size_t budget = kArgumentBytes;
    if (argv) {
        bool terminated = false;
        for (size_t i = 0; i <= kArgumentCount; ++i) {
            uint64_t word = 0;
            if (argv > std::numeric_limits<uint64_t>::max() - i * sizeof(word) ||
                !host::guest_read_exact(argv + i * sizeof(word), &word, sizeof(word)))
                return kParameter;
            if (!word) {
                terminated = true;
                break;
            }
            if (i == kArgumentCount || !budget) return kParameter;
            std::string value;
            if (!read_string(word, budget, value)) return kParameter;
            budget -= value.size() + 1;
            request.argv.push_back(std::move(value));
        }
        if (!terminated) return kParameter;
    }
    std::string error;
    request.host_executable =
        resolve_guest_executable(request.dump_root, request.executable, &error);
    if (request.host_executable.empty()) {
        std::fprintf(stderr, "[loadexec] refused %s: %s\n", request.executable.c_str(),
                     error.c_str());
        return kParameter;
    }
    const auto image = Module::load(request.host_executable, &error);
    const bool entry_valid =
        image && std::any_of(image->loads.begin(), image->loads.end(), [&](const Segment& segment) {
            return segment.type == PT_LOAD && (segment.flags & 1) &&
                   image->e_entry >= segment.vaddr &&
                   image->e_entry - segment.vaddr < segment.filesz;
        });
    if (!image || image->e_type != ET_SCE_DYNEXEC || image->e_machine != 0x3e || !entry_valid) {
        std::fprintf(stderr, "[loadexec] refused %s: not a supported guest executable (%s)\n",
                     request.executable.c_str(), error.c_str());
        return kParameter;
    }
    std::replace(request.executable.begin(), request.executable.end(), '\\', '/');
    if (request.executable.front() != '/') request.executable = "/app0/" + request.executable;
    if (request.argv.empty()) request.argv.push_back(request.executable);
    if (!backend) {
        std::fprintf(stderr, "[loadexec] process replacement backend unavailable\n");
        return kInternal;
    }
    const uint64_t result = backend(request);
    if (!result)
        std::fprintf(stderr, "[loadexec] backend returned without replacing the process\n");
    return result ? result : kInternal;
}
}   // namespace

void set_loadexec_backend(LoadExecBackend backend) {
    std::lock_guard lock(backend_mutex);
    replacement = std::move(backend);
}
void configure_loadexec_app0(const std::string& dump_root) {
    std::lock_guard lock(backend_mutex);
    app0 = dump_root;
}
void register_loadexec_hle() {
    Hle::register_fn("JoBqSQt1yyA", loadexec, "sceSystemServiceLoadExec");
}
}   // namespace prosper
