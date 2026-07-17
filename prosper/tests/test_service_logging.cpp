#include "hle/dispatch.hpp"
#include "hle/nid.hpp"

#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <cstring>
#include <thread>
#ifdef _WIN32
#include <io.h>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {
bool call_open(prosper::HleFn open, void* pointer) {
    return open(1, 0xdcdfb7a0, reinterpret_cast<uintptr_t>(pointer), 1, 0x83, 0) == 0;
}

bool logs_full_values(prosper::HleFn open, void* pointer) {
    FILE* capture = std::tmpfile();
    if (!capture) return false;
    if (std::fflush(stderr) != 0) {
        std::fclose(capture);
        return false;
    }

#ifdef _WIN32
    const int stderr_fd = _fileno(stderr);
    const int saved_stderr = _dup(stderr_fd);
    const bool redirected = saved_stderr >= 0 && _dup2(_fileno(capture), stderr_fd) == 0;
#else
    const int stderr_fd = fileno(stderr);
    const int saved_stderr = dup(stderr_fd);
    const bool redirected = saved_stderr >= 0 && dup2(fileno(capture), stderr_fd) >= 0;
#endif
    if (!redirected) {
        if (saved_stderr >= 0) {
#ifdef _WIN32
            _close(saved_stderr);
#else
            close(saved_stderr);
#endif
        }
        std::fclose(capture);
        return false;
    }

    const bool called =
        open(1, 0xdcdfb7a0, reinterpret_cast<uintptr_t>(pointer), 1, 0x83,
             0x1122334455667788ull) == 0;
    const bool flushed = std::fflush(stderr) == 0;
#ifdef _WIN32
    const bool restored = _dup2(saved_stderr, stderr_fd) == 0;
    _close(saved_stderr);
#else
    const bool restored = dup2(saved_stderr, stderr_fd) >= 0;
    close(saved_stderr);
#endif

    char output[512]{};
    std::rewind(capture);
    const size_t bytes = std::fread(output, 1, sizeof(output) - 1, capture);
    std::fclose(capture);
    output[bytes] = '\0';
    const bool full_argument =
        std::strstr(output, ", 0x1122334455667788)") != nullptr;
    const bool full_word =
        std::strstr(output, "[svc]   a2 -> 1122334455667788") != nullptr;
    if (!full_argument || !full_word)
        std::fprintf(stderr, "captured service log:\n%s", output);
    return called && flushed && restored && full_argument && full_word;
}
}

int main() {
#ifdef _WIN32
    _putenv_s("PROSPER_SVCLOG", "1");
#else
    setenv("PROSPER_SVCLOG", "1", 1);
#endif

    prosper::register_builtin_hle();
    auto open = prosper::Hle::lookup("eaFXjfJv3xs"); // sceImeKeyboardOpen
    if (!open) {
        std::fprintf(stderr, "sceImeKeyboardOpen was not registered\n");
        return 1;
    }

#ifdef _WIN32
    constexpr size_t page_size = 0x1000;
    void* reserved = VirtualAlloc(nullptr, page_size, MEM_RESERVE, PAGE_NOACCESS);
    void* protected_page = VirtualAlloc(nullptr, page_size, MEM_RESERVE | MEM_COMMIT, PAGE_NOACCESS);
    auto* boundary = static_cast<uint8_t*>(VirtualAlloc(
        nullptr, page_size * 2, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    if (!reserved || !protected_page || !boundary) return 1;
    *reinterpret_cast<uint64_t*>(boundary + page_size - 8) = 0x1122334455667788ull;
    DWORD old_protect = 0;
    if (!VirtualProtect(boundary + page_size, page_size, PAGE_NOACCESS, &old_protect)) return 1;

    void* guard_page = VirtualAlloc(nullptr, page_size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!guard_page ||
        !VirtualProtect(guard_page, page_size, PAGE_READWRITE | PAGE_GUARD, &old_protect))
        return 1;
#else
    const size_t page_size = (size_t)sysconf(_SC_PAGESIZE);
    void* reserved = mmap(nullptr, page_size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    void* protected_page = mmap(nullptr, page_size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    auto* boundary = static_cast<uint8_t*>(mmap(
        nullptr, page_size * 2, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (reserved == MAP_FAILED || protected_page == MAP_FAILED || boundary == MAP_FAILED) return 1;
    *reinterpret_cast<uint64_t*>(boundary + page_size - 8) = 0x1122334455667788ull;
    if (mprotect(boundary + page_size, page_size, PROT_NONE) != 0) return 1;
    void* guard_page = protected_page;
#endif

    // The remaining literals came from a Blasphemous 2 call. Each address is pointer-shaped but
    // cannot safely be dereferenced for every requested word.
    bool ok = call_open(open, reserved) && call_open(open, protected_page) &&
              call_open(open, guard_page) && logs_full_values(open, boundary + page_size - 8);

    // Exercise the review finding: the mapping can disappear concurrently with the snapshot.
    std::atomic<bool> run{true};
    std::atomic<bool> race_failed{false};
    std::atomic<unsigned> changes{0};
    std::thread toggler([&] {
        while (run.load(std::memory_order_relaxed)) {
#ifdef _WIN32
            const bool unmapped = VirtualFree(boundary, page_size, MEM_DECOMMIT) != FALSE;
            const bool remapped =
                VirtualAlloc(boundary, page_size, MEM_COMMIT, PAGE_READWRITE) == boundary;
#else
            const bool unmapped = mmap(boundary, page_size, PROT_NONE,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0) == boundary;
            const bool remapped = mmap(boundary, page_size, PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0) == boundary;
#endif
            if (!unmapped || !remapped) {
                race_failed.store(true, std::memory_order_release);
                break;
            }
            changes.fetch_add(1, std::memory_order_release);
        }
    });
    while (changes.load(std::memory_order_acquire) == 0 &&
           !race_failed.load(std::memory_order_acquire))
        std::this_thread::yield();
    for (int i = 0; i < 1000 && ok; ++i) ok = call_open(open, boundary);
    run.store(false, std::memory_order_relaxed);
    toggler.join();
    ok = ok && !race_failed.load(std::memory_order_relaxed) &&
         changes.load(std::memory_order_relaxed) > 0;

#ifdef _WIN32
    VirtualFree(reserved, 0, MEM_RELEASE);
    VirtualFree(protected_page, 0, MEM_RELEASE);
    VirtualFree(guard_page, 0, MEM_RELEASE);
    VirtualFree(boundary, 0, MEM_RELEASE);
#else
    munmap(reserved, page_size);
    munmap(protected_page, page_size);
    munmap(boundary, page_size * 2);
#endif

    return ok ? 0 : 1;
}
