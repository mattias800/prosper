#include "hle/dispatch.hpp"
#include "hle/nid.hpp"

#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <thread>
#ifdef _WIN32
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
              call_open(open, guard_page) && call_open(open, boundary + page_size - 8);

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
