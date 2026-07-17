#include "gpu/gpu_execute.hpp"
#include "host/guest_memory_map.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

int main() {
#ifdef _WIN32
    _putenv_s("PROSPER_NO_GUEST_READ_CACHE", "1");
    constexpr size_t page_size = 4096;
    void* page = VirtualAlloc(nullptr, page_size, MEM_RESERVE | MEM_COMMIT, PAGE_NOACCESS);
#else
    setenv("PROSPER_NO_GUEST_READ_CACHE", "1", 1);
    const size_t page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    void* page = mmap(nullptr, page_size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (page == MAP_FAILED) page = nullptr;
#endif
    if (!page) {
        std::puts("FAIL: could not reserve an unreadable test page");
        return 1;
    }

    const uint64_t begin = reinterpret_cast<uint64_t>(page);
    prosper::host::notify_guest_mapping_added(begin, page_size, true);
    const bool readable = prosper::gpu::guest_readable(begin, 1);
    prosper::host::notify_guest_mapping_removed(begin, page_size);

#ifdef _WIN32
    VirtualFree(page, 0, MEM_RELEASE);
#else
    munmap(page, page_size);
#endif

    if (readable) {
        std::puts("FAIL: disabled cache trusted a registry-backed readable range");
        return 1;
    }
    std::puts("PASS: disabled cache used the OS readability probe");
    return 0;
}
