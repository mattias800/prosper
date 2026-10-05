// test_guest_read_cache_disabled — with PROSPER_NO_GUEST_READ_CACHE=1, guest_readable() must answer
// from the OS readability probe instead of the registry-backed range cache. The arm is a deliberately
// UNREADABLE mapping that has just been announced to the registry: trusting the cache would report a
// page no code can read as readable, and the first guest store through that answer would fault in a
// place that looks nothing like a bad cache entry.
#include "gpu/execute/gpu_execute.hpp"
#include "host/memory/guest_memory_map.hpp"

#include <gtest/gtest.h>

#include <cstdint>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

TEST(GuestReadCacheDisabled, UnreadableMappingStaysUnreadable) {
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
    ASSERT_NE(page, nullptr) << "could not reserve an unreadable test page";

    const uint64_t begin = reinterpret_cast<uint64_t>(page);
    prosper::host::notify_guest_mapping_added(begin, page_size, true);
    const bool readable = prosper::gpu::guest_readable(begin, 1);
    prosper::host::notify_guest_mapping_removed(begin, page_size);

#ifdef _WIN32
    VirtualFree(page, 0, MEM_RELEASE);
#else
    munmap(page, page_size);
#endif

    EXPECT_FALSE(readable)
        << "disabled cache trusted a registry-backed readable range instead of the OS probe";
}