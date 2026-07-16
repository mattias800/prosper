// Guest fd I/O must preserve every byte of binary game content on every host.
// In particular, Windows CRT text mode treats 0x1a as EOF and translates CRLF.
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

using namespace prosper;

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::printf("  [FAIL] %s\n", msg); fails++; } \
                              else        { std::printf("  [ok]   %s\n", msg); } } while (0)

int main() {
    std::printf("== test_file_binary ==\n");
    const char* path = "prosper-test-file-binary.tmp";
    std::array<uint8_t, 512> expected{};
    for (size_t i = 0; i < expected.size(); ++i) expected[i] = (uint8_t)(i * 37u + 11u);
    expected[7] = 0x1a;
    expected[8] = '\r';
    expected[9] = '\n';

    FILE* out = std::fopen(path, "wb");
    CHECK(out != nullptr, "create binary fixture");
    if (out) {
        CHECK(std::fwrite(expected.data(), 1, expected.size(), out) == expected.size(),
              "write binary fixture");
        std::fclose(out);
    }

    register_file_hle();
    HleFn open_fn = Hle::lookup(nid_hash("sceKernelOpen"));
    HleFn read_fn = Hle::lookup(nid_hash("sceKernelRead"));
    HleFn close_fn = Hle::lookup(nid_hash("sceKernelClose"));
    CHECK(open_fn && read_fn && close_fn, "file HLE functions registered");

    std::array<uint8_t, 512> actual{};
    int64_t fd = open_fn ? (int64_t)open_fn((uint64_t)(uintptr_t)path, 0, 0, 0, 0, 0) : -1;
    CHECK(fd >= 0, "open fixture through guest fd HLE");
    int64_t n = fd >= 0 && read_fn
        ? (int64_t)read_fn((uint64_t)fd, (uint64_t)(uintptr_t)actual.data(), actual.size(), 0, 0, 0)
        : -1;
    CHECK(n == (int64_t)expected.size(), "read continues through embedded 0x1a");
    CHECK(n == (int64_t)expected.size() && actual == expected,
          "read preserves binary bytes including CRLF");
    if (fd >= 0 && close_fn) close_fn((uint64_t)fd, 0, 0, 0, 0, 0);

#ifdef _WIN32
    // Windows validates the entire destination range passed to ReadFile before it discovers EOF.
    // Guest libc legitimately asks _read for a large stdio refill even when only a short file remains;
    // if that oversized range crosses the guest allocation's guard page, raw _read fails with EINVAL
    // instead of returning the bytes before EOF. Evergate's 371-byte boot.config exercises this path.
    SYSTEM_INFO si{};
    GetSystemInfo(&si);
    const size_t page = si.dwPageSize;
    auto* guarded = static_cast<uint8_t*>(VirtualAlloc(
        nullptr, page * 2, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    CHECK(guarded != nullptr, "allocate guarded short-read destination");
    if (guarded) {
        DWORD old_protect = 0;
        CHECK(VirtualProtect(guarded + page, page, PAGE_NOACCESS, &old_protect) != 0,
              "protect page immediately after short-read destination");
        uint8_t* tail = guarded + page - expected.size();
        std::memset(tail, 0, expected.size());
        fd = open_fn ? (int64_t)open_fn((uint64_t)(uintptr_t)path, 0, 0, 0, 0, 0) : -1;
        CHECK(fd >= 0, "reopen short fixture through guest fd HLE");
        n = fd >= 0 && read_fn
            ? (int64_t)read_fn((uint64_t)fd, (uint64_t)(uintptr_t)tail, page, 0, 0, 0)
            : -1;
        CHECK(n == (int64_t)expected.size(),
              "oversized read returns short file before guarded page");
        CHECK(n == (int64_t)expected.size() &&
                  std::memcmp(tail, expected.data(), expected.size()) == 0,
              "guarded short read preserves every available byte");
        if (fd >= 0 && close_fn) close_fn((uint64_t)fd, 0, 0, 0, 0, 0);
        VirtualFree(guarded, 0, MEM_RELEASE);
    }

    // Guest virtual/direct memory is frequently reserved and committed on first touch. A host CRT
    // _read writes from kernel context and cannot take prosper's VEH lazy-commit fault, so the file
    // HLE must materialize tracked pages first. Evergate's 2.3 MiB Master.bank read spans many such
    // untouched pages and otherwise fails with EINVAL/EFAULT at 12% loading.
    register_kernel_mem_hle();
    HleFn reserve_fn = Hle::lookup(nid_hash("sceKernelReserveVirtualRange"));
    HleFn unmap_fn = Hle::lookup(nid_hash("sceKernelMunmap"));
    uint64_t reserved = 0;
    CHECK(reserve_fn && unmap_fn, "guest virtual-memory HLE functions registered");
    uint64_t reserve_result = reserve_fn
        ? reserve_fn((uint64_t)(uintptr_t)&reserved, 0x8000, 0, 0x4000, 0, 0)
        : ~uint64_t{0};
    CHECK(reserve_result == 0 && reserved != 0, "reserve untouched guest read destination");
    if (reserved) {
        fd = open_fn ? (int64_t)open_fn((uint64_t)(uintptr_t)path, 0, 0, 0, 0, 0) : -1;
        n = fd >= 0 && read_fn
            ? (int64_t)read_fn((uint64_t)fd, reserved, expected.size(), 0, 0, 0)
            : -1;
        CHECK(n == (int64_t)expected.size(), "read materializes reserved guest pages");
        CHECK(n == (int64_t)expected.size() &&
                  std::memcmp((const void*)(uintptr_t)reserved, expected.data(), expected.size()) == 0,
              "materialized guest destination receives every byte");
        if (fd >= 0 && close_fn) close_fn((uint64_t)fd, 0, 0, 0, 0, 0);
        if (unmap_fn) unmap_fn(reserved, 0x8000, 0, 0, 0, 0);
    }
#endif
    std::remove(path);

    if (fails) { std::printf("== FAIL: %d ==\n", fails); return 1; }
    std::printf("== PASS ==\n");
    return 0;
}
