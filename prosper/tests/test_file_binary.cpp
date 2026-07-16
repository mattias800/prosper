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
    // Unity asks for a whole 64 KiB cache block even when boot.config is only a few hundred bytes.
    // Guest allocators may leave later pages reserved until first touch. Windows _read validates the
    // entire requested destination range before discovering EOF, so a direct host read fails EINVAL;
    // PS5/Linux instead copy the available prefix and return its short byte count.
    void* sparse = VirtualAlloc(nullptr, 0x10000, MEM_RESERVE, PAGE_NOACCESS);
    CHECK(sparse != nullptr, "reserve sparse guest-style read buffer");
    void* committed = sparse ? VirtualAlloc(sparse, 0x4000, MEM_COMMIT, PAGE_READWRITE) : nullptr;
    CHECK(committed == sparse, "commit only the first guest page");
    int64_t sparse_fd = open_fn
        ? (int64_t)open_fn((uint64_t)(uintptr_t)path, 0, 0, 0, 0, 0)
        : -1;
    CHECK(sparse_fd >= 0, "reopen fixture for sparse-buffer short read");
    int64_t sparse_n = sparse_fd >= 0 && read_fn && sparse
        ? (int64_t)read_fn((uint64_t)sparse_fd, (uint64_t)(uintptr_t)sparse, 0x10000, 0, 0, 0)
        : -1;
    CHECK(sparse_n == (int64_t)expected.size(),
          "short read does not probe reserved destination pages past EOF");
    CHECK(sparse_n == (int64_t)expected.size() &&
              std::memcmp(sparse, expected.data(), expected.size()) == 0,
          "short read preserves the available file prefix");
    if (sparse_fd >= 0 && close_fn) close_fn((uint64_t)sparse_fd, 0, 0, 0, 0, 0);
    if (sparse) VirtualFree(sparse, 0, MEM_RELEASE);

    // Tracked guest virtual/direct memory is also committed on first touch. The bounce-buffer copy
    // must take the emulator's lazy-commit path before publishing data into such a destination.
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
