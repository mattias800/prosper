// Guest fd I/O must preserve every byte of binary game content on every host.
// In particular, Windows CRT text mode treats 0x1a as EOF and translates CRLF.
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
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
    HleFn lseek_fn = Hle::lookup(nid_hash("sceKernelLseek"));
    HleFn close_fn = Hle::lookup(nid_hash("sceKernelClose"));
    HleFn dup_fn = Hle::lookup(nid_hash("dup"));
    HleFn kernel_dup_fn = Hle::lookup(nid_hash("sceKernelDup"));
    HleFn dup2_fn = Hle::lookup(nid_hash("dup2"));
    HleFn kernel_dup2_fn = Hle::lookup(nid_hash("sceKernelDup2"));
    HleFn mkdir_fn = Hle::lookup(nid_hash("mkdir"));
    HleFn kernel_mkdir_fn = Hle::lookup(nid_hash("sceKernelMkdir"));
    CHECK(open_fn && read_fn && lseek_fn && close_fn && dup_fn && kernel_dup_fn &&
              dup2_fn && kernel_dup2_fn && mkdir_fn && kernel_mkdir_fn,
          "file HLE functions registered");

    // Creating an existing directory is a failure, not an idempotent success. Linux previously
    // suppressed EEXIST while Windows propagated it, so save-directory control flow differed by host.
    const char* dir_path = "prosper-test-mkdir-eexist.tmp";
    std::error_code remove_error;
    std::filesystem::remove_all(dir_path, remove_error);
    int64_t mkdir_first = mkdir_fn
        ? (int64_t)mkdir_fn((uint64_t)(uintptr_t)dir_path, 0777, 0, 0, 0, 0)
        : -1;
    errno = 0;
    int64_t mkdir_duplicate = mkdir_fn
        ? (int64_t)mkdir_fn((uint64_t)(uintptr_t)dir_path, 0777, 0, 0, 0, 0)
        : 0;
    int duplicate_errno = errno;
    int64_t kernel_mkdir_duplicate = kernel_mkdir_fn
        ? (int64_t)kernel_mkdir_fn((uint64_t)(uintptr_t)dir_path, 0777, 0, 0, 0, 0)
        : 0;
    CHECK(mkdir_first == 0, "mkdir creates a new directory");
    CHECK(mkdir_duplicate == -1, "mkdir reports an existing directory as failure");
    CHECK(duplicate_errno == EEXIST, "mkdir preserves EEXIST for the caller");
    CHECK(kernel_mkdir_duplicate != 0, "sceKernelMkdir does not report false success on EEXIST");
    std::filesystem::remove_all(dir_path, remove_error);

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

    // A duplicate is a distinct descriptor for the same open file description: it shares the
    // current offset and remains usable after the original descriptor is closed.
    int64_t dup_source = open_fn
        ? (int64_t)open_fn((uint64_t)(uintptr_t)path, 0, 0, 0, 0, 0)
        : -1;
    int64_t duplicated = dup_source >= 0 && kernel_dup_fn
        ? (int64_t)kernel_dup_fn((uint64_t)dup_source, 0, 0, 0, 0, 0)
        : -1;
    CHECK(duplicated >= 3 && duplicated != dup_source,
          "sceKernelDup returns a distinct non-stdio descriptor");
    std::array<uint8_t, 16> first_chunk{};
    std::array<uint8_t, 16> second_chunk{};
    int64_t first_n = dup_source >= 0 && read_fn
        ? (int64_t)read_fn((uint64_t)dup_source,
                           (uint64_t)(uintptr_t)first_chunk.data(), first_chunk.size(), 0, 0, 0)
        : -1;
    int64_t second_n = duplicated >= 0 && read_fn
        ? (int64_t)read_fn((uint64_t)duplicated,
                           (uint64_t)(uintptr_t)second_chunk.data(), second_chunk.size(), 0, 0, 0)
        : -1;
    CHECK(first_n == (int64_t)first_chunk.size() &&
              std::memcmp(first_chunk.data(), expected.data(), first_chunk.size()) == 0,
          "original descriptor reads the first chunk");
    CHECK(second_n == (int64_t)second_chunk.size() &&
              std::memcmp(second_chunk.data(), expected.data() + first_chunk.size(),
                          second_chunk.size()) == 0,
          "sceKernelDup shares the original file offset");
    if (dup_source >= 0 && close_fn) close_fn((uint64_t)dup_source, 0, 0, 0, 0, 0);
    int64_t duplicate_seek = duplicated >= 0 && lseek_fn
        ? (int64_t)lseek_fn((uint64_t)duplicated, 0, SEEK_SET, 0, 0, 0)
        : -1;
    CHECK(duplicate_seek == 0, "duplicate remains usable after closing the original");
    if (duplicated >= 0 && close_fn) close_fn((uint64_t)duplicated, 0, 0, 0, 0, 0);

#ifdef _WIN32
    // Force fd 0 free while duplicating an already-open guest file. _dup chooses the lowest free
    // CRT descriptor, so the wrapper must hold that temporary result and return one above stdio.
    int saved_stdin = ::_dup(0);
    int stdin_filler = -1;
    if (saved_stdin < 0) stdin_filler = ::_open("NUL", _O_RDONLY | _O_BINARY);
    CHECK(saved_stdin >= 0 || stdin_filler == 0, "occupy fd 0 before low-slot dup test");
    int64_t low_slot_source = open_fn
        ? (int64_t)open_fn((uint64_t)(uintptr_t)path, 0, 0, 0, 0, 0)
        : -1;
    CHECK(low_slot_source >= 3, "open dup source outside the stdio range");
    CHECK(::_close(0) == 0, "free fd 0 for deterministic dup reuse");
    int64_t low_slot_duplicate = low_slot_source >= 0 && kernel_dup_fn
        ? (int64_t)kernel_dup_fn((uint64_t)low_slot_source, 0, 0, 0, 0, 0)
        : -1;
    CHECK(low_slot_duplicate >= 3,
          "sceKernelDup lifts a recycled Windows stdio descriptor");
    if (low_slot_duplicate >= 0 && close_fn)
        close_fn((uint64_t)low_slot_duplicate, 0, 0, 0, 0, 0);
    if (low_slot_source >= 0 && close_fn)
        close_fn((uint64_t)low_slot_source, 0, 0, 0, 0, 0);

    // With fd 0 still free, the Windows CRT will allocate it for the next open. The HLE must move
    // that live file into the guest-visible range before returning it, without losing its contents.
    int64_t low_slot_open = open_fn
        ? (int64_t)open_fn((uint64_t)(uintptr_t)path, 0, 0, 0, 0, 0)
        : -1;
    CHECK(low_slot_open >= 3, "sceKernelOpen lifts a recycled Windows stdio descriptor");
    std::array<uint8_t, 16> low_slot_open_chunk{};
    int64_t low_slot_open_n = low_slot_open >= 0 && read_fn
        ? (int64_t)read_fn((uint64_t)low_slot_open,
                           (uint64_t)(uintptr_t)low_slot_open_chunk.data(),
                           low_slot_open_chunk.size(), 0, 0, 0)
        : -1;
    CHECK(low_slot_open_n == (int64_t)low_slot_open_chunk.size() &&
              std::memcmp(low_slot_open_chunk.data(), expected.data(),
                          low_slot_open_chunk.size()) == 0,
          "lifted sceKernelOpen descriptor reads the original file");
    if (low_slot_open >= 3 && close_fn)
        close_fn((uint64_t)low_slot_open, 0, 0, 0, 0, 0);
    else if (low_slot_open >= 0)
        ::_close((int)low_slot_open);
    if (saved_stdin >= 0) {
        CHECK(::_dup2(saved_stdin, 0) == 0, "restore fd 0 after low-slot descriptor tests");
        ::_close(saved_stdin);
    }
#endif

    // dup2 must replace an already-open target, share the source offset, and return the target
    // descriptor. The Windows CRT returns zero on success, so this catches a missing ABI translation.
    int64_t dup2_source = open_fn
        ? (int64_t)open_fn((uint64_t)(uintptr_t)path, 0, 0, 0, 0, 0)
        : -1;
    int64_t dup2_target = open_fn
        ? (int64_t)open_fn((uint64_t)(uintptr_t)path, 0, 0, 0, 0, 0)
        : -1;
    constexpr int64_t source_offset = 73;
    constexpr int64_t old_target_offset = 211;
    int64_t source_seek = dup2_source >= 0 && lseek_fn
        ? (int64_t)lseek_fn((uint64_t)dup2_source, source_offset, SEEK_SET, 0, 0, 0)
        : -1;
    int64_t target_seek = dup2_target >= 0 && lseek_fn
        ? (int64_t)lseek_fn((uint64_t)dup2_target, old_target_offset, SEEK_SET, 0, 0, 0)
        : -1;
    int64_t dup2_result = dup2_source >= 0 && dup2_target >= 0 && kernel_dup2_fn
        ? (int64_t)kernel_dup2_fn((uint64_t)dup2_source, (uint64_t)dup2_target, 0, 0, 0, 0)
        : -1;
    CHECK(source_seek == source_offset && target_seek == old_target_offset,
          "position dup2 source and old target independently");
    CHECK(dup2_result == dup2_target, "sceKernelDup2 returns the replaced target descriptor");
    if (dup2_source >= 0 && close_fn) close_fn((uint64_t)dup2_source, 0, 0, 0, 0, 0);
    std::array<uint8_t, 16> dup2_chunk{};
    int64_t dup2_n = dup2_target >= 0 && read_fn
        ? (int64_t)read_fn((uint64_t)dup2_target,
                           (uint64_t)(uintptr_t)dup2_chunk.data(), dup2_chunk.size(), 0, 0, 0)
        : -1;
    CHECK(dup2_n == (int64_t)dup2_chunk.size() &&
              std::memcmp(dup2_chunk.data(), expected.data() + source_offset,
                          dup2_chunk.size()) == 0,
          "sceKernelDup2 replaces the target and shares the source offset");
    if (dup2_target >= 0 && close_fn) close_fn((uint64_t)dup2_target, 0, 0, 0, 0, 0);

#ifdef _WIN32
    // UCRT reports invalid descriptors through its invalid-parameter handler before returning -1.
    // Guest inputs must not reach the default terminating handler, and a failed dup2 must not close
    // or replace its valid target.
    int64_t invalid_duplicate = kernel_dup_fn
        ? (int64_t)kernel_dup_fn(~uint64_t{0}, 0, 0, 0, 0, 0)
        : 0;
    CHECK(invalid_duplicate == -1, "sceKernelDup safely rejects an invalid source descriptor");
    int64_t preserved_target = open_fn
        ? (int64_t)open_fn((uint64_t)(uintptr_t)path, 0, 0, 0, 0, 0)
        : -1;
    constexpr int64_t preserved_offset = 137;
    int64_t preserved_seek = preserved_target >= 0 && lseek_fn
        ? (int64_t)lseek_fn((uint64_t)preserved_target, preserved_offset, SEEK_SET, 0, 0, 0)
        : -1;
    int64_t invalid_dup2 = preserved_target >= 0 && kernel_dup2_fn
        ? (int64_t)kernel_dup2_fn(~uint64_t{0}, (uint64_t)preserved_target, 0, 0, 0, 0)
        : 0;
    int64_t preserved_position = preserved_target >= 0 && lseek_fn
        ? (int64_t)lseek_fn((uint64_t)preserved_target, 0, SEEK_CUR, 0, 0, 0)
        : -1;
    std::array<uint8_t, 16> preserved_chunk{};
    int64_t preserved_n = preserved_target >= 0 && read_fn
        ? (int64_t)read_fn((uint64_t)preserved_target,
                           (uint64_t)(uintptr_t)preserved_chunk.data(), preserved_chunk.size(),
                           0, 0, 0)
        : -1;
    CHECK(preserved_seek == preserved_offset, "position the invalid-dup2 target");
    CHECK(invalid_dup2 == -1, "sceKernelDup2 safely rejects an invalid source descriptor");
    CHECK(preserved_position == preserved_offset &&
              preserved_n == (int64_t)preserved_chunk.size() &&
              std::memcmp(preserved_chunk.data(), expected.data() + preserved_offset,
                          preserved_chunk.size()) == 0,
          "failed sceKernelDup2 leaves the target descriptor unchanged");
    if (preserved_target >= 0 && close_fn)
        close_fn((uint64_t)preserved_target, 0, 0, 0, 0, 0);
#endif

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

    // Destination validation must happen before a sequential read advances the shared file offset.
    // Repeating the reviewer's invalid-first-chunk probe verifies both EFAULT and retry position.
    void* invalid_first = VirtualAlloc(nullptr, 0x10000, MEM_RESERVE, PAGE_NOACCESS);
    CHECK(invalid_first != nullptr, "reserve inaccessible invalid-read destination");
    int64_t invalid_fd = open_fn
        ? (int64_t)open_fn((uint64_t)(uintptr_t)path, 0, 0, 0, 0, 0)
        : -1;
    int64_t invalid_n = invalid_fd >= 0 && read_fn && invalid_first
        ? (int64_t)read_fn((uint64_t)invalid_fd, (uint64_t)(uintptr_t)invalid_first,
                           100, 0, 0, 0)
        : 0;
    int64_t invalid_pos = invalid_fd >= 0 && lseek_fn
        ? (int64_t)lseek_fn((uint64_t)invalid_fd, 0, SEEK_CUR, 0, 0, 0)
        : -1;
    CHECK(invalid_n == -1, "invalid first destination chunk returns EFAULT");
    CHECK(invalid_pos == 0, "failed first destination chunk consumes no file bytes");
    if (invalid_fd >= 0 && close_fn) close_fn((uint64_t)invalid_fd, 0, 0, 0, 0, 0);
    if (invalid_first) VirtualFree(invalid_first, 0, MEM_RELEASE);

    // The same rule applies after earlier chunks were delivered. The fixture is one full bounce
    // chunk plus 512 bytes; only the committed first chunk may be consumed/reported.
    const char* large_path = "prosper-test-file-binary-large.tmp";
    FILE* large_out = std::fopen(large_path, "wb");
    bool large_written = large_out != nullptr;
    if (large_out) {
        for (int block = 0; block < 129 && large_written; ++block)
            large_written = std::fwrite(expected.data(), 1, expected.size(), large_out) ==
                            expected.size();
        std::fclose(large_out);
    }
    CHECK(large_written, "create partial-copy offset fixture");
    void* partial = VirtualAlloc(nullptr, 0x20000, MEM_RESERVE, PAGE_NOACCESS);
    void* partial_committed = partial
        ? VirtualAlloc(partial, 0x10000, MEM_COMMIT, PAGE_READWRITE) : nullptr;
    CHECK(partial && partial_committed == partial,
          "commit only first chunk of partial-copy destination");
    int64_t partial_fd = open_fn && large_written
        ? (int64_t)open_fn((uint64_t)(uintptr_t)large_path, 0, 0, 0, 0, 0)
        : -1;
    int64_t partial_n = partial_fd >= 0 && read_fn && partial
        ? (int64_t)read_fn((uint64_t)partial_fd, (uint64_t)(uintptr_t)partial,
                           0x20000, 0, 0, 0)
        : -1;
    int64_t partial_pos = partial_fd >= 0 && lseek_fn
        ? (int64_t)lseek_fn((uint64_t)partial_fd, 0, SEEK_CUR, 0, 0, 0)
        : -1;
    CHECK(partial_n == 0x10000, "partial read reports only the delivered first chunk");
    CHECK(partial_pos == 0x10000, "failed later destination chunk consumes no extra bytes");
    CHECK(partial_n == 0x10000 &&
              std::memcmp(partial, expected.data(), expected.size()) == 0,
          "partial read preserves delivered prefix bytes");
    if (partial_fd >= 0 && close_fn) close_fn((uint64_t)partial_fd, 0, 0, 0, 0, 0);
    if (partial) VirtualFree(partial, 0, MEM_RELEASE);
    std::remove(large_path);

    // A read whose current position is already past EOF must return zero without validating or
    // touching any byte of the requested destination range. Keep the entire range reserved and
    // inaccessible so a host read that probes it reproduces the Windows CRT failure deterministically.
    void* past_eof = VirtualAlloc(nullptr, 0x10000, MEM_RESERVE, PAGE_NOACCESS);
    CHECK(past_eof != nullptr, "reserve inaccessible past-EOF destination");
    int64_t past_eof_fd = open_fn
        ? (int64_t)open_fn((uint64_t)(uintptr_t)path, 0, 0, 0, 0, 0)
        : -1;
    constexpr int64_t past_eof_offset = 4096;
    int64_t seek_result = past_eof_fd >= 0 && lseek_fn
        ? (int64_t)lseek_fn((uint64_t)past_eof_fd, (uint64_t)past_eof_offset,
                            SEEK_SET, 0, 0, 0)
        : -1;
    CHECK(seek_result == past_eof_offset, "seek beyond fixture EOF");
    int64_t past_eof_n = past_eof_fd >= 0 && read_fn && past_eof
        ? (int64_t)read_fn((uint64_t)past_eof_fd, (uint64_t)(uintptr_t)past_eof,
                           0x10000, 0, 0, 0)
        : -1;
    CHECK(past_eof_n == 0, "read past EOF returns zero without probing destination");
    MEMORY_BASIC_INFORMATION past_eof_info{};
    CHECK(past_eof && VirtualQuery(past_eof, &past_eof_info, sizeof past_eof_info) &&
              past_eof_info.State == MEM_RESERVE,
          "past-EOF destination remains uncommitted and untouched");
    if (past_eof_fd >= 0 && close_fn) close_fn((uint64_t)past_eof_fd, 0, 0, 0, 0, 0);
    if (past_eof) VirtualFree(past_eof, 0, MEM_RELEASE);

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
