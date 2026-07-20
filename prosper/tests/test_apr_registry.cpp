// test_apr_registry — guards the APR container registry and its size-keyed read matching
// (issue #62). The APR read-submit handler identifies WHICH file to read by total byte size,
// because the captured read-request object carries no legible file id. That key is only sound
// when unambiguous, so this locks in: stable 1-based id assignment, path-dedup on re-resolve
// (a duplicate entry would make every read of that file look ambiguous), and the match counts
// the read path uses to refuse ambiguous / unplaceable (chunk) reads instead of guessing.
#include <cstdio>
#include <cstdint>
#include <array>
#include <cstring>
#include <string>
#include <vector>
#include "../src/host/exec_image.hpp"
#include "../src/hle/dispatch.hpp"

namespace prosper {
    uint32_t    prosper_apr_register(const std::string& path, uint64_t size);
    void        prosper_apr_reset_for_test();
    int         prosper_apr_match_by_size(uint64_t size, std::string* out_path);
    std::string prosper_apr_path_for_id(uint32_t id);
}
using namespace prosper;

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } \
                              else        { printf("  [ok]   %s\n", msg); } } while (0)

int main() {
    printf("== test_apr_registry ==\n");
    prosper_apr_reset_for_test();

    // Stable 1-based ids in resolve order.
    uint32_t a = prosper_apr_register("/dump/global.utoc", 645);
    uint32_t b = prosper_apr_register("/dump/global.ucas", 100000);
    CHECK(a == 1 && b == 2, "ids assigned 1-based in resolve order");
    CHECK(prosper_apr_path_for_id(1) == "/dump/global.utoc", "path_for_id(1)");
    CHECK(prosper_apr_path_for_id(2) == "/dump/global.ucas", "path_for_id(2)");
    CHECK(prosper_apr_path_for_id(0).empty() && prosper_apr_path_for_id(3).empty(),
          "out-of-range ids resolve to empty");

    // Unambiguous size keying (the only case the read path may serve).
    std::string p;
    CHECK(prosper_apr_match_by_size(645, &p) == 1 && p == "/dump/global.utoc",
          "unique size matches exactly one container");

    // A count that equals no container's TOTAL size (i.e. a partial/chunk read) must not match.
    CHECK(prosper_apr_match_by_size(644, nullptr) == 0, "chunk-sized count matches nothing");

    // Two containers sharing a size: the read path must see the ambiguity (count > 1).
    uint32_t c = prosper_apr_register("/dump/pakchunk0-ps5.utoc", 645);
    CHECK(c == 3, "distinct path gets a new id even at a colliding size");
    CHECK(prosper_apr_match_by_size(645, &p) == 2, "size collision reported as 2 matches");

    // Re-resolving the same host path must return the SAME id (no duplicate entry), so a game
    // that resolves a container twice doesn't turn every read of it ambiguous.
    uint32_t a2 = prosper_apr_register("/dump/global.utoc", 645);
    CHECK(a2 == a, "re-resolving a path returns its existing id");
    CHECK(prosper_apr_match_by_size(645, &p) == 2, "re-resolve does not add a duplicate entry");

    // Re-resolve with a changed size updates the record (single source of truth per path).
    prosper_apr_register("/dump/global.ucas", 200000);
    CHECK(prosper_apr_match_by_size(100000, nullptr) == 0 &&
          prosper_apr_match_by_size(200000, &p) == 1 && p == "/dump/global.ucas",
          "re-resolve updates the stored size");

    prosper_apr_reset_for_test();
    CHECK(prosper_apr_match_by_size(645, nullptr) == 0 && prosper_apr_path_for_id(1).empty(),
          "reset clears the registry");

    // The AMPR builder is a DMA-style read: callers may consume their destination buffer directly,
    // not only the completion record's data pointer. Evergate loads globalgamemanagers this way.
    const char* fixture_path = "prosper-test-apr-read.tmp";
    std::array<uint8_t, 257> expected{};
    for (size_t i = 0; i < expected.size(); ++i) expected[i] = (uint8_t)(i * 29u + 7u);
    FILE* fixture = std::fopen(fixture_path, "wb");
    CHECK(fixture != nullptr, "create APR read fixture");
    if (fixture) {
        CHECK(std::fwrite(expected.data(), 1, expected.size(), fixture) == expected.size(),
              "write APR read fixture");
        std::fclose(fixture);
    }
    register_file_hle();
    HleFn read_file = Hle::lookup("mQ16-QdKv7k");
    CHECK(read_file != nullptr, "AMPR read-file HLE registered");
    std::string stub_error;
    const std::vector<ImportSlot> slots = {{"libSceAmpr", "mQ16-QdKv7k"}};
    CHECK(install_stubs(slots, 0x720000000ull, 96, &stub_error),
          "generated executable AMPR import stub");
    // sysv_abi is the default on Linux (a no-op there) and forces the guest SysV ABI on MinGW.
    using GuestReadFile = uint64_t (__attribute__((sysv_abi)) *)(
        uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
        uint64_t, uint64_t, uint64_t, uint64_t);
    auto read_file_guest = reinterpret_cast<GuestReadFile>(
        static_cast<uintptr_t>(stub_addr(0)));
    uint32_t fixture_id = prosper_apr_register(fixture_path, expected.size());
    std::array<uint8_t, 0x48> request{};
    std::array<uint64_t, 3> completion{};
    constexpr size_t read_offset = 41;
    constexpr size_t read_size = 73;
    std::array<uint8_t, read_size> destination{};
    uint64_t result = read_file && stub_error.empty()
        ? read_file_guest((uint64_t)(uintptr_t)request.data(), 0,
                          (uint64_t)(uintptr_t)completion.data(), fixture_id,
                          (uint64_t)(uintptr_t)destination.data(), destination.size(),
                          read_offset, 0, 0)
        : ~uint64_t{0};
    CHECK(result == 0, "AMPR read-file completes successfully");
    CHECK(std::memcmp(destination.data(), expected.data() + read_offset, read_size) == 0,
          "AMPR read-file honors its stack-passed offset and fills the DMA destination");
    CHECK(completion[0] == (uint64_t)(uintptr_t)destination.data() &&
              completion[1] == 0 && completion[2] == read_size,
          "AMPR completion publishes destination, success, and byte count");

    // Regression (Terminator 2D: NO FATE, PPSA25872, Unity IL2CPP): some titles pass the completion
    // record INSIDE the request object — a2 = req+0x20, so the bytes-transferred slot a2+0x10 aliases
    // req+0x30, a LIVE guest pointer the engine still uses. A live capture proved the bytes-
    // transferred write clobbered it with the file size (9612 = 0x258c); the guest later freed
    // 0x258c, and the next pop of that lock-free allocator freelist dereferenced 0x258c and SIGSEGV'd.
    // The handler must NOT publish the byte count when its slot overlaps the request object.
    std::array<uint8_t, 0x48> req_overlap{};
    const uint64_t live_ptr = 0x2010216e00ull;                 // the guest's live req+0x30 pointer
    std::memcpy(req_overlap.data() + 0x30, &live_ptr, 8);
    uint64_t* record = reinterpret_cast<uint64_t*>(req_overlap.data() + 0x20);  // a2 == req+0x20
    uint32_t overlap_id = prosper_apr_register(fixture_path, expected.size());
    std::array<uint8_t, read_size> overlap_dst{};
    uint64_t overlap_result = read_file && stub_error.empty()
        ? read_file_guest((uint64_t)(uintptr_t)req_overlap.data(), 0,
                          (uint64_t)(uintptr_t)record, overlap_id,
                          (uint64_t)(uintptr_t)overlap_dst.data(), overlap_dst.size(),
                          read_offset, 0, 0)
        : ~uint64_t{0};
    uint64_t survived = 0; std::memcpy(&survived, req_overlap.data() + 0x30, 8);
    CHECK(overlap_result == 0, "overlapping-record read completes successfully");
    CHECK(survived == live_ptr,
          "bytes-transferred write does not clobber the live req+0x30 pointer (Terminator 2D)");
    CHECK(record[0] == (uint64_t)(uintptr_t)overlap_dst.data() && record[1] == 0,
          "overlapping record still publishes destination + success");

    std::remove(fixture_path);
    prosper_apr_reset_for_test();

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
