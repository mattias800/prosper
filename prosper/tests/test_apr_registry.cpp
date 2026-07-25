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
#include <filesystem>
#include <string>
#include <vector>
#ifndef _WIN32
#include <sys/mman.h>   // mmap/munmap — WriteAddress fault-safety check (#1149/#1154)
#endif
#include "../src/host/exec_image.hpp"
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"

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
    // The handler must NOT publish the byte count for this exact alternate record shape.
    alignas(uint64_t) std::array<uint8_t, 0x48> req_overlap{};
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

    // Do not generalize the live observation into a guessed request-object extent. A caller may
    // place an ordinary three-qword completion record at another address within the storage it
    // passes as a0; a2 remains the authority for that output record.
    std::array<uint64_t, 9> embedded_request{};
    uint64_t* embedded_record = embedded_request.data();
    std::array<uint8_t, read_size> embedded_dst{};
    uint64_t embedded_result = read_file && stub_error.empty()
        ? read_file_guest((uint64_t)(uintptr_t)embedded_request.data(), 0,
                          (uint64_t)(uintptr_t)embedded_record, overlap_id,
                          (uint64_t)(uintptr_t)embedded_dst.data(), embedded_dst.size(),
                          read_offset, 0, 0)
        : ~uint64_t{0};
    CHECK(embedded_result == 0 && embedded_record[2] == read_size,
          "ordinary embedded completion record still publishes byte count");

    std::remove(fixture_path);
    prosper_apr_reset_for_test();

    // sceKernelAprResolveFilepathsWithPrefixToIdsAndFileSizes (GTA V / PPSA04263, RAGE, issue #1130):
    // the prefix-taking twin of the plain resolve. It must prepend `prefix` to each guest path,
    // stat the resolved container, and WRITE all three output arrays (id/size/flags). Stubbing it to
    // success without populating the outputs makes RAGE proceed on a garbage file id (0xffffffff was
    // observed live at the next GetFileStat) and wild-write over its allocator. translate() passes a
    // non-mount path through unchanged, so a relative fixture path exercises the real handler.
    auto resolve_prefix = Hle::lookup(nid_hash("sceKernelAprResolveFilepathsWithPrefixToIdsAndFileSizes"));
    auto resolve_plain  = Hle::lookup(nid_hash("sceKernelAprResolveFilepathsToIdsAndFileSizes"));
    auto resolve_ids    = Hle::lookup("WT-5NKy42fw");
    auto wait_cb        = Hle::lookup("rqwFKI4PAiM");
    CHECK(nid_hash("sceKernelAprResolveFilepathsWithPrefixToIdsAndFileSizes") == "w5fcCG+t31g",
          "WithPrefix resolve hashes to the PS5 3.20 import NID");
    CHECK(resolve_prefix != nullptr && resolve_plain != nullptr, "both APR resolve entries registered");
    CHECK(resolve_ids != nullptr && wait_cb != nullptr,
          "APR id-only resolver and synchronous completion wait registered");

    const char* wp_full = "prosper-test-apr-prefix.tmp";   // prefix + tail == this path
    const char* wp_prefix = "prosper-test-apr-";
    const char* wp_tail   = "prefix.tmp";
    std::array<uint8_t, 321> wp_bytes{};
    for (size_t i = 0; i < wp_bytes.size(); ++i) wp_bytes[i] = (uint8_t)(i * 13u + 5u);
    if (FILE* wf = std::fopen(wp_full, "wb")) { std::fwrite(wp_bytes.data(), 1, wp_bytes.size(), wf); std::fclose(wf); }

    if (resolve_prefix && resolve_plain) {
        // The prefix path is prepended: prefix="prosper-test-apr-", paths[0]="prefix.tmp".
        const char* paths[1] = { wp_tail };
        uint32_t ids[1] = { 0xdeadbeef }; uint64_t sizes[1] = { 0xdead }; uint32_t flags[1] = { 0xdead };
        uint64_t r = resolve_prefix((uint64_t)(uintptr_t)wp_prefix, (uint64_t)(uintptr_t)paths, 1,
                                    (uint64_t)(uintptr_t)ids, (uint64_t)(uintptr_t)sizes, (uint64_t)(uintptr_t)flags);
        CHECK(r == 0, "WithPrefix resolve returns success");
        CHECK(sizes[0] == wp_bytes.size(),
              "WithPrefix resolve prepends the prefix and stats the combined path");
        CHECK(ids[0] >= 1 && flags[0] == 0, "WithPrefix resolve populates id and flags");
        CHECK(prosper_apr_path_for_id(ids[0]) == wp_full,
              "WithPrefix registers the combined host path under the returned id");

        // Sonic/CRI uses the compact (paths,count,ids,errorIndex) resolver.  errorIndex is a scalar,
        // so guard it with canaries and prove a multi-entry call does not treat it as a flags array.
        if (resolve_ids && wait_cb) {
            const char* compact_paths[2] = { wp_full, wp_full };
            uint32_t compact_ids[2] = {0, 0};
            uint32_t error_guard[3] = {0x11111111u, 0xDEADBEEFu, 0x22222222u};
            uint64_t rc = resolve_ids((uint64_t)(uintptr_t)compact_paths, 2,
                                      (uint64_t)(uintptr_t)compact_ids,
                                      (uint64_t)(uintptr_t)&error_guard[1], 0, 0);
            CHECK(rc == 0 && compact_ids[0] >= 1 && compact_ids[0] == compact_ids[1],
                  "id-only APR resolver writes stable ids for every path");
            CHECK(error_guard[0] == 0x11111111u && error_guard[1] == 0 &&
                      error_guard[2] == 0x22222222u,
                  "id-only resolver writes only the scalar errorIndex");
            CHECK(wait_cb(1, 0, 0, 0, 0, 0) == 0,
                  "APR wait observes eagerly completed command buffers");

            // Sonic Origins keeps loose Hedgehog Engine assets below app0/raw, while CRI's ACB
            // metadata names an external AWB relative to that content root (sound/Foo.awb). APR
            // must resolve the missing /app0/sound spelling to /app0/raw/sound without changing an
            // ordinary path that already exists. This is what turns the correctly loaded ACB into
            // an audible bank rather than a silent mixer.
            namespace fs = std::filesystem;
            const fs::path app0 = "prosper-test-apr-app0";
            const fs::path raw_sound = app0 / "raw" / "sound";
            fs::create_directories(raw_sound);
            const fs::path awb = raw_sound / "fallback.awb";
            if (FILE* af = std::fopen(awb.string().c_str(), "wb")) {
                std::fwrite(wp_bytes.data(), 1, wp_bytes.size(), af);
                std::fclose(af);
            }
            set_app0_root(app0.string());
            const char* awb_paths[1] = { "/app0/sound/fallback.awb" };
            uint32_t awb_id = 0, awb_error = ~uint32_t{0};
            uint64_t awb_rc = resolve_ids((uint64_t)(uintptr_t)awb_paths, 1,
                                          (uint64_t)(uintptr_t)&awb_id,
                                          (uint64_t)(uintptr_t)&awb_error, 0, 0);
            CHECK(awb_rc == 0 && awb_id >= 1 && awb_error == 0,
                  "APR content-root fallback resolves /app0/sound through /app0/raw/sound");
            std::error_code awb_path_error;
            const bool same_awb = fs::equivalent(
                fs::path(prosper_apr_path_for_id(awb_id)), awb, awb_path_error);
            CHECK(!awb_path_error && same_awb,
                  "APR content-root fallback registers the real raw-content path");
            set_app0_root(".");
            fs::remove(awb);
            fs::remove(raw_sound);
            fs::remove(app0 / "raw");
            fs::remove(app0);
        }

        // An empty prefix must behave exactly like the plain (non-prefix) resolve.
        prosper_apr_reset_for_test();
        const char* full_paths[1] = { wp_full };
        uint32_t id_wp = 0, id_plain = 0; uint64_t sz_wp = 0, sz_plain = 0; uint32_t fl_wp = 1, fl_plain = 1;
        resolve_prefix((uint64_t)(uintptr_t)"", (uint64_t)(uintptr_t)full_paths, 1,
                       (uint64_t)(uintptr_t)&id_wp, (uint64_t)(uintptr_t)&sz_wp, (uint64_t)(uintptr_t)&fl_wp);
        prosper_apr_reset_for_test();
        resolve_plain((uint64_t)(uintptr_t)full_paths, 1,
                      (uint64_t)(uintptr_t)&id_plain, (uint64_t)(uintptr_t)&sz_plain, (uint64_t)(uintptr_t)&fl_plain, 0);
        CHECK(id_wp == id_plain && sz_wp == sz_plain && sz_wp == wp_bytes.size() && fl_wp == 0,
              "empty prefix matches the plain resolve");

        // A missing container zeroes its outputs (no garbage id) rather than failing the batch.
        prosper_apr_reset_for_test();
        const char* miss_paths[1] = { "prosper-test-apr-does-not-exist.tmp" };
        uint32_t miss_id = 0x55; uint64_t miss_sz = 0x55; uint32_t miss_fl = 0x55;
        uint64_t rm = resolve_prefix((uint64_t)(uintptr_t)"", (uint64_t)(uintptr_t)miss_paths, 1,
                                     (uint64_t)(uintptr_t)&miss_id, (uint64_t)(uintptr_t)&miss_sz,
                                     (uint64_t)(uintptr_t)&miss_fl);
        CHECK(rm == 0 && miss_id == 0 && miss_sz == 0 && miss_fl == 0,
              "unresolvable container zeroes its outputs");

        // A null paths pointer / non-positive count is rejected without touching memory.
        CHECK((uint32_t)resolve_prefix((uint64_t)(uintptr_t)"", 0, 1, 0, 0, 0) == 0x80020016u,
              "WithPrefix resolve rejects a null paths array");
    }
    std::remove(wp_full);
    prosper_apr_reset_for_test();

    // sceAmprCommandBufferWriteAddress (GTA V / PPSA04263, RAGE, issue #1149): the APR completion-
    // notification primitive j0+3uJMxYJY(cb, address, value, flags). It must WRITE `value` to
    // `*address` (the APR engine performs the queued write when the command buffer executes; prosper
    // serves the reads eagerly, so the write happens here). RAGE pre-sets the target to a "pending"
    // sentinel and a waiter thread spin-polls it for the queued value; stubbing the NID to a bare 0
    // (writing nothing) left the target pending forever and hung the main thread on a load future
    // (the title-screen stall). This regression FAILS against the old stub — nothing writes the value.
    // The NID j0+3uJMxYJY is taken verbatim from GTA V's import table (its exact firmware name is not
    // in the PS5 3.20 library dump — it does NOT hash from "sceAmprCommandBufferWriteAddress"; the
    // KytyPS5 label is a guess). The NID is the authority, so the handler is registered by raw NID.
    register_kernel_mem_hle();
    HleFn write_address = Hle::lookup("j0+3uJMxYJY");
    CHECK(write_address != nullptr, "AMPR WriteAddress HLE registered");
    const std::vector<ImportSlot> wa_slots = {{"libSceAmpr", "j0+3uJMxYJY"}};
    std::string wa_stub_error;
    CHECK(install_stubs(wa_slots, 0x730000000ull, 96, &wa_stub_error),
          "generated executable AMPR WriteAddress import stub");
    using GuestWriteAddress = uint64_t (__attribute__((sysv_abi)) *)(
        uint64_t, uint64_t, uint64_t, uint64_t);
    auto write_address_guest = reinterpret_cast<GuestWriteAddress>(
        static_cast<uintptr_t>(stub_addr(0)));
    if (write_address && wa_stub_error.empty()) {
        // The guest pre-arms the target with a pending sentinel and queues the completion value.
        uint64_t target = 0xffffffffffffffffull;   // matches the live GTA V pre-set
        uint64_t cb = 0x20001f13f0ull;              // opaque cb handle (not dereferenced)
        uint64_t r = write_address_guest(cb, (uint64_t)(uintptr_t)&target, 0x2a, 0x60);
        CHECK(r == 0, "WriteAddress returns success");
        CHECK(target == 0x2a, "WriteAddress writes the queued value to the target (clears pending)");

        // The value is written verbatim (RAGE increments a per-submit completion token, not always 0).
        uint64_t token_target = 0;
        write_address_guest(cb, (uint64_t)(uintptr_t)&token_target, 7, 0);
        CHECK(token_target == 7, "WriteAddress writes an arbitrary token verbatim");

        // A null address must be a safe no-op (never fault the HLE).
        CHECK(write_address_guest(cb, 0, 0x2a, 0) == 0, "WriteAddress tolerates a null address");

        // Fault-safety (issue #1154): a non-null but UNMAPPED target must not fault the HLE. Reserve
        // then release a page so the address is valid-shaped but unmapped, and confirm the handler
        // returns cleanly (it drops the undeliverable write and logs, rather than SIGSEGVing).
#ifndef _WIN32
        void* scratch = mmap(nullptr, 0x1000, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (scratch != MAP_FAILED) {
            uint64_t unmapped = (uint64_t)(uintptr_t)scratch;
            munmap(scratch, 0x1000);
            CHECK(write_address_guest(cb, unmapped, 0x2a, 0) == 0,
                  "WriteAddress tolerates an unmapped target without faulting");
        }
#endif
    }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
