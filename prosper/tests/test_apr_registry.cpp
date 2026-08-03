// test_apr_registry — guards APR container registration and read-source resolution (#62/#1901).
// The read path uses the real APR file id first and consults total byte size only as a legacy
// fallback for an unresolved id. This locks in stable ids, path dedup, valid colliding-id reads,
// unique-size fallback, ambiguous fallback refusal, and truthful resolution-method diagnostics.
#include <cstdio>
#include <cstdint>
#include <array>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif
#ifndef _WIN32
#include <sys/mman.h>   // mmap/munmap — WriteAddress fault-safety check (#1149/#1154)
#endif
#include "../src/host/exec_image.hpp"
#include "../src/host/guest_write_watch.hpp"
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

static bool set_test_env(const char* name, const char* value) {
#ifdef _WIN32
    return _putenv_s(name, value) == 0;
#else
    return setenv(name, value, 1) == 0;
#endif
}

static int file_descriptor(FILE* file) {
#ifdef _WIN32
    return _fileno(file);
#else
    return fileno(file);
#endif
}

static int duplicate_descriptor(int fd) {
#ifdef _WIN32
    return _dup(fd);
#else
    return dup(fd);
#endif
}

static int replace_descriptor(int from, int to) {
#ifdef _WIN32
    return _dup2(from, to);
#else
    return dup2(from, to);
#endif
}

static void close_descriptor(int fd) {
#ifdef _WIN32
    _close(fd);
#else
    close(fd);
#endif
}

template <typename Fn>
static std::string capture_stderr(Fn&& fn) {
    FILE* capture = std::tmpfile();
    if (!capture) return {};
    std::fflush(stderr);
    const int saved_stderr = duplicate_descriptor(file_descriptor(stderr));
    if (saved_stderr < 0 || replace_descriptor(file_descriptor(capture), file_descriptor(stderr)) < 0) {
        if (saved_stderr >= 0) close_descriptor(saved_stderr);
        std::fclose(capture);
        return {};
    }
    fn();
    std::fflush(stderr);
    replace_descriptor(saved_stderr, file_descriptor(stderr));
    close_descriptor(saved_stderr);
    std::rewind(capture);
    std::string output;
    char buffer[512];
    while (size_t n = std::fread(buffer, 1, sizeof buffer, capture))
        output.append(buffer, n);
    std::fclose(capture);
    return output;
}

int main() {
    printf("== test_apr_registry ==\n");
    CHECK(set_test_env("PROSPER_FILELOG", "1"), "APR read diagnostics enabled for regression");
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
    HleFn resolve_plain = Hle::lookup(nid_hash("sceKernelAprResolveFilepathsToIdsAndFileSizes"));
    CHECK(read_file != nullptr && resolve_plain != nullptr,
          "AMPR read-file and APR resolve HLEs registered");
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
    std::array<uint8_t, 0x90> destination{};
    uint64_t result = read_file && stub_error.empty()
        ? read_file_guest((uint64_t)(uintptr_t)request.data(), 0,
                          (uint64_t)(uintptr_t)completion.data(), fixture_id,
                          (uint64_t)(uintptr_t)destination.data(), read_size,
                          read_offset, 0, 0)
        : ~uint64_t{0};
    CHECK(result == 0, "AMPR read-file completes successfully");
    CHECK(std::memcmp(destination.data(), expected.data() + read_offset, read_size) == 0,
          "AMPR read-file honors its stack-passed offset and fills the DMA destination");
    CHECK(completion[0] == (uint64_t)(uintptr_t)destination.data() &&
              completion[1] == 0 && completion[2] == read_size,
          "AMPR completion publishes destination, success, and byte count");

    // #1901: equal total sizes do not make real ID-resolved reads ambiguous. Register a second,
    // byte-distinct fixture through the public resolve HLE so this also proves its warning describes
    // only the unresolved-id fallback rather than falsely declaring both files unreadable.
    const char* collision_path = "prosper-test-apr-read-collision.tmp";
    std::array<uint8_t, expected.size()> collision_bytes{};
    for (size_t i = 0; i < collision_bytes.size(); ++i)
        collision_bytes[i] = (uint8_t)(i * 17u + 11u);
    if (FILE* collision = std::fopen(collision_path, "wb")) {
        CHECK(std::fwrite(collision_bytes.data(), 1, collision_bytes.size(), collision) ==
                  collision_bytes.size(),
              "write equal-size APR collision fixture");
        std::fclose(collision);
    } else {
        CHECK(false, "create equal-size APR collision fixture");
    }
    const char* collision_paths[2] = { fixture_path, collision_path };
    uint32_t collision_ids[2]{};
    uint64_t collision_sizes[2]{};
    uint32_t collision_error = ~uint32_t{0};
    uint64_t collision_resolve_result = ~uint64_t{0};
    const std::string collision_warning = capture_stderr([&] {
        collision_resolve_result = resolve_plain(
            (uint64_t)(uintptr_t)collision_paths, 2,
            (uint64_t)(uintptr_t)collision_ids,
            (uint64_t)(uintptr_t)collision_sizes,
            (uint64_t)(uintptr_t)&collision_error, 0);
    });
    CHECK(collision_resolve_result == 0 && collision_error == 0 &&
              collision_ids[0] == fixture_id && collision_ids[1] != collision_ids[0] &&
              collision_sizes[0] == expected.size() && collision_sizes[1] == expected.size(),
          "equal-size containers resolve to distinct usable ids");
    CHECK(collision_warning.find("ID-resolved reads remain valid") != std::string::npos &&
              collision_warning.find("unresolved-id size fallback will be refused") != std::string::npos &&
              collision_warning.find("reads of either will be refused") == std::string::npos,
          "collision warning describes only the conditional size fallback");

    constexpr size_t method_offset = 7;
    constexpr size_t method_size = 31;
    auto read_with_method = [&](uint32_t id, uint64_t total_size,
                                std::array<uint8_t, 0x90>& out,
                                std::array<uint64_t, 3>& out_completion,
                                uint64_t& out_result) {
        std::array<uint8_t, 0x48> method_request{};
        std::memcpy(method_request.data() + 0x30, &total_size, sizeof total_size);
        return capture_stderr([&] {
            out_result = read_file_guest(
                (uint64_t)(uintptr_t)method_request.data(), 0,
                (uint64_t)(uintptr_t)out_completion.data(), id,
                (uint64_t)(uintptr_t)out.data(), method_size,
                method_offset, 0, 0);
        });
    };

    std::array<uint8_t, 0x90> first_id_bytes{}, second_id_bytes{};
    std::array<uint64_t, 3> first_id_completion{}, second_id_completion{};
    uint64_t first_id_result = ~uint64_t{0}, second_id_result = ~uint64_t{0};
    const std::string first_id_log = read_with_method(
        collision_ids[0], expected.size(), first_id_bytes, first_id_completion, first_id_result);
    const std::string second_id_log = read_with_method(
        collision_ids[1], collision_bytes.size(), second_id_bytes,
        second_id_completion, second_id_result);
    CHECK(first_id_result == 0 && second_id_result == 0 &&
              std::memcmp(first_id_bytes.data(), expected.data() + method_offset, method_size) == 0 &&
              std::memcmp(second_id_bytes.data(), collision_bytes.data() + method_offset, method_size) == 0,
          "valid ids read the correct byte-distinct files despite a size collision");
    CHECK(first_id_log.find("method=id") != std::string::npos &&
              second_id_log.find("method=id") != std::string::npos,
          "colliding valid-id reads report method=id");

    // An unknown id may use total-size correlation only when exactly one registered file matches.
    const char* unique_path = "prosper-test-apr-read-fallback.tmp";
    std::array<uint8_t, 263> unique_bytes{};
    for (size_t i = 0; i < unique_bytes.size(); ++i)
        unique_bytes[i] = (uint8_t)(i * 23u + 19u);
    if (FILE* unique = std::fopen(unique_path, "wb")) {
        CHECK(std::fwrite(unique_bytes.data(), 1, unique_bytes.size(), unique) == unique_bytes.size(),
              "write unique-size APR fallback fixture");
        std::fclose(unique);
    } else {
        CHECK(false, "create unique-size APR fallback fixture");
    }
    prosper_apr_register(unique_path, unique_bytes.size());
    constexpr uint32_t unknown_id = 0xf00dcafeu;
    std::array<uint8_t, 0x90> fallback_bytes{};
    std::array<uint64_t, 3> fallback_completion{};
    uint64_t fallback_result = ~uint64_t{0};
    const std::string fallback_log = read_with_method(
        unknown_id, unique_bytes.size(), fallback_bytes, fallback_completion, fallback_result);
    CHECK(fallback_result == 0 &&
              std::memcmp(fallback_bytes.data(), unique_bytes.data() + method_offset, method_size) == 0,
          "unknown id uses an unambiguous total-size fallback");
    CHECK(fallback_log.find("method=size-fallback") != std::string::npos &&
              fallback_log.find("requested=31") != std::string::npos,
          "unique-size fallback reports its method and requested size");

    std::array<uint8_t, 0x90> refused_bytes{};
    refused_bytes.fill(0xa5);
    std::array<uint64_t, 3> refused_completion{
        0x1111111111111111ull, 0x2222222222222222ull, 0x3333333333333333ull};
    uint64_t refused_result = 0;
    const std::string refused_log = read_with_method(
        unknown_id, expected.size(), refused_bytes, refused_completion, refused_result);
    CHECK((uint32_t)refused_result == 0x80020016u &&
              refused_completion[0] == 0x1111111111111111ull &&
              refused_completion[1] == 0x2222222222222222ull &&
              refused_completion[2] == 0x3333333333333333ull && refused_bytes[0] == 0xa5,
          "ambiguous size fallback refuses without completing or writing the read");
    CHECK(refused_log.find("method=refused") != std::string::npos &&
              refused_log.find("matches=2") != std::string::npos &&
              refused_log.find("offset=0x7") != std::string::npos,
          "ambiguous fallback reports refusal, match count, and requested offset");

#ifndef _WIN32
    // APR writes are performed by the host kernel, so they do not enter the guest SIGSEGV handler
    // that normally disarms a renderer/compute dirty-tracking page.  Guard a real mapped destination
    // to prove ReadFile explicitly disarms the watch, copies every byte, and invalidates the cache.
    constexpr size_t watched_size = 0x4000;
    auto* watched_destination = static_cast<uint8_t*>(mmap(
        nullptr, watched_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    CHECK(watched_destination != MAP_FAILED, "map watched APR destination");
    if (watched_destination != MAP_FAILED) {
        constexpr uint64_t watched_phys = 0x7a0000;
        host::guest_write_watch_set_fault_onstack(true);
        host::guest_write_watch_notify_direct_mapping_added(
            (uint64_t)(uintptr_t)watched_destination, watched_size, watched_phys,
            0x3 /* SCE CPU_READ|CPU_WRITE */);
        host::GuestWriteWatch destination_watch = host::GuestWriteWatch::create(
            (uint64_t)(uintptr_t)watched_destination, watched_size);

        std::array<uint64_t, 3> watched_completion{};
        uint64_t watched_result = read_file_guest(
            (uint64_t)(uintptr_t)request.data(), 0,
            (uint64_t)(uintptr_t)watched_completion.data(), fixture_id,
            (uint64_t)(uintptr_t)watched_destination, read_size,
            read_offset, 0, 0);
        CHECK(watched_result == 0 &&
                  std::memcmp(watched_destination, expected.data() + read_offset, read_size) == 0,
              "AMPR read fills a write-protected DMA destination completely");
        CHECK(watched_completion[0] == (uint64_t)(uintptr_t)watched_destination,
              "AMPR publishes the guarded destination instead of a fallback staging pointer");
        if (destination_watch) {
            CHECK(destination_watch.query() == host::GuestWriteWatchQuery::Dirty,
                  "AMPR host write invalidates the destination cache watch");
        } else {
            CHECK(destination_watch.query() == host::GuestWriteWatchQuery::Unknown,
                  "unsupported destination watch retains the exact comparison fallback");
        }

        destination_watch.reset();
        host::guest_write_watch_notify_direct_mapping_removed(
            (uint64_t)(uintptr_t)watched_destination, watched_size);
        host::guest_write_watch_set_fault_onstack(false);
        munmap(watched_destination, watched_size);
    }
#endif

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
    std::array<uint8_t, 0x90> overlap_dst{};
    uint64_t overlap_result = read_file && stub_error.empty()
        ? read_file_guest((uint64_t)(uintptr_t)req_overlap.data(), 0,
                          (uint64_t)(uintptr_t)record, overlap_id,
                          (uint64_t)(uintptr_t)overlap_dst.data(), read_size,
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
    std::array<uint8_t, 0x90> embedded_dst{};
    uint64_t embedded_result = read_file && stub_error.empty()
        ? read_file_guest((uint64_t)(uintptr_t)embedded_request.data(), 0,
                          (uint64_t)(uintptr_t)embedded_record, overlap_id,
                          (uint64_t)(uintptr_t)embedded_dst.data(), read_size,
                          read_offset, 0, 0)
        : ~uint64_t{0};
    CHECK(embedded_result == 0 && embedded_record[2] == read_size,
          "ordinary embedded completion record still publishes byte count");

    std::remove(fixture_path);
    std::remove(collision_path);
    std::remove(unique_path);
    prosper_apr_reset_for_test();

    // sceKernelAprResolveFilepathsWithPrefixToIdsAndFileSizes (GTA V / PPSA04263, RAGE, issue #1130):
    // the prefix-taking twin of the plain resolve. It must prepend `prefix` to each guest path,
    // stat the resolved container, and write its id/size outputs. Stubbing it to
    // success without populating the outputs makes RAGE proceed on a garbage file id (0xffffffff was
    // observed live at the next GetFileStat) and wild-write over its allocator. translate() passes a
    // non-mount path through unchanged, so a relative fixture path exercises the real handler.
    auto resolve_prefix = Hle::lookup(nid_hash("sceKernelAprResolveFilepathsWithPrefixToIdsAndFileSizes"));
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
        uint32_t ids[1] = { 0xdeadbeef }; uint64_t sizes[1] = { 0xdead }; uint32_t error_index = 0xdead;
        uint64_t r = resolve_prefix((uint64_t)(uintptr_t)wp_prefix, (uint64_t)(uintptr_t)paths, 1,
                                    (uint64_t)(uintptr_t)ids, (uint64_t)(uintptr_t)sizes,
                                    (uint64_t)(uintptr_t)&error_index);
        CHECK(r == 0, "WithPrefix resolve returns success");
        CHECK(sizes[0] == wp_bytes.size(),
              "WithPrefix resolve prepends the prefix and stats the combined path");
        CHECK(ids[0] >= 1 && error_index == 0,
              "WithPrefix resolve populates the id and clears the scalar error index");
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
        uint32_t id_wp = 0, id_plain = 0; uint64_t sz_wp = 0, sz_plain = 0;
        uint32_t err_wp = 1, err_plain = 1;
        resolve_prefix((uint64_t)(uintptr_t)"", (uint64_t)(uintptr_t)full_paths, 1,
                       (uint64_t)(uintptr_t)&id_wp, (uint64_t)(uintptr_t)&sz_wp,
                       (uint64_t)(uintptr_t)&err_wp);
        prosper_apr_reset_for_test();
        resolve_plain((uint64_t)(uintptr_t)full_paths, 1,
                      (uint64_t)(uintptr_t)&id_plain, (uint64_t)(uintptr_t)&sz_plain,
                      (uint64_t)(uintptr_t)&err_plain, 0);
        CHECK(id_wp == id_plain && sz_wp == sz_plain && sz_wp == wp_bytes.size() &&
                  err_wp == 0 && err_plain == 0,
              "empty prefix matches the plain resolve");

        // ArcRunner asks for a loose Game.locres that lives only inside its pak. Returning success
        // with id/size zero makes UE parse an errored reader and use stale stack data as a count,
        // trapping startup in a multi-billion-iteration loop. Resolve the preceding valid entry,
        // then fail at the missing entry with its scalar index and without writing past that scalar.
        prosper_apr_reset_for_test();
        const char* miss_paths[2] = { wp_full, "prosper-test-apr-does-not-exist.tmp" };
        uint32_t miss_ids[2] = { 0x55, 0x55 }; uint64_t miss_sizes[2] = { 0x55, 0x55 };
        uint32_t miss_error_guard[3] = { 0x11111111u, 0xDEADBEEFu, 0x22222222u };
        uint64_t rm = resolve_plain((uint64_t)(uintptr_t)miss_paths, 2,
                                    (uint64_t)(uintptr_t)miss_ids,
                                    (uint64_t)(uintptr_t)miss_sizes,
                                    (uint64_t)(uintptr_t)&miss_error_guard[1], 0);
        CHECK((uint32_t)rm == 0x80020002u,
              "missing APR path returns SCE_KERNEL_ERROR_ENOENT");
        CHECK(miss_ids[0] >= 1 && miss_sizes[0] == wp_bytes.size() &&
                  miss_ids[1] == 0xffffffffu && miss_sizes[1] == 0,
              "APR resolver stops with deterministic outputs at the missing entry");
        CHECK(miss_error_guard[0] == 0x11111111u && miss_error_guard[1] == 1 &&
                  miss_error_guard[2] == 0x22222222u,
              "APR resolver reports the scalar failure index without an array overrun");

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


    // #1629: sceKernelAprSubmitCommandBuffer (eE4Szl8sil8) was unimplemented, so the default stub
    // returned 0 = success without consuming the command buffer and CRI ADX2's reads never completed.
    //
    // The 3.20 firmware export database also corrects a name prosper had guessed: ASoW5WE-UPo is
    // sceKernelAprSubmitCommandBufferAndGetResult (cb, ring, out1, out2), NOT the plain submit. The
    // plain form takes no result slots, so a2/a3 hold caller residue — writing a token through them
    // is the specific bug an alias-style "fix" would introduce, and this file already documents that a
    // nonzero value in a completion record marks the read FAILED.
    {
        HleFn submit_plain = Hle::lookup("eE4Szl8sil8");
        HleFn submit_result_a = Hle::lookup("ASoW5WE-UPo");
        CHECK(submit_plain != nullptr,
              "#1629: sceKernelAprSubmitCommandBuffer (eE4Szl8sil8) resolves to a real handler");
        CHECK(submit_result_a != nullptr,
              "#1629: sceKernelAprSubmitCommandBufferAndGetResult (ASoW5WE-UPo) still resolves");
#ifndef _WIN32
        // POSIX only, and this guard hides a REAL gap rather than an irrelevance: on Windows both NIDs
        // share k_ampr_submit, which resets the command buffer and writes nothing, so AndGetResult
        // silently returns success without publishing a completion token. That is pre-existing and is
        // tracked as #1657 — REMOVE THIS GUARD when it is fixed; the assertions below are already the
        // proof. They are guarded rather than deleted so the Windows contract is not quietly forgotten.
        // The plain form's contract IS satisfied on Windows (it has no result slots), which is why its
        // residue assertion further down stays unguarded.
        // Both halves required: before #1629 the plain NID resolved to nothing, so a bare "these
        // differ" comparison passed vacuously against nullptr.
        CHECK(submit_plain && submit_result_a && submit_plain != submit_result_a,
              "#1629: the plain submit exists AND is not an alias of the AndGetResult form");
#endif
        // The _TEST-suffixed exports must stay unimplemented. They are NOT alias NIDs of the two
        // names above — the firmware database resolves them to sceKernelAprSubmitCommandBuffer_TEST
        // and ...AndGetResult_TEST — and binding a debug entry point to a production handler would be
        // inventing behaviour no title evidence supports.
        CHECK(Hle::lookup("Omr9X+YmT7I") == nullptr && Hle::lookup("0ers1N4C9CY") == nullptr,
              "#1629: the _TEST-suffixed submit NIDs are left unimplemented, not aliased");

        // An unbound command buffer is the case that hands a token back through the out slots, so it
        // is the one that can demonstrate the difference.
        alignas(8) uint64_t out1 = 0xdeadbeefcafef00dull;
        alignas(8) uint64_t out2 = 0xdeadbeefcafef00dull;
        constexpr uint64_t kResidue = 0xdeadbeefcafef00dull;
        alignas(64) uint8_t fake_cb[256] = {};
        const uint64_t cb = (uint64_t)(uintptr_t)fake_cb;

#ifndef _WIN32   // see #1657 — Windows publishes no token; remove with that fix
        if (submit_result_a) {
            out1 = out2 = kResidue;
            submit_result_a(cb, 1, (uint64_t)(uintptr_t)&out1, (uint64_t)(uintptr_t)&out2, 0, 0);
            CHECK(out1 != kResidue && out2 != kResidue && out1 == out2,
                  "#1629: AndGetResult still publishes one completion token through both out slots");
        }
#endif
        if (submit_plain) {
            out1 = out2 = kResidue;
            // Exactly the call the plain form receives: whatever happens to be in the argument
            // registers. If the handler treated them as out-pointers it would clobber them.
            submit_plain(cb, 1, (uint64_t)(uintptr_t)&out1, (uint64_t)(uintptr_t)&out2, 0, 0);
            // Holds on BOTH platforms and is the property that actually matters, so it is not
            // guarded: POSIX suppresses the writes deliberately, Windows never had them.
            CHECK(out1 == kResidue && out2 == kResidue,
                  "#1629: the plain submit writes NO result slots — a2/a3 are residue, not outputs");
        }
    }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
