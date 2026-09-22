// test_savedata_dirsearch (#2760) — sceSaveDataDirNameSearch must report the saves that are on disk,
// on every host.
//
// The defect: savedata0_list_dirs(), the enumeration behind sceSaveDataDirNameSearch, had its whole
// body inside `#ifndef _WIN32`. On Windows it returned an empty list, so every title was told it had
// NO saves however many it had written — the saves were on disk and a mount of a known dirName still
// opened them, but a title that builds its load/continue list from the search (#299) looked as if it
// had lost them. Nothing failed visibly: "0 saves, SCE_OK" is exactly what a fresh console answers.
//
// Every arm drives the REGISTERED NID (`dyIhnXq-0SM`), not the helper, so the test cannot pass against
// a build where the guest reaches a different answer than the one tested.
//
// Discriminators vs guards, as in test_savedata_mountinfo: [D] reddens on the pre-fix Windows build
// (the `#ifndef _WIN32` stub) or, for the two exclusion lines in arm 3, on a mutation that drops that
// filter (no directory check / no '.' check) — both measured on MinGW; [G] passes either way and pins
// behaviour that already worked (Linux, or an empty root).
//
//  arm 1  registration   [G]  the NID is registered.
//  arm 2  empty root     [G]  a title with no saves gets hitNum 0 and SCE_OK — the honest fresh
//                             console answer, which the pre-fix Windows build gave for EVERY title.
//  arm 3  enumeration    [D]  three saves — two created through the real mount path, one made BY HAND
//                             outside it (so the positive instance does not come from the same source
//                             as the code under test) — are all reported, and nothing else is: a
//                             regular file and a '.'-prefixed directory under the same root are not
//                             saves. Names are written NUL-padded into 32-byte DirName slots.
//  arm 4  dirName filter [D]  cond->dirName selects exactly the named save; [G] an absent name is 0.
//  arm 5  capacity       [D]  dirNamesNum below the save count clamps hitNum/setNum and writes no
//                             slot past the capacity.
//  arm 6  isolation      [D]  another title does not see these saves (#2734) — gated on arm 3's
//                             positive control so it cannot pass vacuously on an empty listing.
#include "hle/dispatch/dispatch.hpp"
#include "hle/fs/save_paths.hpp"
#include "fixtures/test_scratch.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace prosper;
namespace fs = std::filesystem;

static int fails = 0;
static int checks = 0;
#define CHECK(c, m) do { ++checks; if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

namespace {

void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1); else unsetenv(name);
#endif
}

std::string make_app0(const fs::path& base, const char* name, const std::string& title_id) {
    const fs::path root = base / name;
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root / "sce_sys", ec);
    std::ofstream p(root / "sce_sys" / "param.json", std::ios::binary);
    p << "{\"titleId\":\"" << title_id << "\"}";
    p.close();
    return root.string();
}

// The ABI, stated independently of the implementation (PS4-inherited, pinned by DOLL's callsite —
// see s_savedata_dirsearch):
//   SearchCond   { u32 userId @0; titleId* @8; dirName* @0x10; key/order @0x18; ... }
//   SearchResult { u32 hitNum @0; DirName* dirNames @8; u32 dirNamesNum @0x10; u32 setNum @0x14; ... }
//   DirName      { char data[32] }
constexpr size_t DIRNAME_SIZE = 32;
constexpr uint8_t POISON = 0x5A;
constexpr uint32_t kSlots = 8;

struct SearchCond { uint8_t bytes[0x48]; };
struct SearchResult { uint8_t bytes[0x40]; };

HleFn g_search = nullptr;

struct SearchOutcome {
    uint64_t rc = ~0ull;
    uint32_t hit = 0xFFFFFFFFu;
    uint32_t set = 0xFFFFFFFFu;
    std::vector<std::string> names;   // the first `hit` DirName slots, as NUL-terminated strings
    bool padded = true;               // every reported slot is NUL-padded to 32 bytes
    bool beyond_untouched = true;     // slots at index >= hit still hold the poison
};

SearchOutcome search(const char* dirname_filter, uint32_t capacity) {
    SearchCond cond{};
    memset(cond.bytes, 0, sizeof cond.bytes);
    const uint32_t user = 1;
    memcpy(cond.bytes + 0x00, &user, sizeof user);
    const char* filter = dirname_filter;
    memcpy(cond.bytes + 0x10, &filter, sizeof filter);

    static char slots[kSlots][DIRNAME_SIZE];
    memset(slots, POISON, sizeof slots);
    SearchResult res{};
    memset(res.bytes, 0, sizeof res.bytes);
    char* buf = &slots[0][0];
    memcpy(res.bytes + 0x08, &buf, sizeof buf);
    memcpy(res.bytes + 0x10, &capacity, sizeof capacity);

    SearchOutcome out;
    out.rc = g_search((uint64_t)(uintptr_t)&cond, (uint64_t)(uintptr_t)&res, 0, 0, 0, 0);
    memcpy(&out.hit, res.bytes + 0x00, sizeof out.hit);
    memcpy(&out.set, res.bytes + 0x14, sizeof out.set);
    for (uint32_t i = 0; i < kSlots; ++i) {
        if (i < out.hit) {
            const size_t len = strnlen(slots[i], DIRNAME_SIZE);
            if (len == DIRNAME_SIZE) { out.padded = false; continue; }
            for (size_t k = len; k < DIRNAME_SIZE; ++k)
                if (slots[i][k] != 0) out.padded = false;
            out.names.emplace_back(slots[i], len);
        } else {
            for (size_t k = 0; k < DIRNAME_SIZE; ++k)
                if ((uint8_t)slots[i][k] != POISON) out.beyond_untouched = false;
        }
    }
    return out;
}

bool contains(const std::vector<std::string>& v, const std::string& s) {
    return std::find(v.begin(), v.end(), s) != v.end();
}

}   // namespace

int main() {
    printf("== test_savedata_dirsearch ==\n");
    register_builtin_hle();

    // ------------------------------------------------------------------------ arm 1: registration
    g_search = Hle::lookup("dyIhnXq-0SM");
    CHECK(g_search != nullptr, "[G] sceSaveDataDirNameSearch is registered");
    if (!g_search) {
        printf("== FAIL: %d == (%d assertions executed)\n", fails, checks);
        return 1;
    }

    const fs::path scratch = prosper_test::test_scratch_dir() / "savedata-dirsearch";
    std::error_code ec;
    fs::remove_all(scratch, ec);
    fs::create_directories(scratch, ec);
    const fs::path save_root = scratch / "save0";
    fs::create_directories(save_root, ec);
    set_env("PROSPER_SAVE0", save_root.string().c_str());
    const std::string app0_a = make_app0(scratch, "title-a", "PPSA00071");
    const std::string app0_b = make_app0(scratch, "title-b", "PPSA00072");

    // ------------------------------------------------------------------------ arm 2: empty root
    set_app0_root(app0_a);
    savedata0_umount();
    {
        const SearchOutcome r = search(nullptr, kSlots);
        CHECK(r.rc == 0, "[G] a title with no saves: the search succeeds");
        CHECK(r.hit == 0 && r.set == 0, "[G] ...and reports hitNum == setNum == 0");
        CHECK(r.beyond_untouched, "[G] ...and writes no DirName slot");
    }

    // ------------------------------------------------------------------------ arm 3: enumeration
    CHECK(savedata0_mount("SAVE_A", SaveDataMountPolicy::OpenOrCreate) == SaveDataMountOutcome::Created,
          "[G] title A creates SAVE_A through the mount path");
    CHECK(savedata0_umount(), "[G] ...and unmounts it");
    CHECK(savedata0_mount("SAVE_B", SaveDataMountPolicy::OpenOrCreate) == SaveDataMountOutcome::Created,
          "[G] title A creates SAVE_B through the mount path");
    CHECK(savedata0_umount(), "[G] ...and unmounts it");
    // Built by hand, outside the code under test: the positive instance must not come from the same
    // source as the listing it validates.
    const fs::path title_dir = fs::path(savedata0_dir());
    CHECK(fs::create_directory(title_dir / "SAVE_C", ec) && fs::is_directory(title_dir / "SAVE_C"),
          "[G] a third save directory is made directly on disk");
    // Two things under the same root that are NOT saves.
    { std::ofstream f(title_dir / "not_a_save.bin", std::ios::binary); f << "x"; }
    fs::create_directory(title_dir / ".partial", ec);
    CHECK(fs::is_regular_file(title_dir / "not_a_save.bin") && fs::is_directory(title_dir / ".partial"),
          "[G] a regular file and a '.'-prefixed directory sit beside the saves");
    {
        const SearchOutcome r = search(nullptr, kSlots);
        CHECK(r.rc == 0, "[G] the search succeeds with saves present");
        CHECK(r.hit == 3, "[D] hitNum is exactly the three save directories on disk");
        CHECK(r.set == r.hit, "[D] setNum matches hitNum");
        CHECK(contains(r.names, "SAVE_A") && contains(r.names, "SAVE_B") && contains(r.names, "SAVE_C"),
              "[D] every save is reported by its dirName, including the one made outside the mount path");
        CHECK(!contains(r.names, "not_a_save.bin"), "[D] a regular file is not reported as a save");
        CHECK(!contains(r.names, ".partial") && !contains(r.names, ".") && !contains(r.names, ".."),
              "[D] '.'-prefixed entries (and the . / .. pseudo-entries) are not reported");
        CHECK(r.padded, "[D] each reported DirName is NUL-terminated and NUL-padded to 32 bytes");
        CHECK(r.beyond_untouched, "[G] no slot past hitNum is written");
    }

    // ------------------------------------------------------------------------ arm 4: dirName filter
    {
        const SearchOutcome r = search("SAVE_B", kSlots);
        CHECK(r.rc == 0 && r.hit == 1 && r.names.size() == 1 && r.names[0] == "SAVE_B",
              "[D] cond->dirName selects exactly that save");
        const SearchOutcome none = search("SAVE_MISSING", kSlots);
        CHECK(none.rc == 0 && none.hit == 0, "[G] a dirName that does not exist is 0 hits, not an error");
    }

    // ------------------------------------------------------------------------ arm 5: capacity
    {
        const SearchOutcome r = search(nullptr, 2);
        CHECK(r.rc == 0 && r.hit == 2 && r.set == 2,
              "[D] dirNamesNum below the save count clamps hitNum and setNum to the capacity");
        CHECK(r.names.size() == 2 && r.beyond_untouched,
              "[D] ...writing exactly that many DirName slots and none past them");
    }

    // ------------------------------------------------------------------------ arm 6: isolation
    // Only meaningful once arm 3 has shown the listing is not empty; otherwise "title B sees none of
    // title A's saves" is true of a listing that sees nothing at all.
    const bool control = search(nullptr, kSlots).hit == 3;
    set_app0_root(app0_b);
    {
        const SearchOutcome r = search(nullptr, kSlots);
        CHECK(control && r.rc == 0 && r.hit == 0,
              "[D] another title is offered none of these saves (positive control: title A sees 3)");
    }

    printf("== %s: %d == (%d assertions executed)\n", fails ? "FAIL" : "PASS", fails, checks);
    return fails ? 1 : 0;
}
