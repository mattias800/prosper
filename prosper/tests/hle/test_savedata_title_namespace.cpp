// test_savedata_title_namespace (#2734) — two titles that choose the SAME guest save name must not
// share one host save.
//
// The defect this guards is not "saving is broken" — saving worked. Both save backends resolved to a
// single flat host directory shared by every title, and the subdirectory below it was a name the
// GUEST picks. UE4 titles pick generic ones (`OptionSettings`, `GameUserSettingsini`, `Profile`,
// `Global`, `Inputini`), and SaveDataMemory is keyed by (userId, slotId), which is `1, 0` for
// essentially every Unity title. So a second installed title read the first one's bytes.
//
// What that looks like from the player's seat, and why a round-trip test is not enough: *Little
// Nightmares III* read an `OptionSettings` slot a different Unreal title had written, correctly
// decided it was not its own save format, DELETED it, and held on a modal reading "Your options save
// has corrupted and has been deleted". prosper destroyed a real save and the game reported it in the
// game's own words, so it reads as the title's defect. A test that saves and loads under one title
// passes perfectly on that broken code — the collision only exists BETWEEN titles.
//
// Every arm below therefore runs TWO titles over ONE guest-chosen name, and each is written so that
// it fails on the pre-fix code:
//
//   1. mount     — title B's mount of "OptionSettings" must report CREATED, not OPENED. Pre-fix it
//                  found title A's directory and opened it. This is the LN3 repro in miniature.
//   2. payload   — the bytes title A wrote must not be visible to title B. This is the arm that
//                  proves the outcome above is not just a different status code on shared storage.
//   3. search    — sceSaveDataDirNameSearch (savedata0_list_dirs) must not offer title A's saves to
//                  title B. Pre-fix, every title's load/continue list showed all 34 of this machine's
//                  accumulated slots.
//   4. memory    — the SaveDataMemory API through its real NIDs: title B's Setup of (user 1, slot 0)
//                  must report existed == 0 and read back zeros, not title A's block.
//   5. mutation  — a deliberately malformed titleId must NOT become a host directory name, and both
//                  malformed-title runs must land in the same explicit unknown-title namespace.
//                  Without this arm, "the namespace is derived from param.json" is indistinguishable
//                  from "the namespace is some guest-influenced string pasted into a path".
//   6. default   — with no PROSPER_SAVE0 the root follows the per-user data location rather than the
//                  RAM-backed tmpfs the old default used.
#include "hle/dispatch/dispatch.hpp"
#include "hle/fs/save_paths.hpp"
#include "hle/service/hle_addcontent.hpp"
#include "fixtures/test_scratch.h"

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
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

namespace {

void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1); else unsetenv(name);
#endif
}

// A minimal application root that declares `title_id`, exactly as a real dump does. Going through
// set_app0_root() rather than a test-only setter is deliberate: it exercises the one param.json
// parse the shipping code uses, so the test cannot pass against a derivation the guest never sees.
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

bool write_file(const fs::path& path, const std::string& bytes) {
    std::error_code ec;
    fs::create_directories(path.parent_path(), ec);
    std::ofstream out(path, std::ios::binary);
    out << bytes;
    return out.good();
}

std::string read_file(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

bool contains(const std::vector<std::string>& haystack, const std::string& needle) {
    for (const std::string& s : haystack) if (s == needle) return true;
    return false;
}

// Guest struct mirrors for the SaveDataMemory ABI (natural x86-64 alignment == the guest's LP64
// layout); the same shapes test_savedata_ime asserts offsets for.
struct MemData { void* buf; uint64_t bufSize; int64_t offset; uint8_t rsv[40]; };
struct Setup2  { uint32_t option; int32_t userId; uint64_t memSize; uint64_t iconSize;
                 const void* initParam; const void* initIcon; uint32_t slotId; uint8_t rsv[20]; };
struct SetupResult { uint64_t existed; uint8_t rsv[16]; };
struct Set2    { int32_t userId; uint8_t pad[4]; const MemData* data; const void* param;
                 const void* icon; uint32_t dataNum; uint32_t slotId; uint8_t rsv[32]; };
struct Get2    { int32_t userId; uint8_t pad[4]; MemData* data; const void* param;
                 const void* icon; uint32_t slotId; uint8_t rsv[28]; };
// sceSaveDataSyncSaveDataMemory takes a pointer, not scalars: userId@0, slotId@4.
struct SyncReq { int32_t userId; uint32_t slotId; uint8_t rsv[32]; };

// The guest-chosen name two titles collide on. Verbatim the one Little Nightmares III reported as
// corrupt after another Unreal title wrote it.
constexpr const char* kSharedSlot = "OptionSettings";
constexpr const char* kTitleAOnlySlot = "TitleAPrivateSave";

}   // namespace

int main() {
    printf("== test_savedata_title_namespace ==\n");
    register_builtin_hle();

    const fs::path scratch = prosper_test::test_scratch_dir() / "savedata-title-namespace";
    std::error_code ec;
    fs::remove_all(scratch, ec);
    fs::create_directories(scratch, ec);

    const fs::path save_root = scratch / "save0";
    const fs::path mem_root  = scratch / "savemem";
    fs::create_directories(save_root, ec);
    fs::create_directories(mem_root, ec);
    set_env("PROSPER_SAVE0", save_root.string().c_str());
    set_env("PROSPER_SAVEDATA_DIR", mem_root.string().c_str());

    const std::string app0_a = make_app0(scratch, "title-a", "PPSA00001");
    const std::string app0_b = make_app0(scratch, "title-b", "PPSA00002");

    // ---------------------------------------------------------------- title identity
    set_app0_root(app0_a);
    CHECK(app_param_declaration().title_id == "PPSA00001",
          "the running title's id comes from the one sce_sys/param.json parse");
    CHECK(save_title_namespace() == "PPSA00001",
          "the save namespace is that title id");
    const std::string dir_a = savedata0_dir();

    set_app0_root(app0_b);
    CHECK(save_title_namespace() == "PPSA00002", "switching application changes the namespace");
    const std::string dir_b = savedata0_dir();

    // Kills the whole defect class: pre-fix both of these were the SAME string.
    CHECK(dir_a != dir_b, "two titles resolve to different /savedata0 directories");
    CHECK(dir_a.rfind(save_root.string(), 0) == 0 && dir_b.rfind(save_root.string(), 0) == 0,
          "both live under the configured PROSPER_SAVE0 root");
    CHECK(dir_a != save_root.string() && dir_b != save_root.string(),
          "PROSPER_SAVE0 is a ROOT the titles are namespaced under, not the save directory itself");

    // ---------------------------------------------------------------- arm 1 + 2: mount and payload
    set_app0_root(app0_a);
    CHECK(savedata0_mount(kSharedSlot, SaveDataMountPolicy::OpenOrCreate) ==
              SaveDataMountOutcome::Created,
          "title A creates its own 'OptionSettings' save");
    CHECK(write_file(fs::path(dir_a) / kSharedSlot / "ue4savegame.sav", "TITLE-A-PAYLOAD"),
          "title A writes a payload into it");
    CHECK(savedata0_mount(kTitleAOnlySlot, SaveDataMountPolicy::OpenOrCreate) ==
              SaveDataMountOutcome::Created,
          "title A creates a second, privately named save");
    CHECK(savedata0_umount(), "title A unmounts");

    set_app0_root(app0_b);
    // THE load-bearing assertion. Pre-fix this returned Opened: title B found title A's directory
    // sitting under the shared root and mounted it. Little Nightmares III then read a save it could
    // not parse and deleted it.
    CHECK(savedata0_mount(kSharedSlot, SaveDataMountPolicy::OpenOrCreate) ==
              SaveDataMountOutcome::Created,
          "title B's identically named save is CREATED, not opened onto title A's");
    // And the assertion that the status is backed by real separation rather than bookkeeping.
    CHECK(read_file(fs::path(dir_b) / kSharedSlot / "ue4savegame.sav").empty(),
          "title B cannot see title A's payload under the same guest save name");
    CHECK(read_file(fs::path(dir_a) / kSharedSlot / "ue4savegame.sav") == "TITLE-A-PAYLOAD",
          "and title A's payload is still intact where title A left it");

    // ---------------------------------------------------------------- arm 3: dirName search
    const std::vector<std::string> visible_to_b = savedata0_list_dirs();
    CHECK(contains(visible_to_b, kSharedSlot),
          "sceSaveDataDirNameSearch shows title B its own save");
    CHECK(!contains(visible_to_b, kTitleAOnlySlot),
          "sceSaveDataDirNameSearch does not offer title A's saves to title B");
    CHECK(savedata0_umount(), "title B unmounts");

    set_app0_root(app0_a);
    const std::vector<std::string> visible_to_a = savedata0_list_dirs();
    CHECK(contains(visible_to_a, kSharedSlot) && contains(visible_to_a, kTitleAOnlySlot),
          "title A still sees both of its own saves");

    // ---------------------------------------------------------------- arm 4: SaveDataMemory
    HleFn setup = Hle::lookup("oQySEUfgXRA"), set = Hle::lookup("cduy9v4YmT4"),
          get   = Hle::lookup("QwOO7vegnV8"), sync = Hle::lookup("wiT9jeC7xPw");
    CHECK(setup && set && get && sync, "SaveData memory functions are registered");
    if (setup && set && get && sync) {
        constexpr int32_t kUser = 1;
        constexpr uint32_t kSlot = 0;     // what essentially every Unity title uses
        uint8_t payload[32];
        memset(payload, 0xA5, sizeof payload);

        // Title A: allocate, write, and commit to disk.
        set_app0_root(app0_a);
        Setup2 su{}; su.userId = kUser; su.memSize = sizeof payload; su.slotId = kSlot;
        SetupResult sr{};
        setup((uint64_t)(uintptr_t)&su, (uint64_t)(uintptr_t)&sr, 0, 0, 0, 0);
        MemData md{}; md.buf = payload; md.bufSize = sizeof payload; md.offset = 0;
        Set2 st{}; st.userId = kUser; st.slotId = kSlot; st.data = &md; st.dataNum = 1;
        CHECK(set((uint64_t)(uintptr_t)&st, 0, 0, 0, 0, 0) == 0, "title A writes its memory slot");
        SyncReq sy{}; sy.userId = kUser; sy.slotId = kSlot;
        CHECK(sync((uint64_t)(uintptr_t)&sy, 0, 0, 0, 0, 0) == 0, "title A syncs it to disk");

        // Title B: the same (userId, slotId) must be a fresh, empty slot.
        set_app0_root(app0_b);
        Setup2 sub{}; sub.userId = kUser; sub.memSize = sizeof payload; sub.slotId = kSlot;
        SetupResult srb{}; srb.existed = 0xDEAD;
        setup((uint64_t)(uintptr_t)&sub, (uint64_t)(uintptr_t)&srb, 0, 0, 0, 0);
        CHECK(srb.existed == 0,
              "title B's (user 1, slot 0) reports no existing save, not title A's");
        uint8_t readback[32];
        memset(readback, 0x77, sizeof readback);
        MemData mdb{}; mdb.buf = readback; mdb.bufSize = sizeof readback; mdb.offset = 0;
        Get2 gt{}; gt.userId = kUser; gt.slotId = kSlot; gt.data = &mdb;
        get((uint64_t)(uintptr_t)&gt, 0, 0, 0, 0, 0);
        bool leaked = false;
        for (uint8_t byte : readback) if (byte == 0xA5) leaked = true;
        CHECK(!leaked, "title B reads zeros, not title A's SaveDataMemory bytes");

        // And title A's own block survived title B touching the same key.
        set_app0_root(app0_a);
        CHECK(fs::exists(fs::path(savedata_mem_dir()) / "savemem_1_0.bin"),
              "title A's memory slot file is still where title A wrote it");
        CHECK(savedata_mem_dir() != mem_root.string(),
              "PROSPER_SAVEDATA_DIR is a root the titles are namespaced under too");
    }

    // ---------------------------------------------------------------- arm 5: mutation
    // A titleId that is not one must never reach a host path. Two DIFFERENT malformed declarations
    // are used on purpose: a namespace derived from the raw string would give them different
    // directories and quietly pass an "it is namespaced" test, so the arm asserts they collapse onto
    // one explicit placeholder that no valid title id can spell.
    const std::string app0_bad1 = make_app0(scratch, "title-bad1", "../../escape");
    const std::string app0_bad2 = make_app0(scratch, "title-bad2", "NOTATITLE");
    set_app0_root(app0_bad1);
    const std::string bad1 = save_title_namespace();
    set_app0_root(app0_bad2);
    const std::string bad2 = save_title_namespace();
    CHECK(bad1 == kUnknownTitleNamespace && bad2 == kUnknownTitleNamespace,
          "a malformed titleId yields the explicit unknown-title namespace, not the declared string");
    CHECK(bad1.find('/') == std::string::npos && bad1.find('\\') == std::string::npos &&
              bad1.find("..") == std::string::npos,
          "the namespace can never carry a path separator or a traversal component");
    CHECK(!valid_title_id(kUnknownTitleNamespace),
          "and the placeholder is a name no real title id could collide with");
    CHECK(app_param_declaration().title_id.empty(),
          "a rejected titleId is not published as the application's identity");

    // ---------------------------------------------------------------- arm 6: default location
    // The old default put persistent user data on /tmp, which on the Linux development box is a
    // RAM-backed tmpfs with a per-user quota shared by every concurrent agent.
    const fs::path fake_home = scratch / "home";
    fs::create_directories(fake_home, ec);
    set_env("PROSPER_SAVE0", nullptr);
    set_env("PROSPER_SAVEDATA_DIR", nullptr);
#ifdef _WIN32
    set_env("APPDATA", (fake_home / "AppData" / "Roaming").string().c_str());
#else
    set_env("XDG_DATA_HOME", nullptr);
    set_env("HOME", fake_home.string().c_str());
#endif
    const std::string default_root = savedata0_root();
    const std::string default_mem_root = savedata_mem_root();
    CHECK(default_root.rfind(fake_home.string(), 0) == 0 &&
              default_mem_root.rfind(fake_home.string(), 0) == 0,
          "with no override both roots follow the per-user data location");
    CHECK(default_root.rfind("/tmp/", 0) != 0 && default_mem_root.rfind("/tmp/", 0) != 0,
          "and neither defaults onto the tmpfs the old layout used");
    CHECK(default_root != default_mem_root,
          "the /savedata0 mount and SaveDataMemory keep separate roots");

    printf("== %s: %d ==\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
