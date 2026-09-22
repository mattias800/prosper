// test_savedata_capacity (#3654) — a save keeps the allocation it was created with, and
// sceSaveDataGetMountInfo reports blocks / freeBlocks from the save store.
//
// The defect: all three mount entry points decoded dirName and mode and discarded `blocks`, and
// GetMountInfo answered blocks == freeBlocks == 0x40000 whatever was mounted or written. So both the
// allocation-retention arms and the usage arms below fail against the old code: the reported blocks
// never equal what was requested, and writing a file never changes freeBlocks.
//
// Every guest-facing arm goes through the REGISTERED NIDs (Mount 32HQAQdwM2o, Mount2 0z45PIH+SNI,
// Mount3 ZP4e7rlzOUk, Umount BMR4F-Uek3E, GetMountInfo 65VH0Qaaz6s, DirNameSearch dyIhnXq-0SM),
// so a build where a decoder reads the wrong offset fails here rather than in a helper test.
//
// Expected numbers come from the contract in hle/fs/save_capacity.hpp, restated here independently:
// one block is 32768 bytes; each regular file costs ceil(size / 32768) blocks; directories cost
// nothing; the allocation is the CREATING mount's request; free = max(0, allocation - used); a save
// whose allocation is unknown keeps the legacy answer, blocks = max(0x40000, used), free = blocks - used.
//
// Uses a disposable save root under the test scratch directory, never a developer's real saves.
#include "hle/dispatch/dispatch.hpp"
#include "hle/fs/save_capacity.hpp"
#include "fixtures/test_scratch.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using namespace prosper;
namespace fs = std::filesystem;

static int fails = 0;
static int checks = 0;
#define CHECK(c, m) do { ++checks; if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

namespace {

constexpr uint64_t kBlock = 32768;                // the contract's block size, stated independently
constexpr uint64_t kLegacy = 0x40000;             // the pre-#3654 answer an unknown allocation keeps
constexpr uint64_t ERR_NOT_MOUNTED = 0x809F0004ull;
constexpr uint32_t MODE_RDWR = 0x02, MODE_CREATE = 0x04;

void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1); else unsetenv(name);
#endif
}

HleFn g_mount = nullptr, g_mount2 = nullptr, g_mount3 = nullptr, g_umount = nullptr,
      g_info = nullptr, g_search = nullptr;

// A mount result with canaries on both sides: the wrappers must write exactly its 0x40 bytes.
constexpr uint8_t CANARY = 0xA5;
struct GuardedResult {
    uint8_t lead[16];
    uint8_t bytes[0x40];
    uint8_t trail[16];
    GuardedResult() { memset(lead, CANARY, 16); memset(bytes, 0, 0x40); memset(trail, CANARY, 16); }
    bool intact() const {
        for (uint8_t b : lead) if (b != CANARY) return false;
        for (uint8_t b : trail) if (b != CANARY) return false;
        return true;
    }
};

// Each wrapper's own descriptor layout (savedata.cpp documents them; restated here):
//   Mount  (0x50): dirName* @0x10, blocks @0x20, mode @0x28
//   Mount2 (0x40): dirName* @0x08, blocks @0x10, mode @0x18
//   Mount3 (0x30): dirName* @0x08, blocks @0x10, mode @0x20
// The descriptor sits between canary-filled padding so an out-of-range read would see 0xA5 bytes,
// which would decode to a blocks value no arm below expects.
enum class Api { Mount, Mount2, Mount3 };
uint64_t mount(Api api, const char* dirname, uint32_t mode, uint64_t blocks, GuardedResult& r) {
    struct { uint8_t lead[16]; uint8_t d[0x50]; uint8_t trail[16]; } desc;
    memset(&desc, CANARY, sizeof desc);
    size_t size = 0x30, off_name = 0x08, off_blocks = 0x10, off_mode = 0x20;
    if (api == Api::Mount) { size = 0x50; off_name = 0x10; off_blocks = 0x20; off_mode = 0x28; }
    if (api == Api::Mount2) { size = 0x40; off_name = 0x08; off_blocks = 0x10; off_mode = 0x18; }
    memset(desc.d, 0, size);
    memcpy(desc.d + off_name, &dirname, sizeof dirname);
    memcpy(desc.d + off_blocks, &blocks, sizeof blocks);
    memcpy(desc.d + off_mode, &mode, sizeof mode);
    HleFn fn = api == Api::Mount ? g_mount : api == Api::Mount2 ? g_mount2 : g_mount3;
    return fn((uint64_t)(uintptr_t)desc.d, (uint64_t)(uintptr_t)r.bytes, 0, 0, 0, 0);
}
uint64_t mount3(const char* dirname, uint32_t mode, uint64_t blocks) {
    GuardedResult r;
    return mount(Api::Mount3, dirname, mode, blocks, r);
}

struct MountPoint { char data[16]; };
uint64_t umount() {
    MountPoint mp{};
    snprintf(mp.data, sizeof mp.data, "/savedata0");
    return g_umount((uint64_t)(uintptr_t)&mp, 0, 0, 0, 0, 0);
}

struct Info { uint64_t rc = ~0ull, blocks = 0, free_blocks = 0; };
Info info() {
    MountPoint mp{};
    snprintf(mp.data, sizeof mp.data, "/savedata0");
    uint8_t out[48];
    memset(out, 0x5A, sizeof out);
    Info i;
    i.rc = g_info((uint64_t)(uintptr_t)&mp, (uint64_t)(uintptr_t)out, 0, 0, 0, 0);
    memcpy(&i.blocks, out + 0, 8);
    memcpy(&i.free_blocks, out + 8, 8);
    return i;
}

void write_file(const fs::path& p, uint64_t bytes) {
    fs::create_directories(p.parent_path());
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    std::vector<char> buf(bytes, 'x');
    if (bytes) f.write(buf.data(), (std::streamsize)bytes);
}

std::string make_app0(const fs::path& base, const char* name, const std::string& title_id) {
    const fs::path root = base / name;
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root / "sce_sys", ec);
    std::ofstream p(root / "sce_sys" / "param.json", std::ios::binary);
    p << "{\"titleId\":\"" << title_id << "\"}";
    return root.string();
}

std::string read_all(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

bool register_nids() {
    register_builtin_hle();
    g_mount = Hle::lookup("32HQAQdwM2o");
    g_mount2 = Hle::lookup("0z45PIH+SNI");
    g_mount3 = Hle::lookup("ZP4e7rlzOUk");
    g_umount = Hle::lookup("BMR4F-Uek3E");
    g_info = Hle::lookup("65VH0Qaaz6s");
    g_search = Hle::lookup("dyIhnXq-0SM");
    return g_mount && g_mount2 && g_mount3 && g_umount && g_info && g_search;
}

// The "new process" arm: re-executed with --child <save root> <app0>, it opens the save a previous
// process created and reports what GetMountInfo says. Exit 0 = the allocation survived.
int child_main(const char* save_root, const char* app0) {
    if (!register_nids()) return 2;
    set_env("PROSPER_SAVE0", save_root);
    set_app0_root(app0);
    if (mount3("Persist", MODE_RDWR, /*a different request=*/7) != 0) return 3;
    const Info i = info();
    umount();
    // Created with 333 blocks, holding one 1-byte file: 333 / 332.
    return (i.rc == 0 && i.blocks == 333 && i.free_blocks == 332) ? 0 : 4;
}

}   // namespace

int main(int argc, char** argv) {
    if (argc == 4 && std::string(argv[1]) == "--child") return child_main(argv[2], argv[3]);
    printf("== test_savedata_capacity ==\n");
    CHECK(register_nids(), "[G] the mount, umount, GetMountInfo and DirNameSearch NIDs are registered");
    if (fails) return 1;

    const fs::path scratch = prosper_test::test_scratch_dir() / "savedata-capacity";
    std::error_code ec;
    fs::remove_all(scratch, ec);
    const fs::path save_root = scratch / "save0";
    fs::create_directories(save_root, ec);
    set_env("PROSPER_SAVE0", save_root.string().c_str());
    const std::string app_a = make_app0(scratch, "titleA", "PPSA00042");
    const std::string app_b = make_app0(scratch, "titleB", "PPSA00043");
    set_app0_root(app_a);
    const fs::path title_a = save_root / "PPSA00042";
    savedata0_umount();

    // ----------------------------------------------------- arm 1: each decoder carries its blocks
    {
        struct Case { Api api; const char* name; uint64_t blocks; const char* what; };
        const Case cases[] = {
            {Api::Mount, "ViaMount", 200, "sceSaveDataMount carries blocks@0x20 to the save's allocation"},
            {Api::Mount2, "ViaMount2", 300, "sceSaveDataMount2 carries blocks@0x10 to the save's allocation"},
            {Api::Mount3, "ViaMount3", 400, "sceSaveDataMount3 carries blocks@0x10 to the save's allocation"},
        };
        for (const Case& c : cases) {
            GuardedResult r;
            const uint64_t rc = mount(c.api, c.name, MODE_CREATE | MODE_RDWR, c.blocks, r);
            const Info i = info();
            CHECK(rc == 0 && r.intact(), "[G] the creating mount succeeds and writes only its result");
            CHECK(i.rc == 0 && i.blocks == c.blocks && i.free_blocks == c.blocks, c.what);
            umount();
        }
    }

    // ---------------------------------------------------------------- arm 2: usage follows files
    {
        CHECK(mount3("Usage", MODE_CREATE | MODE_RDWR, 400) == 0, "[G] create 'Usage' with 400 blocks");
        const fs::path dir = savedata0_mounted_dir();
        CHECK(info().free_blocks == 400, "[D] an empty save has all 400 blocks free");
        write_file(dir / "a.bin", 1);
        CHECK(info().free_blocks == 399, "[D] a 1-byte file costs one whole block");
        write_file(dir / "a.bin", kBlock);
        CHECK(info().free_blocks == 399, "[D] exactly one block's worth still costs one block");
        write_file(dir / "a.bin", kBlock + 1);
        CHECK(info().free_blocks == 398, "[D] one byte past a block boundary costs a second block");
        write_file(dir / "sub" / "b.bin", 3 * kBlock);
        CHECK(info().free_blocks == 395, "[D] files in subdirectories count; the directory itself does not");
        write_file(dir / "a.bin", 0);
        CHECK(info().free_blocks == 397, "[D] truncating a file to zero releases its blocks");
        fs::remove(dir / "sub" / "b.bin", ec);
        CHECK(info().free_blocks == 400, "[D] deleting a file releases its blocks");
        CHECK(info().blocks == 400, "[D] usage never changes the reported allocation");
        umount();
    }

    // -------------------------------------------- arm 3: reopen keeps the creating request
    {
        CHECK(mount3("Usage", MODE_RDWR, 999) == 0, "[G] reopen 'Usage' asking for 999 blocks");
        CHECK(info().blocks == 400, "[D] an opening mount's request does not resize an existing save");
        umount();
        CHECK(mount3("Usage", MODE_RDWR, 0) == 0 && info().blocks == 400,
              "[D] ...nor does an opening mount with no request");
        umount();
    }

    // ------------------------------------------------------------- arm 4: a new process
    {
        CHECK(mount3("Persist", MODE_CREATE | MODE_RDWR, 333) == 0, "[G] create 'Persist' with 333 blocks");
        write_file(fs::path(savedata0_mounted_dir()) / "one.bin", 1);
        umount();
        const std::string cmd = "\"" + std::string(argv[0]) + "\" --child \"" + save_root.string() +
                                "\" \"" + app_a + "\"";
        const int rc = std::system(cmd.c_str());
        CHECK(rc == 0, "[D] a new process opening the save reports the allocation this one created it with");
    }

    // --------------------------------------------- arm 5: two titles, one dirName
    {
        CHECK(mount3("Shared", MODE_CREATE | MODE_RDWR, 250) == 0, "[G] title A creates 'Shared' (250)");
        umount();
        set_app0_root(app_b);
        CHECK(mount3("Shared", MODE_CREATE | MODE_RDWR, 150) == 0, "[G] title B creates 'Shared' (150)");
        CHECK(info().blocks == 150, "[D] title B's 'Shared' reports its own allocation");
        umount();
        set_app0_root(app_a);
        CHECK(mount3("Shared", MODE_RDWR, 0) == 0 && info().blocks == 250,
              "[D] ...and title A's 'Shared' still reports 250");
        umount();
    }

    // --------------------------------------------- arm 6: legacy saves with no record
    {
        fs::create_directories(title_a / "Legacy", ec);
        write_file(title_a / "Legacy" / "old.bin", 2 * kBlock);
        CHECK(mount3("Legacy", MODE_RDWR, 0) == 0, "[G] open a pre-existing save with no record, no request");
        Info i = info();
        CHECK(i.rc == 0 && i.free_blocks != 0,
              "[D] a legacy save with no record, opened with blocks=0, still reports free space");
        CHECK(i.blocks == kLegacy && i.free_blocks == kLegacy - 2,
              "[D] ...exactly the legacy capacity less the 2 blocks it uses");
        umount();
        CHECK(mount3("Legacy", MODE_RDWR, 120) == 0, "[G] open it again with a 120-block request");
        i = info();
        CHECK(i.blocks == 120 && i.free_blocks == 118,
              "[D] the first request for a save with no record is adopted as its allocation");
        umount();
        CHECK(fs::exists(title_a / "Legacy" / "old.bin", ec) && fs::file_size(title_a / "Legacy" / "old.bin", ec) == 2 * kBlock,
              "[G] the legacy save's own contents are untouched");
    }

    // --------------------------------------------- arm 7: a corrupt record stays untouched
    {
        CHECK(mount3("Corrupt", MODE_CREATE | MODE_RDWR, 64) == 0, "[G] create 'Corrupt' (64)");
        umount();
        const fs::path rec = title_a / kSaveCapacityDirName / "Corrupt";
        { std::ofstream f(rec, std::ios::binary | std::ios::trunc); f << "not a record"; }
        CHECK(mount3("Corrupt", MODE_RDWR, 500) == 0, "[G] open it with a 500-block request");
        const Info i = info();
        CHECK(i.rc == 0 && i.blocks == kLegacy && i.free_blocks == kLegacy,
              "[D] a corrupt record keeps the legacy answer, not the 500 an open asked for");
        umount();
        CHECK(read_all(rec) == "not a record", "[D] an opening mount never overwrites a corrupt record");
        uint64_t b = 0;
        CHECK(save_allocation_read(title_a.string(), "Corrupt", b) == SaveAllocationState::Corrupt,
              "[G] the record reads back as Corrupt");
    }

    // --------------------------------------------- arm 8: unwritable bookkeeping destination
    {
        const fs::path root_c = scratch / "save0-blocked";
        fs::create_directories(root_c / "PPSA00042", ec);
        { std::ofstream f(root_c / "PPSA00042" / kSaveCapacityDirName); f << "a file, not a directory"; }
        set_env("PROSPER_SAVE0", root_c.string().c_str());
        CHECK(mount3("Blocked", MODE_CREATE | MODE_RDWR, 80) == 0,
              "[D] a save whose record cannot be written still mounts");
        const Info i = info();
        CHECK(i.rc == 0 && i.blocks == kLegacy && i.free_blocks == kLegacy,
              "[D] ...and reports its allocation as unknown (legacy answer), not as the unrecorded 80");
        umount();
        set_env("PROSPER_SAVE0", save_root.string().c_str());
    }

    // --------------------------------------------- arm 9: usage past the declared allocation
    {
        CHECK(mount3("Tiny", MODE_CREATE | MODE_RDWR, 1) == 0, "[G] create 'Tiny' with 1 block");
        write_file(fs::path(savedata0_mounted_dir()) / "big.bin", 3 * kBlock);
        const Info i = info();
        CHECK(i.blocks == 1 && i.free_blocks == 0,
              "[D] usage beyond the allocation reports free = 0 and does NOT enlarge the allocation");
        umount();
    }

    // --------------------------------------------- arm 10: a stale record under a new save
    {
        CHECK(mount3("Stale", MODE_CREATE | MODE_RDWR, 90) == 0, "[G] create 'Stale' (90)");
        umount();
        fs::remove_all(title_a / "Stale", ec);   // the save deleted by hand; its record survives
        CHECK(mount3("Stale", MODE_CREATE | MODE_RDWR, 45) == 0 && info().blocks == 45,
              "[D] re-creating a deleted save records the new request, not the stale one");
        umount();
    }

    // --------------------------------------------- arm 11: bookkeeping is invisible to the guest
    {
        const std::vector<std::string> dirs = savedata0_list_dirs();
        CHECK(std::find(dirs.begin(), dirs.end(), kSaveCapacityDirName) == dirs.end() &&
                  std::find(dirs.begin(), dirs.end(), "Usage") != dirs.end(),
              "[D] enumeration lists the saves and not the bookkeeping directory");
        uint8_t res[0x30] = {};
        char names[64][32] = {};
        const uint64_t names_ptr = (uint64_t)(uintptr_t)names;
        const uint32_t cap = 64;
        memcpy(res + 0x08, &names_ptr, sizeof names_ptr);
        memcpy(res + 0x10, &cap, sizeof cap);
        CHECK(g_search(0, (uint64_t)(uintptr_t)res, 0, 0, 0, 0) == 0, "[G] DirNameSearch succeeds");
        uint32_t hits = 0;
        memcpy(&hits, res, sizeof hits);
        bool leaked = false;
        for (uint32_t k = 0; k < hits && k < 64; ++k)
            if (std::string(names[k]) == kSaveCapacityDirName) leaked = true;
        CHECK(hits > 0 && !leaked, "[D] DirNameSearch never reports the bookkeeping directory");
        CHECK(mount3(kSaveCapacityDirName, MODE_RDWR, 0) != 0,
              "[D] the bookkeeping directory cannot be mounted as a save");
        CHECK(info().rc == ERR_NOT_MOUNTED, "[G] ...and nothing is mounted afterwards");
    }

    // --------------------------------------------- arm 12: arithmetic edges
    {
        CHECK(save_blocks_for_bytes(0) == 0 && save_blocks_for_bytes(1) == 1 &&
                  save_blocks_for_bytes(kBlock) == 1 && save_blocks_for_bytes(kBlock + 1) == 2,
              "[G] block rounding at the boundaries");
        CHECK(save_blocks_for_bytes(UINT64_MAX) == UINT64_MAX / kBlock + 1,
              "[G] rounding the largest size does not overflow");
        const SaveCapacity over = save_capacity_from(true, 10, UINT64_MAX);
        CHECK(over.blocks == 10 && over.free_blocks == 0, "[G] usage at the limit reports 0 free");
        const SaveCapacity unknown_big = save_capacity_from(false, 0, kLegacy + 5);
        CHECK(unknown_big.blocks == kLegacy + 5 && unknown_big.free_blocks == 0,
              "[D] an unknown allocation smaller than usage reports blocks = used, free = 0");
        CHECK(save_allocation_write(title_a.string(), "Huge", UINT64_MAX), "[G] the largest allocation is recordable");
        uint64_t b = 0;
        CHECK(save_allocation_read(title_a.string(), "Huge", b) == SaveAllocationState::Present && b == UINT64_MAX,
              "[G] ...and reads back exactly");
        { std::ofstream f(title_a / kSaveCapacityDirName / "Huge", std::ios::binary | std::ios::trunc);
          f << "prosper-savedata-capacity 1\nblocks 18446744073709551616\n"; }
        CHECK(save_allocation_read(title_a.string(), "Huge", b) == SaveAllocationState::Corrupt,
              "[D] an allocation that overflows 64 bits is corruption, not a wrapped small number");
    }

    savedata0_umount();
    fs::remove_all(scratch, ec);
    if (fails) { printf("== FAIL: %d == (%d assertions executed)\n", fails, checks); return 1; }
    printf("== PASS == (%d assertions executed)\n", checks);
    return 0;
}
