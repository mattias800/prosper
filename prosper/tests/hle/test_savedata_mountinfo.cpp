// test_savedata_mountinfo (#3653) — `sceSaveDataGetMountInfo` must describe a mount that is there.
//
// The defect: the handler ignored its mount-point argument entirely. Given any non-null output
// pointer it cleared 48 bytes, wrote two fixed capacity numbers and returned SCE_OK — so a NULL
// mount point, "/savedata1", and "/savedata0" *after an unmount* all received a successful capacity
// report for a mount that was not there. The one call whose job is to say "this mount point is live
// and this big" could not say that it was absent.
//
// Every arm goes through the REGISTERED NIDs — GetMountInfo `65VH0Qaaz6s`, and Mount3/Umount for the
// lifecycle — because a test that called the internal helper would pass against a build where the
// guest still reaches the dispatcher's `return 0`.
//
// WHICH ARMS ARE DISCRIMINATORS AND WHICH ARE GUARDS. A discriminator fails when the validation is
// reverted; a guard passes either way and exists so a later change cannot quietly break something
// that already worked. Each CHECK below is tagged [D] or [G], and the tags are not decoration: they
// are the claim the without-fix arm in the PR is measured against.
//
//  arm 1  registration      [G]  the NID is registered at all.
//  arm 2  argument refusal  [D]  null / "/savedata1" / short / unterminated-16 mount points are
//                                refused. [G] for the null OUTPUT pointer, which the pre-fix code
//                                already rejected.
//  arm 3  lifecycle         [D]  a valid mount point with nothing mounted, and the same mount point
//                                after an unmount, are NOT_MOUNTED. [G] that NOT_MOUNTED is a
//                                different code from the argument error — two constants compared,
//                                which no implementation change can redden.
//  arm 4  output policy     [D]  a refused call leaves the caller's 48 bytes byte-for-byte untouched
//                                (the pre-fix code memset them and reported success).
//                           [G]  a successful call writes EXACTLY 48 bytes — canaries on both sides —
//                                with blocks/freeBlocks at their ABI offsets and reserved[32] zeroed.
//  arm 5  mount sequence    [D]/[G] mount A, unmount A, mount B, unmount B: the answer follows the
//                                current mount at every step, and success does not depend on WHICH
//                                save is mounted.
//  arm 6  ABI constants     [G]  the 48-byte layout, pinned in one falsifiable place.
//
// Capacity ACCOUNTING is deliberately out of scope here and is issue #3654: this fixture asserts
// that the figures are present and self-consistent, never that they describe the host filesystem.
#include "hle/dispatch/dispatch.hpp"
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

// The ABI this implements, stated independently of the implementation's own constants so the test
// asserts the contract rather than restating whatever the code happens to do.
//   SceSaveDataMountInfo { u64 blocks; u64 freeBlocks; u8 reserved[32] } — 48 bytes.
constexpr size_t MOUNT_INFO_SIZE = 48;
constexpr size_t OFF_BLOCKS = 0, OFF_FREE_BLOCKS = 8, OFF_RESERVED = 16;
constexpr size_t RESERVED_SIZE = 32;
// The two codes this call has to be able to tell apart. Their exact values are MED confidence in the
// implementation; what this fixture pins is that they are DIFFERENT and that neither is SCE_OK.
constexpr uint64_t ERR_PARAMETER = 0x809F0000ull;
constexpr uint64_t ERR_NOT_MOUNTED = 0x809F0004ull;

// SceSaveDataMountPoint { char data[16] }.
struct MountPoint { char data[16]; };
MountPoint mount_point(const char* path) {
    MountPoint mp{};
    snprintf(mp.data, sizeof mp.data, "%s", path);
    return mp;
}

// The output buffer with 16 canary bytes on each side. A write of more (or fewer) than the 48 bytes
// the ABI defines shows up as a damaged canary rather than as silent corruption of a guest struct
// the real caller has other fields in.
constexpr uint8_t CANARY = 0xA5;
constexpr uint8_t POISON = 0x5A;
// What a poisoned buffer reads as when interpreted as one of the ABI's u64 fields, so "the call
// wrote a real number here" is distinguishable from "the call wrote nothing here".
constexpr uint64_t POISON_WORD = 0x5A5A5A5A5A5A5A5Aull;
struct GuardedInfo {
    uint8_t lead[16];
    uint8_t info[MOUNT_INFO_SIZE];
    uint8_t trail[16];

    explicit GuardedInfo(uint8_t fill) {
        memset(lead, CANARY, sizeof lead);
        memset(info, fill, sizeof info);
        memset(trail, CANARY, sizeof trail);
    }
    bool canaries_intact() const {
        for (uint8_t b : lead) if (b != CANARY) return false;
        for (uint8_t b : trail) if (b != CANARY) return false;
        return true;
    }
    bool info_is_all(uint8_t value) const {
        for (uint8_t b : info) if (b != value) return false;
        return true;
    }
    uint64_t at(size_t offset) const {
        uint64_t v = 0;
        memcpy(&v, info + offset, sizeof v);
        return v;
    }
};

HleFn g_mount_info = nullptr, g_mount3 = nullptr, g_umount = nullptr;

uint64_t get_mount_info(const void* mp, void* info) {
    return g_mount_info((uint64_t)(uintptr_t)mp, (uint64_t)(uintptr_t)info, 0, 0, 0, 0);
}

// sceSaveDataMount3(const Mount3* mount, MountResult* result): dirName @+0x08, mountMode @+0x20.
// mountMode 0x20 is OPEN-OR-CREATE, 0x04 exclusive CREATE, neither bit set is OPEN.
struct Mount3Desc { uint8_t bytes[0x30]; };
struct MountResult { uint8_t bytes[0x40]; };
uint64_t mount_save(const char* dirname, uint32_t mode, MountResult& result) {
    Mount3Desc desc{};
    memset(desc.bytes, 0, sizeof desc.bytes);
    const char* name = dirname;
    memcpy(desc.bytes + 0x08, &name, sizeof name);
    memcpy(desc.bytes + 0x20, &mode, sizeof mode);
    memset(result.bytes, 0, sizeof result.bytes);
    return g_mount3((uint64_t)(uintptr_t)&desc, (uint64_t)(uintptr_t)&result, 0, 0, 0, 0);
}
uint64_t umount_save(const MountPoint& mp) {
    return g_umount((uint64_t)(uintptr_t)&mp, 0, 0, 0, 0, 0);
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

}   // namespace

int main() {
    printf("== test_savedata_mountinfo ==\n");
    register_builtin_hle();

    // ------------------------------------------------------------------------ arm 1: registration
    g_mount_info = Hle::lookup("65VH0Qaaz6s");
    g_mount3 = Hle::lookup("ZP4e7rlzOUk");
    g_umount = Hle::lookup("BMR4F-Uek3E");
    CHECK(g_mount_info != nullptr, "[G] sceSaveDataGetMountInfo is registered");
    CHECK(g_mount3 != nullptr, "[G] sceSaveDataMount3 is registered");
    CHECK(g_umount != nullptr, "[G] sceSaveDataUmount is registered");
    if (!g_mount_info || !g_mount3 || !g_umount) {
        printf("== FAIL: %d == (%d assertions executed)\n", fails, checks);
        return 1;
    }

    // A disposable save root: this fixture never touches a game dump and never touches the
    // developer's real saves.
    const fs::path scratch = prosper_test::test_scratch_dir() / "savedata-mountinfo";
    std::error_code ec;
    fs::remove_all(scratch, ec);
    fs::create_directories(scratch, ec);
    const fs::path save_root = scratch / "save0";
    fs::create_directories(save_root, ec);
    set_env("PROSPER_SAVE0", save_root.string().c_str());
    set_app0_root(make_app0(scratch, "title", "PPSA00042"));

    const MountPoint mp = mount_point("/savedata0");

    // -------------------------------------------------------------------- arm 3a: nothing mounted
    // Runs before any mount, and before the argument arms, because it is the single assertion the
    // whole issue is about: on the unfixed build this returns SCE_OK with a capacity report.
    savedata0_umount();
    {
        GuardedInfo g(POISON);
        const uint64_t rc = get_mount_info(&mp, g.info);
        CHECK(rc == ERR_NOT_MOUNTED,
              "[D] a valid mount point with NOTHING mounted returns NOT_MOUNTED");
        CHECK(rc != 0, "[D] ...and in particular does not report success");
        CHECK(g.info_is_all(POISON) && g.canaries_intact(),
              "[D] ...and leaves the caller's 48-byte buffer byte-for-byte untouched");
    }

    // ---------------------------------------------------------------------- arm 2: argument refusal
    // Each of these must be refused whether or not a save is mounted, so the arm is run twice: once
    // with nothing mounted (here) and once with a live mount (after arm 4). A check that only held
    // in the unmounted state would be the lifecycle answer wearing an argument error's clothes.
    auto argument_arms = [&](const char* when) {
        char label[160];
        auto say = [&](const char* what) {
            snprintf(label, sizeof label, "%s (%s)", what, when);
            return label;
        };
        GuardedInfo g(POISON);
        CHECK(get_mount_info(nullptr, g.info) == ERR_PARAMETER,
              say("[D] a NULL mount point is a parameter error"));
        const MountPoint other = mount_point("/savedata1");
        CHECK(get_mount_info(&other, g.info) == ERR_PARAMETER,
              say("[D] a mount point this build does not serve is a parameter error"));
        const MountPoint truncated = mount_point("/savedata");
        CHECK(get_mount_info(&truncated, g.info) == ERR_PARAMETER,
              say("[D] a mount point that is a PREFIX of the real one is a parameter error"));
        // Exactly 16 bytes, no terminator anywhere, heap-allocated so the allocation ends right
        // after the field: under a sanitizer build an unbounded compare also reddens here, and
        // without one the mismatch alone still has to be detected.
        std::vector<char> unterminated(16, 'x');
        memcpy(unterminated.data(), "/savedata0AAAAAA", 16);
        CHECK(get_mount_info(unterminated.data(), g.info) == ERR_PARAMETER,
              say("[D] an UNTERMINATED 16-byte mount-point field is a parameter error"));
        // The pre-fix code already rejected this one, so it is a guard: it is here to stop the null
        // check being lost while the new checks are added in front of it.
        CHECK(get_mount_info(&mp, nullptr) == ERR_PARAMETER,
              say("[G] a NULL output pointer is a parameter error"));
        CHECK(g.info_is_all(POISON) && g.canaries_intact(),
              say("[D] no refused call wrote anything into the caller's buffer"));
    };
    argument_arms("nothing mounted");

    // ------------------------------------------------------------ arm 4: the success-path contract
    MountResult result{};
    CHECK(mount_save("SlotA", 0x20, result) == 0, "[G] a save is created and mounted");
    // What a real title passes to GetMountInfo is the mount point Mount3 wrote into its result, not
    // a literal it made up, so the arm below uses exactly that.
    const MountPoint from_result = *(const MountPoint*)result.bytes;
    CHECK(strncmp(from_result.data, "/savedata0", 16) == 0,
          "[G] Mount3 reported the mount point this build serves");
    {
        GuardedInfo g(POISON);
        CHECK(get_mount_info(&from_result, g.info) == 0,
              "[G] GetMountInfo on the LIVE mount succeeds");
        CHECK(g.canaries_intact(),
              "[G] ...writing exactly 48 bytes -- neither canary was touched");
        CHECK(g.at(OFF_BLOCKS) != 0 && g.at(OFF_BLOCKS) != POISON_WORD,
              "[G] ...with a total block count at its ABI offset");
        CHECK(g.at(OFF_FREE_BLOCKS) != 0 && g.at(OFF_FREE_BLOCKS) <= g.at(OFF_BLOCKS),
              "[G] ...and a free block count that does not exceed it");
        bool reserved_zero = true;
        for (size_t i = 0; i < RESERVED_SIZE; ++i)
            if (g.info[OFF_RESERVED + i] != 0) reserved_zero = false;
        CHECK(reserved_zero, "[G] ...and reserved[32] cleared, as the pre-fix code also did");
    }
    // The argument checks must still refuse while a save IS mounted: otherwise "refused" could be
    // the lifecycle answer rather than the argument answer.
    argument_arms("a save is mounted");

    // ---------------------------------------------------------------------- arm 3b: after unmount
    {
        CHECK(umount_save(from_result) == 0, "[G] the save is unmounted through sceSaveDataUmount");
        GuardedInfo g(POISON);
        const uint64_t rc = get_mount_info(&from_result, g.info);
        CHECK(rc == ERR_NOT_MOUNTED,
              "[D] the SAME mount point returns NOT_MOUNTED once the save is unmounted");
        CHECK(g.info_is_all(POISON) && g.canaries_intact(),
              "[D] ...without writing a capacity report for a mount that is gone");
    }
    // The two refusals are different facts and must be different codes: a title that cannot tell
    // "your argument is wrong" from "nothing is mounted" cannot recover from either. Tagged [G]
    // because it compares two constants and nothing else — it cannot fail against the unfixed
    // handler, and it is here to state the requirement the two arms above depend on.
    CHECK(ERR_PARAMETER != ERR_NOT_MOUNTED && ERR_PARAMETER != 0 && ERR_NOT_MOUNTED != 0,
          "[G] the argument error and the lifecycle error are distinct, and neither is SCE_OK");

    // ------------------------------------------------------------------- arm 5: a mount SEQUENCE
    // Mount A, unmount A, mount B, unmount B. Success must track the mount that is current, not the
    // fact that a mount once happened.
    {
        MountResult a{}, b{};
        GuardedInfo g(POISON);
        CHECK(mount_save("SlotA", 0x20, a) == 0, "[G] SlotA is mounted again");
        CHECK(get_mount_info(&mp, g.info) == 0, "[G] ...and is reported as mounted");
        CHECK(umount_save(mp) == 0, "[G] SlotA is unmounted");
        CHECK(get_mount_info(&mp, g.info) == ERR_NOT_MOUNTED,
              "[D] ...after which the mount point is NOT_MOUNTED");
        CHECK(mount_save("SlotB", 0x20, b) == 0, "[G] a DIFFERENT save, SlotB, is mounted");
        CHECK(get_mount_info(&mp, g.info) == 0,
              "[G] ...and the mount point is live again: success does not depend on which save");
        CHECK(umount_save(mp) == 0, "[G] SlotB is unmounted");
        CHECK(get_mount_info(&mp, g.info) == ERR_NOT_MOUNTED,
              "[D] ...and the mount point is NOT_MOUNTED once more");
        CHECK(g.canaries_intact() && g.at(OFF_BLOCKS) != POISON_WORD,
              "[G] across the whole sequence nothing wrote outside the 48 bytes, and the one buffer "
              "still holds what the last SUCCESSFUL call put there");
    }

    // ---------------------------------------------------------------- arm 6: the ABI layout itself
    // Pinned in one falsifiable place: a live capture that contradicts an offset reddens a named arm
    // instead of silently disagreeing with the guest.
    CHECK(OFF_BLOCKS == 0 && OFF_FREE_BLOCKS == 8 && OFF_RESERVED == 16 &&
              OFF_RESERVED + RESERVED_SIZE == MOUNT_INFO_SIZE && MOUNT_INFO_SIZE == 48,
          "[G] SceSaveDataMountInfo: blocks@0, freeBlocks@8, reserved[32]@16, 48 bytes");

    savedata0_umount();
    if (fails) { printf("== FAIL: %d == (%d assertions executed)\n", fails, checks); return 1; }
    printf("== PASS == (%d assertions executed)\n", checks);
    return 0;
}
