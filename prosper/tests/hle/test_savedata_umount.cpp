// test_savedata_umount (#3666) -- the two umount entry points must read what they were given.
//
// THE DEFECT, in two halves.
//
// sceSaveDataUmount2 (uW4vfTwMQVo) read NONE of its arguments: it unmounted /savedata0, queued a
// completion event and returned SCE_OK whatever it was handed. A title passing a mount point this
// build does not serve had a LIVE MOUNT DISCARDED on its behalf and was told it succeeded. That is
// worse than #3653's GetMountInfo, which only misreported -- this one mutates.
//
// sceSaveDataUmount (BMR4F-Uek3E) did read its mount point, through its own inline copy of the
// spelling check, but answered NOT_FOUND both for a bad argument and for "nothing is mounted",
// where its four siblings answer PARAMETER and NOT_MOUNTED. One entry point collapsed two different
// facts into a code that means neither.
//
// THE ARGUMENT ORDER IS THE FINDING THIS FIXTURE EXISTS TO PIN. Umount2's mount point is its
// SECOND argument; the first is a flags word. Established by live capture (Dead Cells PPSA15552 on
// its own snapshot route under PROSPER_SVCLOG=1 -- three calls, a1 always "/savedata0") and by
// guest disassembly in four independent titles, where the pointer handed to Umount2 is the
// MountResult a mount call had just filled in. Reading it as (mountPoint, ...) by analogy with
// Umount would validate the flags word as a pointer and refuse every correct call -- so arm 2
// asserts the order directly, in both directions.
//
// Every arm goes through the REGISTERED NIDs, never the internal helpers: a test that called
// savedata0_umount() directly would pass against a build where the guest still reaches the
// dispatcher's `return 0`.
//
// WHICH ARMS ARE DISCRIMINATORS AND WHICH ARE GUARDS. [D] fails when the fix is reverted; [G]
// passes either way and is present so a later change cannot quietly break what already worked.
//
//  arm 1  registration        [G]  both NIDs, and GetEventResult, are registered.
//  arm 2  Umount2 arguments   [D]  a bad mount point in a1 is refused AND LEAVES THE MOUNT LIVE;
//                                  the mount point is not read from a0.
//  arm 3  Umount2 lifecycle   [D]  valid mount point, nothing mounted -> NOT_MOUNTED, not SCE_OK.
//  arm 4  Umount2 flags       [G]  a0 is deliberately NOT validated: every observed flags value,
//                                  and an arbitrary one, behaves identically. Documents the cap.
//  arm 5  completion events   [D]  a REFUSED umount queues no completion event; a successful one
//                                  queues exactly one.
//  arm 6  Umount arguments    [D]  PARAMETER (not NOT_FOUND) for a bad mount point. [G] for the
//                                  null case, which the pre-fix code already rejected.
//  arm 7  Umount lifecycle    [D]  NOT_MOUNTED (not NOT_FOUND) when nothing is mounted.
//  arm 8  the two agree       [D]  same argument class, same answer, through both entry points,
//                                  and neither of them mutates state on the way to the answer.
//  arm 9  codes are distinct  [G]  two constants compared -- it cannot redden against the unfixed
//                                  handler, and says so. (#3665 shipped this arm mis-tagged [D].)
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

// The contract, stated independently of the implementation's own constants.
constexpr uint64_t ERR_PARAMETER   = 0x809F0000ull;
constexpr uint64_t ERR_NOT_MOUNTED = 0x809F0004ull;
constexpr uint64_t ERR_NO_EVENT    = 0x809F0008ull;   // GetEventResult, queue drained
// The value the pre-fix Umount answered for BOTH error classes. Named so arm 6/7 can say what they
// are ruling out rather than only what they expect.
constexpr uint64_t ERR_NOT_FOUND   = 0x809F0008ull;

// SceSaveDataMountPoint { char data[16] }.
struct MountPoint { char data[16]; };
MountPoint mount_point(const char* path) {
    MountPoint mp{};
    snprintf(mp.data, sizeof mp.data, "%s", path);
    return mp;
}

// sceSaveDataGetEventResult's event: { u32 type; s32 errorCode; s32 userId; ... } -- 104 bytes.
constexpr size_t EVENT_SIZE = 104;
constexpr uint32_t EVENT_TYPE_UMOUNT_BACKUP = 1;

HleFn g_mount3 = nullptr, g_umount = nullptr, g_umount2 = nullptr, g_get_event = nullptr;

// sceSaveDataMount3(const Mount3* mount, MountResult* result): dirName @+0x08, mountMode @+0x20.
struct Mount3Desc { uint8_t bytes[0x30]; };
struct MountResult { uint8_t bytes[0x40]; };
uint64_t mount_save(const char* dirname, MountResult& result) {
    Mount3Desc desc{};
    memset(desc.bytes, 0, sizeof desc.bytes);
    const char* name = dirname;
    const uint32_t mode = 0x20;                    // open-or-create
    memcpy(desc.bytes + 0x08, &name, sizeof name);
    memcpy(desc.bytes + 0x20, &mode, sizeof mode);
    memset(result.bytes, 0, sizeof result.bytes);
    return g_mount3((uint64_t)(uintptr_t)&desc, (uint64_t)(uintptr_t)&result, 0, 0, 0, 0);
}
uint64_t umount(const void* mp) {
    return g_umount((uint64_t)(uintptr_t)mp, 0, 0, 0, 0, 0);
}
// The shape this change establishes: flags first, mount point second.
uint64_t umount2(uint64_t flags, const void* mp) {
    return g_umount2(flags, (uint64_t)(uintptr_t)mp, 0, 0, 0, 0);
}
uint64_t get_event(void* event) {
    return g_get_event(0, (uint64_t)(uintptr_t)event, 0, 0, 0, 0);
}
// Drain whatever the queue holds, so an arm's event count is its own. Bounded: a drain loop that
// could not terminate would hang the suite rather than fail it.
int drain_events() {
    uint8_t event[EVENT_SIZE];
    for (int i = 0; i < 64; ++i) {
        memset(event, 0, sizeof event);
        if (get_event(event) != 0) return i;
    }
    return -1;
}
bool mounted() { return !savedata0_mounted_dir().empty(); }

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
    printf("== test_savedata_umount ==\n");
    register_builtin_hle();

    // ------------------------------------------------------------------------ arm 1: registration
    g_mount3    = Hle::lookup("ZP4e7rlzOUk");
    g_umount    = Hle::lookup("BMR4F-Uek3E");
    g_umount2   = Hle::lookup("uW4vfTwMQVo");
    g_get_event = Hle::lookup("j8xKtiFj0SY");
    CHECK(g_mount3 != nullptr, "[G] sceSaveDataMount3 is registered");
    CHECK(g_umount != nullptr, "[G] sceSaveDataUmount is registered");
    CHECK(g_umount2 != nullptr, "[G] sceSaveDataUmount2 is registered");
    CHECK(g_get_event != nullptr, "[G] sceSaveDataGetEventResult is registered");
    if (!g_mount3 || !g_umount || !g_umount2 || !g_get_event) {
        printf("== FAIL: %d == (%d assertions executed)\n", fails, checks);
        return 1;
    }

    // A disposable save root: this fixture never touches a game dump or a developer's real saves.
    const fs::path scratch = prosper_test::test_scratch_dir() / "savedata-umount";
    std::error_code ec;
    fs::remove_all(scratch, ec);
    fs::create_directories(scratch, ec);
    const fs::path save_root = scratch / "save0";
    fs::create_directories(save_root, ec);
    set_env("PROSPER_SAVE0", save_root.string().c_str());
    set_app0_root(make_app0(scratch, "title", "PPSA00042"));

    const MountPoint mp          = mount_point("/savedata0");
    const MountPoint other       = mount_point("/savedata1");
    const MountPoint truncated   = mount_point("/savedata");
    // Exactly 16 bytes with no terminator, heap-allocated so the allocation ends right after the
    // field: an unbounded compare reddens under a sanitizer build, and without one the mismatch
    // still has to be detected.
    std::vector<char> unterminated(16, 'x');
    memcpy(unterminated.data(), "/savedata0AAAAAA", 16);

    savedata0_umount();
    drain_events();

    // ------------------------------------------------------- arm 2: Umount2's arguments and order
    // Each refusal is checked while a save IS mounted, because the assertion that matters is not
    // the return value -- it is that the mount SURVIVES. On the unfixed build every one of these
    // discards the live mount and reports success.
    MountResult result{};
    CHECK(mount_save("SlotA", result) == 0, "[G] a save is created and mounted");
    CHECK(mounted(), "[G] ...and the mount is live");
    // What a real title passes is the mount point the mount call wrote into its result.
    const MountPoint from_result = *(const MountPoint*)result.bytes;
    CHECK(strncmp(from_result.data, "/savedata0", 16) == 0,
          "[G] Mount3 reported the mount point this build serves");

    CHECK(umount2(1, nullptr) == ERR_PARAMETER,
          "[D] Umount2 with a NULL mount point is a parameter error");
    CHECK(mounted(), "[D] ...and the live mount is still mounted");
    CHECK(umount2(1, &other) == ERR_PARAMETER,
          "[D] Umount2 with a mount point this build does not serve is a parameter error");
    CHECK(mounted(), "[D] ...and /savedata0 was NOT discarded on its behalf");
    CHECK(umount2(1, &truncated) == ERR_PARAMETER,
          "[D] Umount2 with a PREFIX of the real mount point is a parameter error");
    CHECK(mounted(), "[D] ...and the mount survives that too");
    CHECK(umount2(1, unterminated.data()) == ERR_PARAMETER,
          "[D] Umount2 with an UNTERMINATED 16-byte mount-point field is a parameter error");
    CHECK(mounted(), "[D] ...and the mount survives that too");

    // The order itself, asserted in both directions. Passing the mount point where the FLAGS go --
    // the reading a1-blind fix would have taken -- must be refused, because a1 is then not a mount
    // point. This is the arm that reddens if someone later "simplifies" the handler to read a0.
    CHECK(umount2((uint64_t)(uintptr_t)&mp, 0) == ERR_PARAMETER,
          "[D] a mount point in a0 with nothing in a1 is refused: a0 is NOT the mount point");
    CHECK(mounted(), "[D] ...and that call did not unmount anything either");

    // And the positive direction: the same pointer in a1 with a plain flags word in a0 succeeds.
    // [G], and the tag was MEASURED rather than reasoned: the unfixed handler returned 0 for
    // everything, so a correct call succeeding cannot distinguish the two builds. It is here so a
    // future validator cannot start refusing the one call shape every title actually makes.
    CHECK(umount2(1, &from_result) == 0,
          "[G] Umount2(flags, mountPoint) with the mount point Mount3 reported succeeds");
    CHECK(!mounted(), "[G] ...and the mount is gone");

    // ------------------------------------------------------------------ arm 3: Umount2 lifecycle
    CHECK(umount2(1, &mp) == ERR_NOT_MOUNTED,
          "[D] Umount2 on a valid mount point with NOTHING mounted returns NOT_MOUNTED");
    CHECK(umount2(1, &mp) != 0,
          "[D] ...and in particular does not report an unmount that did not happen");

    // -------------------------------------------------------------- arm 4: the flags word is free
    // a0's bit meanings are NOT established, so nothing is rejected on them. Live capture observed
    // 0x1 and 0x10001; static disassembly observed 0x0. An arbitrary value must behave the same, so
    // that a later flags discovery is a behaviour to ADD rather than a refusal to unpick.
    for (uint64_t flags : {0ull, 1ull, 0x10001ull, 0xdeadbeefull}) {
        MountResult r{};
        char label[128];
        snprintf(label, sizeof label, "[G] Umount2 accepts flags=%#llx unchanged",
                 (unsigned long long)flags);
        CHECK(mount_save("SlotF", r) == 0, "[G] a save is mounted for the flags arm");
        CHECK(umount2(flags, &mp) == 0, label);
    }

    // ------------------------------------------------------------- arm 5: the completion event
    // A refused call starts no operation, so it completes none. The pre-fix handler bumped the
    // counter unconditionally, handing GetEventResult a fabricated UMOUNT_BACKUP completion for an
    // unmount that never happened.
    {
        const int drained = drain_events();
        CHECK(drained >= 0, "[G] the event queue drains in bounded time");
        uint8_t event[EVENT_SIZE];
        memset(event, 0, sizeof event);
        CHECK(get_event(event) == ERR_NO_EVENT, "[G] ...and is then empty");

        CHECK(!mounted(), "[G] nothing is mounted");
        CHECK(umount2(1, &mp) == ERR_NOT_MOUNTED, "[D] a lifecycle-refused Umount2 fails");
        memset(event, 0, sizeof event);
        CHECK(get_event(event) == ERR_NO_EVENT,
              "[D] ...and queued NO completion event for an unmount that did not happen");

        MountResult r{};
        CHECK(mount_save("SlotE", r) == 0, "[G] a save is mounted");
        CHECK(umount2(1, &other) == ERR_PARAMETER, "[D] an argument-refused Umount2 fails");
        memset(event, 0, sizeof event);
        CHECK(get_event(event) == ERR_NO_EVENT,
              "[D] ...and queued no completion event either");

        CHECK(umount2(1, &mp) == 0, "[G] a real unmount succeeds");
        memset(event, 0, sizeof event);
        uint32_t type = 0;
        const uint64_t rc = get_event(event);
        memcpy(&type, event, sizeof type);
        CHECK(rc == 0 && type == EVENT_TYPE_UMOUNT_BACKUP,
              "[G] ...and DOES queue one UMOUNT_BACKUP completion");
        memset(event, 0, sizeof event);
        CHECK(get_event(event) == ERR_NO_EVENT, "[G] ...exactly one, not more");
    }

    // ------------------------------------------------------------------- arm 6: Umount's arguments
    // The pre-fix handler answered NOT_FOUND to each of these. Both halves are asserted: the code is
    // PARAMETER, and it is specifically no longer NOT_FOUND.
    {
        MountResult r{};
        CHECK(mount_save("SlotB", r) == 0, "[G] a save is mounted for the Umount arms");
        // Already rejected before the fix, so a guard: it stops the null check being lost while the
        // shared validator is put in front of it.
        CHECK(umount(nullptr) == ERR_PARAMETER, "[G] Umount with a NULL mount point is PARAMETER");
        CHECK(umount(&other) == ERR_PARAMETER,
              "[D] Umount with a mount point this build does not serve is PARAMETER");
        CHECK(umount(&other) != ERR_NOT_FOUND, "[D] ...and specifically NOT the old NOT_FOUND");
        CHECK(umount(&truncated) == ERR_PARAMETER,
              "[D] Umount with a PREFIX of the real mount point is PARAMETER");
        CHECK(umount(unterminated.data()) == ERR_PARAMETER,
              "[D] Umount with an UNTERMINATED 16-byte field is PARAMETER");
        CHECK(mounted(), "[G] no refused Umount discarded the live mount");
        CHECK(umount(&mp) == 0, "[G] Umount with the real mount point succeeds");
        CHECK(!mounted(), "[G] ...and the mount is gone");
    }

    // ------------------------------------------------------------------- arm 7: Umount's lifecycle
    CHECK(umount(&mp) == ERR_NOT_MOUNTED,
          "[D] Umount on a valid mount point with nothing mounted is NOT_MOUNTED");
    CHECK(umount(&mp) != ERR_NOT_FOUND,
          "[D] ...and specifically NOT the old NOT_FOUND, which also meant 'bad argument' here");

    // ------------------------------------------------- arm 8: both entry points answer alike
    // The defect this issue was filed for: one library's umount disagreeing with the other about
    // what an argument error is. Asserted as equality rather than against a literal, so a later
    // correction to the numeric codes cannot make the two drift apart again without reddening.
    {
        MountResult r{};
        CHECK(mount_save("SlotC", r) == 0, "[G] a save is mounted for the agreement arm");
        CHECK(umount(&other) == umount2(1, &other),
              "[D] both entry points answer a bad mount point the same way");
        CHECK(mounted(), "[D] ...and neither of them unmounted anything");
        // Re-establish the state explicitly instead of inheriting it from the arm above, so the
        // lifecycle comparison below fails for its OWN reason or not at all. (Written the other way
        // first: the follow-on unmount then reddened under the revert purely as collateral of the
        // discarded mount, which would have made a [G] look like a discriminator.)
        savedata0_umount();
        CHECK(!mounted(), "[G] nothing is mounted for the lifecycle comparison");
        CHECK(umount(&mp) == umount2(1, &mp),
              "[D] both entry points answer 'nothing is mounted' the same way");
    }

    // ---------------------------------------------------------------- arm 9: the codes are distinct
    // Tagged [G] deliberately: it compares two constants and nothing else, so it cannot redden
    // against the unfixed handler. It is here to state the requirement arms 2-8 depend on.
    CHECK(ERR_PARAMETER != ERR_NOT_MOUNTED && ERR_PARAMETER != 0 && ERR_NOT_MOUNTED != 0,
          "[G] the argument error and the lifecycle error are distinct, and neither is SCE_OK");

    savedata0_umount();
    drain_events();
    if (fails) { printf("== FAIL: %d == (%d assertions executed)\n", fails, checks); return 1; }
    printf("== PASS == (%d assertions executed)\n", checks);
    return 0;
}
