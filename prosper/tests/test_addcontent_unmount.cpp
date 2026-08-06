// test_addcontent_unmount — sceAppContentAddcontUnmount releases the claim sceAppContentAddcontMount
// takes, so a mount/unmount/re-mount cycle works (#2004).
//
// The bug this guards is a FALSE SUCCESS (#2081): the unmount NID was unregistered, so it reached
// the dispatcher's `return 0` — SCE_OK — without clearing `InstalledAddcontent::mounted`. The guest
// was told the release succeeded and every later mount of the same add-content returned BUSY,
// forever, with no way for the title to recover. Content that IS present locally became permanently
// unreachable, which is the under-reporting half of the local-inventory rule.
//
// The load-bearing arm is therefore the THIRD call in the cycle, not the second: an unmount that
// returns 0 proves nothing on its own, because the unregistered stub returned 0 too. Only the
// re-mount separates "released the claim" from "said it did".
#include "../src/hle/dispatch.hpp"
#include "../src/hle/hle_addcontent.hpp"
#include "../src/hle/nid.hpp"
#include "test_scratch.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

using namespace prosper;
namespace fs = std::filesystem;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

namespace {
constexpr const char* kAddcontMount   = "VANhIWcqYak";  // sceAppContentAddcontMount
constexpr const char* kAddcontUnmount = "3rHWaV-1KC4";  // sceAppContentAddcontUnmount (PS5 3.20)
constexpr uint64_t kErrBusy      = 0x80D90003ull;
constexpr uint64_t kErrNotFound  = 0x80D90005ull;
constexpr uint64_t kErrParameter = 0x80D90002ull;
} // namespace

int main() {
    printf("== test_addcontent_unmount ==\n");
    register_builtin_hle();

    // A minimal local inventory: one installed, mountable PSAC entry at /app0/dlc1.
    const fs::path scratch = prosper_test::test_scratch_dir() / "addcont-unmount";
    const fs::path app0 = scratch / "app0";
    std::error_code ec;
    fs::remove_all(scratch, ec);
    fs::create_directories(app0 / "sce_sys", ec);
    fs::create_directories(app0 / "dlc1", ec);
    { std::ofstream p(app0 / "sce_sys" / "param.json", std::ios::binary);
      p << R"({"titleId":"PPSA00001"})"; }
    { std::ofstream m(app0 / "dlc_emu.ini", std::ios::binary);
      m << "[PSAC]\n"
           "content_id=UP0000-PPSA00001_00-00000000000DLC01\n"
           "download_status=INSTALLED\n"
           "mount_point=/app0/dlc1\n"; }
    set_app0_root(app0.string());

    HleFn mount = Hle::lookup(kAddcontMount);
    HleFn unmount = Hle::lookup(kAddcontUnmount);
    CHECK(mount != nullptr, "sceAppContentAddcontMount is registered");
    // Kills: leaving the NID unregistered — the bug itself. Unregistered, Hle::lookup is null and
    // the guest would have reached the dispatcher's SCE_OK.
    CHECK(unmount != nullptr, "sceAppContentAddcontUnmount is registered");
    if (!mount || !unmount) { printf("== FAIL: %d ==\n", fails); return 1; }

    // The entitlement label the manifest declares, as a 20-byte SceNpUnifiedEntitlementLabel.
    char label[20]; memset(label, 0, sizeof(label));
    memcpy(label, "00000000000DLC01", 16);

    char mp[16]; memset(mp, 0xAB, sizeof(mp));
    uint64_t r = mount(0, (uint64_t)(uintptr_t)label, (uint64_t)(uintptr_t)mp, 0, 0, 0);
    CHECK(r == 0, "first mount succeeds");
    CHECK(strncmp(mp, "/app0/dlc1", 10) == 0, "mount writes the declared mount point");

    // Precondition for the whole test: the claim really is exclusive. Without this, the re-mount arm
    // below could pass simply because mounting was never refused in the first place.
    char mp2[16]; memset(mp2, 0xAB, sizeof(mp2));
    CHECK(mount(0, (uint64_t)(uintptr_t)label, (uint64_t)(uintptr_t)mp2, 0, 0, 0) == kErrBusy,
          "a second mount of a claimed entry is BUSY");

    CHECK(unmount((uint64_t)(uintptr_t)mp, 0, 0, 0, 0, 0) == 0, "unmount reports success");

    // THE load-bearing arm. An unmount returning 0 is exactly what the unregistered stub did, so
    // only a successful re-mount shows the claim was actually released.
    // Kills: registering the NID to a bare `return 0` no-op, which would satisfy every arm above.
    char mp3[16]; memset(mp3, 0xAB, sizeof(mp3));
    CHECK(mount(0, (uint64_t)(uintptr_t)label, (uint64_t)(uintptr_t)mp3, 0, 0, 0) == 0,
          "re-mount after unmount succeeds (the claim was really released)");
    CHECK(strncmp(mp3, "/app0/dlc1", 10) == 0, "re-mount writes the declared mount point again");

    // Unmounting something that is not currently claimed must FAIL rather than be waved through.
    // Kills: an implementation that clears state for any input, or ignores its argument entirely
    // and frees whatever is mounted — which would pass the cycle above while releasing the wrong
    // claim in a title with several add-contents.
    char bogus[16]; memset(bogus, 0, sizeof(bogus));
    memcpy(bogus, "/app0/nope", 10);
    CHECK(unmount((uint64_t)(uintptr_t)bogus, 0, 0, 0, 0, 0) == kErrNotFound,
          "unmounting an unknown mount point reports NOT_FOUND");
    // The real entry must still be mounted after that failed call.
    char mp4[16]; memset(mp4, 0xAB, sizeof(mp4));
    CHECK(mount(0, (uint64_t)(uintptr_t)label, (uint64_t)(uintptr_t)mp4, 0, 0, 0) == kErrBusy,
          "a failed unmount released nothing");

    // Double unmount: the second one has nothing to release and must say so.
    CHECK(unmount((uint64_t)(uintptr_t)mp, 0, 0, 0, 0, 0) == 0, "unmount of the live claim succeeds");
    CHECK(unmount((uint64_t)(uintptr_t)mp, 0, 0, 0, 0, 0) == kErrNotFound,
          "a second unmount of the same point reports NOT_FOUND");

    // A non-pointer argument is a parameter error, not a fault and not a success.
    CHECK(unmount(0, 0, 0, 0, 0, 0) == kErrParameter, "a null mount point reports PARAMETER");

    fs::remove_all(scratch, ec);
    if (fails) { printf("== FAIL: %d check(s) failed ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
