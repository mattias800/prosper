// test_avplayer (#324) — libSceAvPlayer lifecycle. prosper doesn't decode video, but the two init
// entry points have DIFFERENT ABIs and must be distinguished, or the guest's video wrapper misreads the
// result. sceAvPlayerInit RETURNS the handle; sceAvPlayerInitEx returns an int32 error code (0 = success)
// and WRITES the handle to its out-param. Registering InitEx to the return-the-handle handler made
// PPSA02664's PS5VideoPlayback wrapper read a non-zero handle as an error code and abort the intro video
// ("[PS5VideoPlayback] ERROR: sceAvPlayerInitEx() failed"). This locks the InitEx out-param contract.
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include <cstdio>
#include <cstdint>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

int main() {
    printf("== test_avplayer ==\n");
    register_builtin_hle();

    HleFn init   = Hle::lookup(nid_hash("sceAvPlayerInit"));
    HleFn initex = Hle::lookup(nid_hash("sceAvPlayerInitEx"));
    HleFn active = Hle::lookup(nid_hash("sceAvPlayerIsActive"));
    CHECK(init && initex && active, "AvPlayer Init/InitEx/IsActive registered");
    if (!(init && initex && active)) { printf("== FAIL ==\n"); return 1; }

    // sceAvPlayerInit(data) -> handle (the RETURN value is the handle; non-NULL so the game proceeds).
    uint64_t h1 = init(0, 0, 0, 0, 0, 0);
    CHECK(h1 != 0, "sceAvPlayerInit returns a non-NULL handle");

    // sceAvPlayerInitEx(data, out) -> 0 (success error-code), handle written to *out. It must NOT return
    // the handle as its result — that is the regression this test guards.
    uint64_t out = 0xDEAD;
    uint64_t rc = initex(0, (uint64_t)(uintptr_t)&out, 0, 0, 0, 0);
    CHECK(rc == 0, "sceAvPlayerInitEx returns 0 (success error code), NOT the handle");
    CHECK(out != 0 && out != 0xDEAD, "sceAvPlayerInitEx WROTE a valid non-NULL handle to the out-param");

    // IsActive must report "not active" so the game's while(IsActive) playback loop ends and it advances.
    CHECK(active(out, 0, 0, 0, 0, 0) == 0, "sceAvPlayerIsActive -> 0 (finished, proceed)");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
