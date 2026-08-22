// test_savedata_event_drain — sceSaveDataGetEventResult must report a DRAINED event queue with the
// code the guest's own drain loop terminates on, and must never report the distinct "still in
// flight" code for a state in which nothing is in flight.
//
// THE TWO CODES ARE DIFFERENT THINGS, and both are established by shipping titles' bytes:
//
//   0x809F0018  "the operation is still running, keep waiting"  — four titles sleep and re-poll on
//               it: PPSA15552 Dead Cells (+0x173c9b0 and +0x173cda0), PPSA28061 Earthion (+0x12f6d),
//               PPSA03831 Sonic Frontiers (+0x18a2285), PPSA05325 Sonic Origins (+0x940385).
//   0x809F0008  "there is no event" — the same value as NOT_FOUND.
//
// prosper answered 0x809F0018 for a drained queue. Nothing here is ever in flight (every file
// operation completes synchronously), so that was a well-corroborated "still busy" returned in a
// state where nothing is busy — a permanent lie, and a guaranteed hang for any of those four.
//
// PPSA20447 (The First Berserker: Khazan) pins the drained code directly. Its game thread's loop,
// transcribed from the eboot:
//
//     eboot+0x796eb2c   jmp    0x796eb3d             ; loop ENTRY — it polls first, sleeps after
//     eboot+0x796eb38   call   0x1565790             ; FPlatformProcess::Sleep(float)
//     eboot+0x796eb41.. vmovups/mov                  ; zero a 104-byte SceSaveDataEvent (96 + 8)
//     eboot+0x796eb67   xor    edi,edi               ; eventParam = NULL
//     eboot+0x796eb6f   call   0x8eaf3d0             ; sceSaveDataGetEventResult(NULL, &event)
//     eboot+0x796eb7b   mov    r12d,eax              ; r12d IS the return value
//     eboot+0x796eb88   cmp    r12d,0x809f0008       ; <-- the ONLY value that ends the wait
//     eboot+0x796eb8f   jne    0x796eb30             ; anything else: sleep and poll again
//
// Earthion const-compares BOTH in consecutive instructions — 0x809F0018 -> sleep 50 ms and re-poll
// (+0x12f6d), then 0x809F0008 -> give up and return (+0x12f82) — so a title really can tell them
// apart and the assertions below keep them apart.
//
// Each assertion fails against a specific wrong implementation: the drain-code check kills any wrong
// constant including the old 0x809F0018; the "no path returns the in-flight code" check stops it
// being quietly reintroduced; the sign check pins the property guest sites actually gate on; and the
// queued-event checks kill "always answer drained", which would silently discard the completion
// Dead Cells consumes.
#include "hle/dispatch/dispatch.hpp"
#include <cstdint>
#include <cstdio>
#include <cstring>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); ++fails; } \
                         else std::printf("  [ok]   %s\n", m); } while (0)

// PPSA20447's loop, transcribed: it leaves the wait only when the call returns exactly this.
static constexpr uint64_t kKhazanDrainCode = 0x809F0008ull;
// The distinct code four other titles sleep-and-repoll on. Nothing in prosper is ever in flight, so
// no path here may return it.
static constexpr uint64_t kInFlightCode = 0x809F0018ull;

int main() {
    std::printf("== test_savedata_event_drain ==\n");
    register_service_hle();

    HleFn get_event = Hle::lookup("j8xKtiFj0SY");   // sceSaveDataGetEventResult
    HleFn umount2   = Hle::lookup("uW4vfTwMQVo");   // sceSaveDataUmount2
    CHECK(get_event != nullptr, "sceSaveDataGetEventResult is registered");
    CHECK(umount2 != nullptr, "sceSaveDataUmount2 is registered");
    if (!get_event || !umount2) { std::printf("FAILED (%d)\n", fails + 1); return 1; }

    // Guest pointers are host pointers in this build (hle_service.cpp's PW() is a cast), so a plain
    // buffer is a valid `SceSaveDataEvent*`. 104 bytes is the size the API writes; the guard bytes
    // catch a handler that writes past it.
    uint8_t buf[104 + 16];
    auto arm = [&] { std::memset(buf, 0xCD, sizeof buf); };
    const uint64_t ev = (uint64_t)(uintptr_t)buf;

    // Drain whatever an earlier case in this process left queued, so the state is known.
    for (int i = 0; i < 64; i++) { arm(); if (get_event(0, ev, 0, 0, 0, 0) == kKhazanDrainCode) break; }

    // 1. Empty queue -> the code the guest's `cmp/jne` accepts. This one assertion catches ANY wrong
    //    constant, including the old 0x809F0018; the two that follow test different facts.
    arm();
    const uint64_t drained = get_event(0, ev, 0, 0, 0, 0);
    CHECK(drained == kKhazanDrainCode, "an empty queue answers 0x809F0008");
    //    The property guest sites gate on beyond the exact value: every error here is negative as
    //    int32, which is how a caller that only checks the sign reads it.
    CHECK((int32_t)(uint32_t)drained < 0, "the drained code is negative as int32");
    //    A drained queue must never claim an operation is still running. Separate fact, separate
    //    failure mode: this is what stops the in-flight code being reintroduced by a future edit.
    CHECK(drained != kInFlightCode,
          "a drained queue never answers the in-flight code 0x809F0018");

    // 2. A queued completion is still delivered, and delivered BEFORE the drain answer — otherwise
    //    this change would trade one title's boot for the completion Dead Cells consumes.
    umount2(0, 0, 0, 0, 0, 0);
    arm();
    const uint64_t got = get_event(0, ev, 0, 0, 0, 0);
    CHECK(got == 0, "a queued umount completion is reported as success");
    CHECK(*(const uint32_t*)(buf + 0) == 1u, "  its type is UMOUNT_BACKUP (1)");
    CHECK(*(const int32_t*)(buf + 8) == 1, "  its userId is the initial user");
    CHECK(buf[104] == 0xCD && buf[104 + 15] == 0xCD, "  exactly 104 bytes are written");

    // 3. One queued event is consumed once; the queue then reads drained again.
    arm();
    CHECK(get_event(0, ev, 0, 0, 0, 0) == kKhazanDrainCode,
          "the queue reads drained again once the event has been consumed");

    // 4. Two queued events are both delivered before the drain answer returns.
    umount2(0, 0, 0, 0, 0, 0);
    umount2(0, 0, 0, 0, 0, 0);
    arm(); CHECK(get_event(0, ev, 0, 0, 0, 0) == 0, "first of two queued events delivered");
    arm(); CHECK(get_event(0, ev, 0, 0, 0, 0) == 0, "second of two queued events delivered");
    arm(); CHECK(get_event(0, ev, 0, 0, 0, 0) == kKhazanDrainCode, "then drained");

    // 5. The null-event-pointer contract is unchanged and is NOT the drain code.
    const uint64_t bad = get_event(0, 0, 0, 0, 0, 0);
    CHECK(bad != 0 && bad != kKhazanDrainCode,
          "a null event pointer is a parameter error, distinct from 'drained'");

    // 6. NO path returns the in-flight code, whatever the queue state. prosper completes every file
    //    operation synchronously, so "still running" is never true here; a future async Mount3 that
    //    genuinely has work outstanding must add its own path and update this arm deliberately.
    umount2(0, 0, 0, 0, 0, 0);
    bool any_in_flight = (bad == kInFlightCode);
    for (int i = 0; i < 8; i++) { arm(); any_in_flight |= (get_event(0, ev, 0, 0, 0, 0) == kInFlightCode); }
    CHECK(!any_in_flight,
          "no queue state reports 0x809F0018 -- nothing in prosper is ever in flight");

    std::printf(fails ? "FAILED (%d)\n" : "PASSED\n", fails);
    return fails ? 1 : 0;
}
