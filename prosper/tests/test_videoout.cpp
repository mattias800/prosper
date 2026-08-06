// test_videoout — guards the libSceVideoOut HLE (hle_graphics.cpp): the query/config functions must
// return real, self-consistent 1080p60 values (not zeroed stubs), and the display-buffer registration
// (swapchain scaffolding) must record the surface the game set up. Drives the functions through the
// NID registry exactly as the guest does, then asserts the reported display + recorded buffers.
#include "../src/hle/dispatch.hpp"
#include "../src/gpu/videoout_present.hpp"
#include "../src/gpu/tile.hpp"              // tile/detile round trip for the flipped-buffer image
#include "../src/host/guest_memory_map.hpp" // the registered-mapping proof for the padded read
#include <algorithm>
#include <vector>
#include <cstdio>
#include <cstdint>
#include <cstdlib>   // setenv/_putenv_s (PROSPER_HDR both-modes test)
#include <cstring>
#include <chrono>
#include <thread>

using namespace prosper;

extern "C" int      prosper_vo_buffer_count();
extern "C" uint32_t prosper_vo_display_width();
extern "C" uint32_t prosper_vo_display_height();
extern "C" uint64_t prosper_vo_display_format();
extern "C" uint64_t prosper_vo_buffer_addr(int i);
extern "C" uint64_t prosper_vo_flip_count();
extern "C" int prosper_vo_flip_rate();
extern "C" void prosper_vo_set_vblank_now_for_test(uint64_t now_ns);
extern "C" void prosper_vo_flip_from_gpu(uint32_t handle, int32_t bufidx,
                                           uint32_t flip_mode, int64_t flip_arg);

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

using Hle7Fn = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t,
                            uint64_t, uint64_t, uint64_t);
using Hle8Fn = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t,
                            uint64_t, uint64_t, uint64_t, uint64_t);

int main() {
    printf("== test_videoout ==\n");
    register_builtin_hle();

    auto open   = Hle::lookup(nid_hash("sceVideoOutOpen"));
    auto close  = Hle::lookup(nid_hash("sceVideoOutClose"));
    auto res    = Hle::lookup(nid_hash("sceVideoOutGetResolutionStatus"));
    auto vbl    = Hle::lookup(nid_hash("sceVideoOutGetVblankStatus"));
    auto waitvbl = Hle::lookup(nid_hash("sceVideoOutWaitVblank"));   // j6RaAUlaLv0 (#1746)
    auto cap    = Hle::lookup(nid_hash("sceVideoOutGetDeviceCapabilityInfo"));
    auto issup  = Hle::lookup("Nv8c-Kb+DUM");   // IsOutputSupported
    auto setba  = Hle::lookup(nid_hash("sceVideoOutSetBufferAttribute"));
    auto regb   = Hle::lookup(nid_hash("sceVideoOutRegisterBuffers"));
    auto setba2 = reinterpret_cast<Hle8Fn>(
        Hle::lookup("PjS5uASwcV8"));             // SetBufferAttribute2
    auto regb2  = Hle::lookup("rKBUtgRrtbk");   // RegisterBuffers2
    auto unreg  = Hle::lookup("N5KDtkIjjJ4");   // UnregisterBuffers
    auto labels = Hle::lookup("OcQybQejHEY");   // GetBufferLabelAddress
    auto setrate = Hle::lookup(nid_hash("sceVideoOutSetFlipRate"));
    auto cfg    = Hle::lookup("w0hLuNarQxY");   // ConfigureOutput
    auto outstat = Hle::lookup("utPrVdxio-8");  // GetOutputStatus
    auto flip   = Hle::lookup(nid_hash("sceVideoOutSubmitFlip"));
    auto pending = Hle::lookup(nid_hash("sceVideoOutIsFlipPending"));
    auto addflip = Hle::lookup(nid_hash("sceVideoOutAddFlipEvent"));
    auto addvblank = Hle::lookup(nid_hash("sceVideoOutAddVblankEvent"));
    auto fstat  = Hle::lookup(nid_hash("sceVideoOutGetFlipStatus"));
    auto margins = Hle::lookup("MTxxrOCeSig");  // SetWindowModeMargins
    CHECK(open && close && res && vbl && cap && issup && setba && regb && setba2 && regb2 &&
              unreg && labels && setrate && cfg && outstat && flip && pending && addflip &&
              addvblank && fstat && margins,
          "VideoOut functions registered");
    if (!(open && close && res && vbl && cap && issup && setba && regb && setba2 && regb2 &&
          unreg && labels && setrate && cfg && outstat && flip && pending && addflip &&
          addvblank && fstat && margins)) {
        printf("== FAIL ==\n"); return 1;
    }

    const uint64_t handle = open(0, 0, 0, 0, 0, 0);
    CHECK((int64_t)handle > 0, "VideoOutOpen returns a live positive handle");

    // #394 F10 (status-query output scope): a null result is an invalid address, not a successful
    // query that silently leaves the caller without status data.
    constexpr uint32_t kInvalidAddress = 0x80290002u;
    CHECK((uint32_t)fstat(handle, 0, 0, 0, 0, 0) == kInvalidAddress,
          "GetFlipStatus rejects a null output pointer");
    CHECK((uint32_t)res(handle, 0, 0, 0, 0, 0) == kInvalidAddress,
          "GetResolutionStatus rejects a null output pointer");
    CHECK((uint32_t)vbl(handle, 0, 0, 0, 0, 0) == kInvalidAddress,
          "GetVblankStatus rejects a null output pointer");
    CHECK((uint32_t)cap(handle, 0, 0, 0, 0, 0) == kInvalidAddress,
          "GetDeviceCapabilityInfo rejects a null output pointer");
    CHECK((uint32_t)outstat(handle, 0, 0, 0, 0, 0) == kInvalidAddress,
          "GetOutputStatus rejects a null output pointer");

    // PROSPER_HDR (the --hdr flag): default advertises SDR (capability 0, output mode 0); with the
    // env set, the BT2020/PQ capability bit and dynamic-range mode 2 — the DOLL-RE'd HDR branch
    // trigger (issue #82).
    {
        auto set_hdr = [](const char* value) {
#ifdef _WIN32
            _putenv_s("PROSPER_HDR", value ? value : "");
#else
            if (value) setenv("PROSPER_HDR", value, 1);
            else unsetenv("PROSPER_HDR");
#endif
        };
        uint64_t capability = 0xEEEEEEEEEEEEEEEEull;
        uint32_t status[2] = {0xEEEEEEEEu, 0xEEEEEEEEu};
        set_hdr(nullptr);
        CHECK(cap(handle, (uint64_t)(uintptr_t)&capability, 0, 0, 0, 0) == 0 && capability == 0,
              "default device capability advertises SDR (0)");
        CHECK(outstat(handle, (uint64_t)(uintptr_t)status, 0, 0, 0, 0) == 0 && status[1] == 0,
              "default output status advertises dynamic-range mode 0 (SDR)");
        set_hdr("1");
        capability = 0xEEEEEEEEEEEEEEEEull; status[1] = 0xEEEEEEEEu;
        CHECK(cap(handle, (uint64_t)(uintptr_t)&capability, 0, 0, 0, 0) == 0 && capability == 0x2,
              "PROSPER_HDR advertises the BT2020/PQ capability bit");
        CHECK(outstat(handle, (uint64_t)(uintptr_t)status, 0, 0, 0, 0) == 0 && status[1] == 2,
              "PROSPER_HDR advertises dynamic-range mode 2 (HDR)");
        set_hdr(nullptr);
    }

    // #394 F10 (handle scope): every handle-taking entrypoint must reject an unknown handle before
    // it mutates output, flip/present state, event registrations, or the display-buffer registry.
    constexpr uint32_t kInvalidHandle = 0x8029000bu;
    const uint64_t invalid_handle = handle + 0x10000;
    uint8_t invalid_output[0x80]; memset(invalid_output, 0xEE, sizeof invalid_output);
    uintptr_t invalid_label_address = UINTPTR_MAX;
    const uint64_t invalid_flip_count = prosper_vo_flip_count();
    const uint64_t invalid_present_count = gpu::present_count();
    CHECK((uint32_t)close(invalid_handle, 0, 0, 0, 0, 0) == kInvalidHandle,
          "Close rejects an unknown handle");
    CHECK((uint32_t)addflip(0, invalid_handle, 0, 0, 0, 0) == kInvalidHandle &&
              (uint32_t)addvblank(0, invalid_handle, 0, 0, 0, 0) == kInvalidHandle,
          "event registration rejects an unknown handle before touching an equeue");
    CHECK((uint32_t)labels(invalid_handle,
                           (uint64_t)(uintptr_t)&invalid_label_address, 0, 0, 0, 0) ==
              kInvalidHandle && invalid_label_address == UINTPTR_MAX,
          "GetBufferLabelAddress rejects an unknown handle without writing output");
    CHECK((uint32_t)setrate(invalid_handle, 1, 0, 0, 0, 0) == kInvalidHandle &&
              prosper_vo_flip_rate() == 0,
          "SetFlipRate rejects an unknown handle without changing rate state");
    CHECK((uint32_t)flip(invalid_handle, 0, 0, 0xBAD, 0, 0) == kInvalidHandle &&
              prosper_vo_flip_count() == invalid_flip_count &&
              gpu::present_count() == invalid_present_count,
          "SubmitFlip rejects an unknown handle before completion side effects");
    prosper_vo_flip_from_gpu((uint32_t)invalid_handle, 0, 0, 0xBAD);
    CHECK(prosper_vo_flip_count() == invalid_flip_count &&
              gpu::present_count() == invalid_present_count,
          "in-stream GPU flip ignores an unknown handle");
    CHECK((uint32_t)pending(invalid_handle, 0, 0, 0, 0, 0) == kInvalidHandle,
          "IsFlipPending rejects an unknown handle");
    CHECK((uint32_t)fstat(invalid_handle, (uint64_t)(uintptr_t)invalid_output,
                          0, 0, 0, 0) == kInvalidHandle &&
              (uint32_t)res(invalid_handle, (uint64_t)(uintptr_t)invalid_output,
                            0, 0, 0, 0) == kInvalidHandle &&
              (uint32_t)vbl(invalid_handle, (uint64_t)(uintptr_t)invalid_output,
                            0, 0, 0, 0) == kInvalidHandle &&
              (uint32_t)cap(invalid_handle, (uint64_t)(uintptr_t)invalid_output,
                            0, 0, 0, 0) == kInvalidHandle &&
              (uint32_t)outstat(invalid_handle, (uint64_t)(uintptr_t)invalid_output,
                                0, 0, 0, 0) == kInvalidHandle,
          "status queries reject an unknown handle");
    bool invalid_output_untouched = true;
    for (uint8_t byte: invalid_output) invalid_output_untouched &= byte == 0xEE;
    CHECK(invalid_output_untouched, "invalid-handle status queries leave output untouched");
    CHECK((uint32_t)issup(invalid_handle, 0xd000000a, 0, 0, 0, 0) == kInvalidHandle &&
              (uint32_t)cfg(invalid_handle, 1, 0, 0, 0, 0) == kInvalidHandle &&
              (uint32_t)margins(invalid_handle, 0, 0, 0, 0, 0) == kInvalidHandle,
          "output configuration entrypoints reject an unknown handle");
    CHECK((uint32_t)regb(invalid_handle, 0, 0, 1, 0, 0) == kInvalidHandle &&
              (uint32_t)regb2(invalid_handle, 0, 0, 0, 1, 0) == kInvalidHandle &&
              (uint32_t)unreg(invalid_handle, 0, 0, 0, 0, 0) == kInvalidHandle,
          "buffer registration entrypoints reject an unknown handle before argument/state work");

    // #394 F7: the rate selector is per-port state rather than a success-returning no-op. All
    // documented rates are accepted; invalid values leave the last valid selection unchanged.
    CHECK(prosper_vo_flip_rate() == 0, "flip rate defaults to 60 Hz");
    CHECK(setrate(handle, 1, 0, 0, 0, 0) == 0 && prosper_vo_flip_rate() == 1,
          "SetFlipRate stores the 30 Hz selector");
    CHECK(setrate(handle, 2, 0, 0, 0, 0) == 0 && prosper_vo_flip_rate() == 2,
          "SetFlipRate stores the 20 Hz selector");
    CHECK((uint32_t)setrate(handle, (uint64_t)(int64_t)-1, 0, 0, 0, 0) == 0x80290001u &&
              prosper_vo_flip_rate() == 2,
          "SetFlipRate rejects -1 without changing the selected rate");
    CHECK((uint32_t)setrate(handle, 3, 0, 0, 0, 0) == 0x80290001u &&
              prosper_vo_flip_rate() == 2,
          "SetFlipRate rejects values above 2 without changing the selected rate");
    CHECK(setrate(handle, 0, 0, 0, 0, 0) == 0 && prosper_vo_flip_rate() == 0,
          "SetFlipRate restores the 60 Hz selector");

    // #394 F4: the query must initialize its output with stable, guest-writable storage for all
    // 16 buffer labels. A missing handler returned success without touching this pointer.
    uintptr_t label_address = UINTPTR_MAX;
    CHECK((uint32_t)labels(handle, 0, 0, 0, 0, 0) == 0x80290002u,
          "GetBufferLabelAddress rejects a null output pointer");
    CHECK(labels(handle, (uint64_t)(uintptr_t)&label_address, 0, 0, 0, 0) == 16 &&
              label_address != 0 && label_address != UINTPTR_MAX,
          "GetBufferLabelAddress returns 16 initialized labels");
    auto* buffer_labels = (uint64_t*)label_address;
    uintptr_t second_label_address = 0;
    CHECK(labels(handle, (uint64_t)(uintptr_t)&second_label_address, 0, 0, 0, 0) == 16 &&
              second_label_address == label_address,
          "GetBufferLabelAddress returns stable storage");

    // #394 F3: before any completed flip, -1 is the ABI's "none yet" sentinel for both the argument
    // and current buffer. Zero is valid and used to make a frame pacer advance prematurely.
    uint8_t initial_fs[0x48]; memset(initial_fs, 0xEE, sizeof initial_fs);
    CHECK(fstat(handle, (uint64_t)(uintptr_t)initial_fs, 0, 0, 0, 0) == 0,
          "initial GetFlipStatus succeeds");
    CHECK(*(uint64_t*)(initial_fs + 0x00) == 0 &&
          *(uint64_t*)(initial_fs + 0x08) == 0 &&
          *(uint64_t*)(initial_fs + 0x10) == 0 &&
          *(int64_t*)(initial_fs + 0x18) == -1 &&
          *(uint64_t*)(initial_fs + 0x20) == 0 &&
          *(uint64_t*)(initial_fs + 0x28) == 0 &&
          *(int32_t*)(initial_fs + 0x30) == 0 &&
          *(int32_t*)(initial_fs + 0x34) == 0 &&
          *(int32_t*)(initial_fs + 0x38) == -1,
          "initial flip status reports empty timing/queues and -1 argument/buffer sentinels");
    bool initial_tail_untouched = true;
    for (size_t i = 0x40; i < sizeof initial_fs; ++i)
        initial_tail_untouched &= initial_fs[i] == 0xEE;
    CHECK(initial_tail_untouched, "GetFlipStatus writes exactly its 0x40-byte ABI struct");

    // #394 F10 (SubmitFlip index scope): reject values outside -1..15 before any completion
    // bookkeeping. An invalid call must not advance either VideoOut status or present state.
    const uint64_t initial_present_count = gpu::present_count();
    CHECK((uint32_t)flip(handle, (uint64_t)(int64_t)-2, 0, 0xBAD, 0, 0) == 0x8029000au,
          "SubmitFlip rejects an index below the -1 special slot");
    CHECK((uint32_t)flip(handle, 16, 0, 0xBAD, 0, 0) == 0x8029000au,
          "SubmitFlip rejects an index above slot 15");
    CHECK(prosper_vo_flip_count() == 0 && gpu::present_count() == initial_present_count,
          "rejected flips do not advance status or present state");

    // Resolution: real 1080p @ 59.94Hz (refresh enum 3), not zeroed. The ABI struct is 0x30
    // bytes: flags/reserved0 at 0x1c and reserved1[3] at 0x20 must not retain caller garbage.
    uint8_t rs[0x38]; memset(rs, 0xEE, sizeof rs);
    res(handle, (uint64_t)(uintptr_t)rs, 0, 0, 0, 0);
    CHECK(*(uint32_t*)(rs + 0x00) == 1920 && *(uint32_t*)(rs + 0x04) == 1080, "resolution 1920x1080 (full)");
    CHECK(*(uint32_t*)(rs + 0x08) == 1920 && *(uint32_t*)(rs + 0x0c) == 1080, "resolution 1920x1080 (pane)");
    CHECK(*(uint64_t*)(rs + 0x10) == 3, "refresh rate = 59.94Hz (enum 3)");
    bool resolution_reserved_zero = true;
    for (size_t i = 0x1c; i < 0x30; ++i) resolution_reserved_zero &= rs[i] == 0;
    CHECK(resolution_reserved_zero, "resolution flags and reserved tail are initialized");
    bool resolution_canary_untouched = true;
    for (size_t i = 0x30; i < sizeof rs; ++i) resolution_canary_untouched &= rs[i] == 0xEE;
    CHECK(resolution_canary_untouched,
          "GetResolutionStatus writes exactly its 0x30-byte ABI struct");

    // Vblank counter advances with TIME (one tick per ~16.68 ms vblank period), not per poll —
    // the old ++-per-call behavior made every poll look like a fresh vblank (issue #82). Two
    // immediate calls land in the same period; a > one-period sleep must advance the count.
    uint8_t vb[0x28];
    vbl(handle, (uint64_t)(uintptr_t)vb, 0, 0, 0, 0); uint64_t c0 = *(uint64_t*)vb;
    CHECK(c0 > 0, "first vblank query reports a live non-zero counter");
    vbl(handle, (uint64_t)(uintptr_t)vb, 0, 0, 0, 0); uint64_t c1 = *(uint64_t*)vb;
    CHECK(c1 - c0 <= 1, "vblank counter does NOT tick per poll (immediate re-poll ~same period)");
    std::this_thread::sleep_for(std::chrono::milliseconds(40));   // > 2 vblank periods
    vbl(handle, (uint64_t)(uintptr_t)vb, 0, 0, 0, 0); uint64_t c2 = *(uint64_t*)vb;
    CHECK(c2 > c1, "vblank counter advances across a >1-period sleep");
    CHECK(*(uint64_t*)(vb + 0x08) > 0, "vblank processTime filled (Kyty layout @0x08)");

    // WaitVblank must BLOCK to the next ~16.68 ms boundary and land on the tick it waited for
    // (#1746). Unregistered it fell to the return-0 stub and came back instantly, turning a
    // title's `for(;;) WaitVblank(port)` display thread into a spin loop with no pacing.
    CHECK(waitvbl != nullptr, "sceVideoOutWaitVblank is registered (j6RaAUlaLv0)");
    if (waitvbl) {
        CHECK((uint32_t)waitvbl(invalid_handle, 0, 0, 0, 0, 0) == kInvalidHandle,
              "WaitVblank rejects an invalid handle");
        // Five waits must consume at least four full vblank periods of real time. A stub that
        // returns immediately finishes in microseconds and fails this outright.
        vbl(handle, (uint64_t)(uintptr_t)vb, 0, 0, 0, 0);
        const uint64_t wait_c0 = *(uint64_t*)vb;
        const auto wait_t0 = std::chrono::steady_clock::now();
        bool wait_ok = true;
        for (int i = 0; i < 5; ++i) wait_ok &= waitvbl(handle, 0, 0, 0, 0, 0) == 0;
        const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                    std::chrono::steady_clock::now() - wait_t0).count();
        vbl(handle, (uint64_t)(uintptr_t)vb, 0, 0, 0, 0);
        const uint64_t wait_c1 = *(uint64_t*)vb;
        CHECK(wait_ok, "WaitVblank returns 0 on a live handle");
        CHECK(elapsed_us >= 4 * 16683,
              "five WaitVblank calls block for at least four vblank periods");
        CHECK(wait_c1 >= wait_c0 + 4,
              "WaitVblank advances GetVblankStatus's counter by one tick per wait");

        // PHASE, which the counter assertion above does NOT test. A WaitVblank with its own epoch
        // shifted half a period — same period, different phase — advances the counter by exactly
        // the same amount and passes everything above, and phase is the entire content of the
        // "shared timebase" claim: a game that waits and then reads must land ON the tick it
        // waited for, not between two.
        //
        // This must not depend on host scheduling. #1770's minimum-residue sampler wrapped after
        // long overshoots; #1793 found that its replacement's admission rule was also invalid. A
        // wait between consecutive late wakes takes approximately
        //
        //     period - previous_lateness + current_lateness
        //
        // rather than `period + current_lateness`, so consistent Rosetta lateness cancels and an
        // out-of-phase sample is incorrectly admitted. Drive the real HLE calls with deterministic
        // time instead: WaitVblank's test sleeper advances the same clock GetVblankStatus reads.
        constexpr uint64_t kPeriodNs = 16683350ull;
        vbl(handle, (uint64_t)(uintptr_t)vb, 0, 0, 0, 0);
        const uint64_t live_tsc = *(uint64_t*)(vb + 0x10);
        const uint64_t live_process_ns = *(uint64_t*)(vb + 0x08) * 1000ull;
        constexpr uint64_t kStartPhaseNs = 3 * kPeriodNs / 4;
        const uint64_t advance_ns =
            (kStartPhaseNs + kPeriodNs - live_process_ns % kPeriodNs) % kPeriodNs;
        const uint64_t fake_start_ns = live_tsc + advance_ns;
        prosper_vo_set_vblank_now_for_test(fake_start_ns);

        vbl(handle, (uint64_t)(uintptr_t)vb, 0, 0, 0, 0);
        const uint64_t fake_c0 = *(uint64_t*)vb;
        CHECK(*(uint64_t*)(vb + 0x10) == fake_start_ns,
              "WaitVblank PHASE test clock is active through GetVblankStatus");
        CHECK(waitvbl(handle, 0, 0, 0, 0, 0) == 0,
              "WaitVblank accepts a live handle under deterministic time");
        vbl(handle, (uint64_t)(uintptr_t)vb, 0, 0, 0, 0);
        const uint64_t fake_end_ns = *(uint64_t*)(vb + 0x10);
        CHECK(fake_end_ns > fake_start_ns && fake_end_ns <= fake_start_ns + kPeriodNs &&
                  *(uint64_t*)vb == fake_c0 + 1,
              "WaitVblank advances deterministic time to exactly the next status tick");

        const uint64_t phase_ns = (*(uint64_t*)(vb + 0x08) * 1000ull) % kPeriodNs;
        const uint64_t phase_distance_ns =
            phase_ns < kPeriodNs - phase_ns ? phase_ns : kPeriodNs - phase_ns;
        // processTime is reported in whole microseconds, so an exact boundary can reconstruct up
        // to 999 ns below zero. Circular distance keeps that ABI quantisation from looking like a
        // nearly-full-period phase error. A half-period private epoch remains ~8.34 ms away.
        CHECK(phase_distance_ns < 1000,
              "WaitVblank shares GetVblankStatus's PHASE, not just its period "
              "(wakes land on a boundary of the same grid)");
        prosper_vo_set_vblank_now_for_test(0);
    }

    // Device capability: plain SDR display (capability 0).
    uint64_t dc = 0xDEAD;
    cap(handle, (uint64_t)(uintptr_t)&dc, 0, 0, 0, 0);
    CHECK(dc == 0, "device capability = 0 (SDR)");

    // IsOutputSupported → supported.
    CHECK(issup(handle, 0xd000000a, 0, 0, 0, 0) == 1, "IsOutputSupported returns 1");

    // #394 F2: the legacy setter is an output function, not a success-returning no-op. It fills
    // the seven public fields, zeros option/reserved storage, and writes exactly 0x28 bytes.
    uint8_t legacy_attr[0x30]; memset(legacy_attr, 0xEE, sizeof legacy_attr);
    auto setba7 = reinterpret_cast<Hle7Fn>(setba);
    setba7((uint64_t)(uintptr_t)legacy_attr, 0x80002200u, 1, 2, 1280, 720, 1344);
    CHECK(*(uint32_t*)(legacy_attr + 0x00) == 0x80002200u &&
          *(uint32_t*)(legacy_attr + 0x04) == 1 &&
          *(uint32_t*)(legacy_attr + 0x08) == 2,
          "legacy attribute format, tiling, and aspect set");
    CHECK(*(uint32_t*)(legacy_attr + 0x0c) == 1280 &&
          *(uint32_t*)(legacy_attr + 0x10) == 720 &&
          *(uint32_t*)(legacy_attr + 0x14) == 1344,
          "legacy attribute dimensions and pitch set");
    bool legacy_reserved_zero = true;
    for (size_t i = 0x18; i < 0x28; ++i) legacy_reserved_zero &= legacy_attr[i] == 0;
    CHECK(legacy_reserved_zero, "legacy attribute option and reserved fields initialized");
    bool legacy_canary_untouched = true;
    for (size_t i = 0x28; i < sizeof legacy_attr; ++i)
        legacy_canary_untouched &= legacy_attr[i] == 0xEE;
    CHECK(legacy_canary_untouched,
          "SetBufferAttribute writes exactly its 0x28-byte ABI struct");

    // #394 F1: legacy registration consumes a flat address array, assigns the lowest free group,
    // and records its non-zero slot range using the Gen4 attribute layout.
    uint8_t legacy_fb2[16], legacy_fb3[16];
    const void* legacy_buffers[2] = {legacy_fb2, legacy_fb3};
    CHECK((uint32_t)regb(handle, 15, (uint64_t)(uintptr_t)legacy_buffers, 2,
                         (uint64_t)(uintptr_t)legacy_attr, 0) == 0x80290001u,
          "legacy RegisterBuffers rejects a range past slot 15");
    buffer_labels[2] = 0xAAAA;
    buffer_labels[3] = 0xBBBB;
    uint64_t legacy_group = regb(handle, 2, (uint64_t)(uintptr_t)legacy_buffers, 2,
                                 (uint64_t)(uintptr_t)legacy_attr, 0);
    CHECK(legacy_group == 0, "legacy RegisterBuffers returns the lowest free attribute group");
    CHECK(prosper_vo_buffer_count() == 4 &&
          prosper_vo_buffer_addr(2) == (uint64_t)(uintptr_t)legacy_fb2 &&
          prosper_vo_buffer_addr(3) == (uint64_t)(uintptr_t)legacy_fb3,
          "legacy RegisterBuffers records the flat framebuffer array at the requested slots");
    CHECK(buffer_labels[2] == 0 && buffer_labels[3] == 0,
          "legacy RegisterBuffers resets labels for its registered slots");
    CHECK(prosper_vo_display_width() == 1280 && prosper_vo_display_height() == 720 &&
          prosper_vo_display_format() == 0x80002200u,
          "legacy RegisterBuffers records Gen4 surface geometry and format");
    CHECK((uint32_t)regb(handle, 3, (uint64_t)(uintptr_t)legacy_buffers, 2,
                         (uint64_t)(uintptr_t)legacy_attr, 0) == 0x80290010u,
          "legacy RegisterBuffers rejects an overlapping slot range");
    CHECK(unreg(handle, legacy_group, 0, 0, 0, 0) == 0 &&
          prosper_vo_buffer_count() == 0 && prosper_vo_display_width() == 0,
          "legacy buffer group unregisters cleanly before Gen5 registration");

    // SetBufferAttribute2 fills exactly the 0x50-byte Gen5 attribute, including the two stack args.
    uint8_t attr[0x58]; memset(attr, 0xEE, sizeof attr);
    uint64_t fmt = 0x8000000000000000ull;   // the PS5 format the game requests (A8R8G8B8 sRGB)
    constexpr uint64_t option = 0x1122334455667788ull;
    constexpr uint32_t dcc_control = 0xa1b2c3d4u;
    constexpr uint64_t dcc_clear = 0x8877665544332211ull;
    setba2((uint64_t)(uintptr_t)attr, fmt, 0 /*tiling*/, 1920, 1080, option,
           dcc_control, dcc_clear);
    CHECK(*(uint32_t*)(attr + 0x0c) == 1920 && *(uint32_t*)(attr + 0x10) == 1080, "attribute width/height set");
    CHECK(*(uint64_t*)(attr + 0x20) == fmt, "attribute pixel_format set");
    CHECK(*(uint32_t*)(attr + 0x04) == 0, "attribute tiling_mode set");
    CHECK(*(uint32_t*)(attr + 0x14) == 0 && *(uint64_t*)(attr + 0x18) == option,
          "attribute pitch and option set");
    CHECK(*(uint64_t*)(attr + 0x28) == dcc_clear &&
          *(uint32_t*)(attr + 0x30) == dcc_control,
          "attribute DCC clear color and control set from stack arguments");
    bool attr_reserved_zero = true;
    for (size_t i = 0x34; i < 0x50; ++i) attr_reserved_zero &= attr[i] == 0;
    CHECK(attr_reserved_zero, "attribute pad and reserved fields initialized");
    bool attr_canary_untouched = true;
    for (size_t i = 0x50; i < sizeof attr; ++i) attr_canary_untouched &= attr[i] == 0xEE;
    CHECK(attr_canary_untouched, "SetBufferAttribute2 writes exactly its 0x50-byte ABI struct");

    // ConfigureOutput accepts the (single advertised) mode.
    CHECK(cfg(handle, 1, 0, 0, 0, 0) == 0, "ConfigureOutput accepted");

    // RegisterBuffers2: triple-buffered surface. buffers = array of VideoOutBuffers {data, ...}.
    struct VOB { const void* data; const void* metadata; const void* reserved[2]; };
    uint8_t fb0[16], fb1[16], fb2[16];
    VOB buffers[3] = { {fb0,0,{0,0}}, {fb1,0,{0,0}}, {fb2,0,{0,0}} };
    buffer_labels[0] = 0x1111;
    buffer_labels[1] = 0x2222;
    buffer_labels[2] = 0x3333;
    uint64_t rc = regb2(handle, 0 /*set*/, 0 /*start*/, (uint64_t)(uintptr_t)buffers, 3, (uint64_t)(uintptr_t)attr);
    CHECK(rc == 0, "RegisterBuffers2 accepted 3 buffers");
    CHECK(buffer_labels[0] == 0 && buffer_labels[1] == 0 && buffer_labels[2] == 0,
          "RegisterBuffers2 resets labels for its registered slots");

    // Swapchain scaffolding recorded the surface.
    CHECK(prosper_vo_buffer_count() == 3, "registry recorded 3 display buffers");
    CHECK(prosper_vo_display_width() == 1920 && prosper_vo_display_height() == 1080, "registry recorded 1080p surface");
    CHECK(prosper_vo_display_format() == fmt, "registry recorded the pixel format");
    CHECK(prosper_vo_buffer_addr(0) == (uint64_t)(uintptr_t)fb0 &&
          prosper_vo_buffer_addr(2) == (uint64_t)(uintptr_t)fb2, "registry recorded each framebuffer address");

    // A set identifier owns one registration until it is unregistered. Rejecting the duplicate
    // must happen before any slot is changed, even when the proposed range itself is free.
    uint8_t fb4a[16], fb5a[16];
    VOB candidate_buffers[2] = { {fb4a,0,{0,0}}, {fb5a,0,{0,0}} };
    rc = regb2(handle, 0 /*duplicate set*/, 4 /*free start*/,
               (uint64_t)(uintptr_t)candidate_buffers, 2, (uint64_t)(uintptr_t)attr);
    CHECK((uint32_t)rc == 0x8029000au,
          "RegisterBuffers2 rejects a duplicate active set identifier");

    // A different set may not claim any slot already owned by set 0. The whole range is validated
    // before mutation, so this failure cannot replace either overlapping address or reserve set 1.
    rc = regb2(handle, 1 /*new set*/, 1 /*occupied start*/,
               (uint64_t)(uintptr_t)candidate_buffers, 2, (uint64_t)(uintptr_t)attr);
    CHECK((uint32_t)rc == 0x80290010u,
          "RegisterBuffers2 rejects slots owned by another active set");
    CHECK(prosper_vo_buffer_count() == 3 &&
          prosper_vo_buffer_addr(0) == (uint64_t)(uintptr_t)fb0 &&
          prosper_vo_buffer_addr(1) == (uint64_t)(uintptr_t)fb1 &&
          prosper_vo_buffer_addr(2) == (uint64_t)(uintptr_t)fb2,
          "failed registrations leave all existing slots unchanged");

    uint8_t fs[0x40]; memset(fs, 0xEE, sizeof fs);
    buffer_labels[2] = 1;
    CHECK(flip(handle, 2 /*buffer*/, 0 /*mode*/, 0x12345678 /*flipArg*/, 0, 0) == 0,
          "SubmitFlip accepted buffer 2");
    CHECK(buffer_labels[2] == 1,
          "first completed flip retains the current buffer label");
    fstat(handle, (uint64_t)(uintptr_t)fs, 0, 0, 0, 0);
    CHECK(*(uint64_t*)(fs + 0x00) == 1, "flip status count increments");
    CHECK(*(uint64_t*)(fs + 0x10) > 0 &&
              *(uint64_t*)(fs + 0x08) == *(uint64_t*)(fs + 0x10) / 1000,
          "flip status reports processTime and tsc on the shared guest clock");
    CHECK(*(int64_t*) (fs + 0x18) == 0x12345678, "flip status reports the submitted flipArg");
    CHECK(*(uint64_t*)(fs + 0x20) == *(uint64_t*)(fs + 0x10),
          "synchronous flip reports one submit/completion timestamp");
    CHECK(*(uint64_t*)(fs + 0x28) == 0 && *(int32_t*)(fs + 0x30) == 0 &&
              *(int32_t*)(fs + 0x34) == 0,
          "synchronous flip reports zero reserved and pending-queue fields");
    CHECK(*(int32_t*) (fs + 0x38) == 2, "flip status reports the submitted currentBuffer");

    // Range validation: out-of-range buffer counts are rejected.
    CHECK(regb2(handle, 2, 0, (uint64_t)(uintptr_t)buffers, 99, (uint64_t)(uintptr_t)attr) != 0,
          "RegisterBuffers2 rejects an out-of-range buffer_num");

    CHECK((uint32_t)unreg(handle, 3 /*unregistered set*/, 0, 0, 0, 0) == 0x8029000au,
          "UnregisterBuffers rejects an absent set without changing registration");
    CHECK(prosper_vo_buffer_count() == 3 &&
              prosper_vo_buffer_addr(0) == (uint64_t)(uintptr_t)fb0,
          "failed unregistration preserves the active buffer set");
    // A second set at a non-zero start is valid. Give it different geometry so flips can prove the
    // present path takes dimensions from the selected buffer's owning set, not the latest global.
    uint8_t high_attr[0x50];
    memcpy(high_attr, attr, sizeof high_attr);
    *(uint32_t*)(high_attr + 0x0c) = 1280;
    *(uint32_t*)(high_attr + 0x10) = 720;
    uint8_t fb4[16], fb5[16];
    VOB high_buffers[2] = { {fb4,0,{0,0}}, {fb5,0,{0,0}} };
    uint64_t high_rc = regb2(handle, 1 /*set*/, 4 /*start*/,
                             (uint64_t)(uintptr_t)high_buffers, 2,
                             (uint64_t)(uintptr_t)high_attr);
    CHECK(high_rc == 0 && prosper_vo_buffer_count() == 6 &&
              prosper_vo_display_width() == 1280 && prosper_vo_display_height() == 720,
          "registry tracks two independently owned buffer sets");
    flip(handle, 2 /*set 0 buffer*/, 0, 0, 0, 0);
    CHECK(buffer_labels[2] == 0,
          "SubmitFlip releases the previously displayed buffer label");
    CHECK(gpu::present_width() == 1920 && gpu::present_height() == 1080,
          "present uses set 0 geometry when an older set's buffer is flipped");
    buffer_labels[2] = 1;
    buffer_labels[4] = 1;
    prosper_vo_flip_from_gpu(handle, 4 /*set 1 buffer*/, 0, 0);
    CHECK(buffer_labels[2] == 0 && buffer_labels[4] == 1,
          "GPU flip releases the previous label without overwriting the current label");
    CHECK(gpu::present_width() == 1280 && gpu::present_height() == 720,
          "present uses set 1 geometry when its buffer is flipped");
    CHECK(unreg(handle, 1 /*set*/, 0, 0, 0, 0) == 0 &&
              prosper_vo_buffer_count() == 3 &&
              prosper_vo_buffer_addr(0) == (uint64_t)(uintptr_t)fb0 &&
              prosper_vo_buffer_addr(2) == (uint64_t)(uintptr_t)fb2 &&
              prosper_vo_display_width() == 1920 && prosper_vo_display_height() == 1080,
          "unregistering one set restores the remaining set's range and geometry");
    CHECK(regb2(handle, 1 /*reused set*/, 4 /*reused start*/,
                (uint64_t)(uintptr_t)high_buffers, 2,
                (uint64_t)(uintptr_t)high_attr) == 0,
          "an unregistered buffer range can be registered again");
    buffer_labels[4] = 1;
    flip(handle, 4 /*new registration in reused slot*/, 0, 0, 0, 0);
    CHECK(buffer_labels[4] == 1,
          "first flip of a reused slot retains the new registration's current label");
    CHECK(unreg(handle, 1 /*reused set*/, 0, 0, 0, 0) == 0 &&
              buffer_labels[4] == 0,
          "unregistering a displayed slot retires its label identity");
    buffer_labels[2] = 1;
    flip(handle, 2 /*remaining set 0 buffer*/, 0, 0, 0, 0);
    CHECK(buffer_labels[2] == 1,
          "first flip after the previous slot is unregistered retains the current label");
    CHECK(gpu::present_width() == 1920 && gpu::present_height() == 1080,
          "present continues with the surviving set after unregister");
    CHECK(unreg(handle, 0 /*set*/, 0, 0, 0, 0) == 0 &&
              prosper_vo_buffer_count() == 0 && prosper_vo_buffer_addr(0) == 0 &&
              prosper_vo_display_width() == 0 && prosper_vo_display_height() == 0,
          "unregistering the remaining set clears the registry and geometry");

    // ---- The flipped buffer read as an IMAGE, not as bytes (#1968) --------------------------------
    // A registered scanout carries its own tiling mode, and until this was honoured every consumer
    // that treated the read as pixels got a swizzled surface: Sonic Frontiers' guest-composited
    // frames reached prosper as horizontal-band noise rather than as the frames they are. The
    // round trip below is the contract — tile a known image into guest memory exactly as the GPU
    // would, flip it, and require the present layer to hand back the original linear image.
    {
        constexpr uint32_t kW = 256, kH = 128;   // one whole SW_64KB_R_X block row: footprint == W*H*4
        std::vector<uint8_t> reference(static_cast<size_t>(kW) * kH * 4);
        for (size_t i = 0; i < reference.size(); i += 4) {
            const size_t texel = i / 4;
            reference[i + 0] = (uint8_t)(texel & 0xff);
            reference[i + 1] = (uint8_t)((texel >> 8) & 0xff);
            reference[i + 2] = (uint8_t)(texel * 7u);
            reference[i + 3] = 0xff;
        }
        const uint32_t tiled_mode = gpu::videoout_scanout_tile_mode(0 /*TILE*/, 4);
        CHECK(tiled_mode == (uint32_t)gpu::TileMode::Sw64KbRX,
              "a TILE-mode 32-bpp scanout is the SW_64KB_R_X render-target swizzle");
        CHECK(gpu::videoout_scanout_tile_mode(1 /*LINEAR*/, 4) == (uint32_t)gpu::TileMode::Linear &&
                  gpu::videoout_scanout_tile_mode(0, 8) == (uint32_t)gpu::TileMode::Linear &&
                  gpu::videoout_scanout_tile_mode(7 /*unknown*/, 4) == (uint32_t)gpu::TileMode::Linear,
              "LINEAR, non-32-bpp and unrecognized VideoOut tiling modes stay a straight copy");
        const size_t tiled_bytes = gpu::tiled_surface_bytes(kW, kH, tiled_mode, 0, 4);
        CHECK(tiled_bytes == reference.size(),
              "the chosen extent has no block padding, so the round trip is exact");

        std::vector<uint8_t> tiled_fb(tiled_bytes, 0);
        gpu::tile_surface(tiled_fb.data(), reference.data(), kW, kH, tiled_mode, 0, 4);
        CHECK(tiled_fb != reference, "the tiled framebuffer really is swizzled, not a copy");
        const std::vector<uint8_t> tiled_at_registration = tiled_fb;

        uint8_t tiled_attr[0x50];
        setba2((uint64_t)(uintptr_t)tiled_attr, fmt, 0 /*TILE*/, kW, kH, option, 0, 0);
        VOB tiled_buffers[1] = { {tiled_fb.data(), 0, {0, 0}} };
        CHECK(regb2(handle, 0, 0, (uint64_t)(uintptr_t)tiled_buffers, 1,
                    (uint64_t)(uintptr_t)tiled_attr) == 0,
              "registered a TILE-mode scanout");
        CHECK(flip(handle, 0, 0, 0, 0, 0) == 0, "flipped the TILE-mode scanout");

        gpu::PresentSnapshot snapshot;
        const bool got = gpu::present_snapshot(snapshot);
        CHECK(got && snapshot.source == gpu::PresentSource::RawScanout,
              "with no rendered frame the present layer reads the guest scanout");
        CHECK(got && snapshot.width == kW && snapshot.height == kH &&
                  snapshot.rgba.size() == reference.size(),
              "the guest scanout read is the registered extent, linear");
        CHECK(got && snapshot.rgba == reference,
              "a TILE-mode scanout is de-swizzled into the image the guest composited");

        // ---- guest authorship (#2044, review finding B2) -----------------------------------------
        // The precondition for publishing a flipped buffer as a present source. The buffer here is
        // full of non-zero bytes and has NOT been written since it was registered, so the predicate
        // that reverted the first attempt — "not all bytes are zero" — would call this authored.
        VideoOutLinearRead read;
        CHECK(videoout_read_front_linear(read) && read.pixels == reference,
              "the image reader returns the same de-swizzled frame as the present snapshot");
        CHECK(!read.guest_authored,
              "a registered-but-unwritten buffer is NOT authored, however non-zero its bytes are");

        // The guest composites into it. Anything that changes the surface counts; this writes the
        // whole thing, which is what a real composite does.
        for (size_t i = 0; i < tiled_fb.size(); ++i) tiled_fb[i] = (uint8_t)(tiled_fb[i] ^ 0x5a);
        CHECK(videoout_read_front_linear(read) && read.guest_authored,
              "a buffer whose contents changed after registration is authored");
        // Sticky: authorship already demonstrated is not lost by a later frame that happens to match
        // the registration baseline — otherwise a title returning to its opening image would flicker.
        tiled_fb = tiled_at_registration;
        CHECK(videoout_read_front_linear(read) && read.guest_authored,
              "authorship is sticky across a frame that coincidentally matches the baseline");

        // Bendy's case (PPSA27616), stated exactly: the SAME memory, still holding a real frame, is
        // re-registered. Authorship is evidence about one registration, so the new one starts with
        // none and the branch declines — where "not all bytes are zero" would publish stale bytes.
        CHECK(unreg(handle, 0, 0, 0, 0, 0) == 0, "unregistered the TILE-mode scanout");
        CHECK(regb2(handle, 0, 0, (uint64_t)(uintptr_t)tiled_buffers, 1,
                    (uint64_t)(uintptr_t)tiled_attr) == 0 && flip(handle, 0, 0, 0, 0, 0) == 0,
              "re-registered the same memory, still holding the previous frame");
        CHECK(videoout_read_front_linear(read) && !read.guest_authored,
              "re-registering over reused memory does NOT inherit the previous authorship");
        CHECK(read.pixels == reference,
              "…and the frame is still read correctly; only the authorship claim is withheld");

        // The raw byte primitive stays raw: a diagnostic that dumps guest memory must keep seeing
        // exactly what is there, so only the image reader de-swizzles.
        std::vector<uint8_t> raw;
        VideoOutBufferSnapshot raw_meta;
        CHECK(videoout_copy_front_buffer(raw, raw_meta) && raw == tiled_fb,
              "videoout_copy_front_buffer still returns the untouched guest bytes");
        CHECK(unreg(handle, 0, 0, 0, 0, 0) == 0, "unregistered the reused-memory scanout");

        // LINEAR keeps the historical straight copy end to end.
        uint8_t linear_attr[0x50];
        setba2((uint64_t)(uintptr_t)linear_attr, fmt, 1 /*LINEAR*/, kW, kH, option, 0, 0);
        VOB linear_buffers[1] = { {reference.data(), 0, {0, 0}} };
        CHECK(regb2(handle, 0, 0, (uint64_t)(uintptr_t)linear_buffers, 1,
                    (uint64_t)(uintptr_t)linear_attr) == 0,
              "registered a LINEAR scanout");
        CHECK(flip(handle, 0, 0, 0, 0, 0) == 0, "flipped the LINEAR scanout");
        gpu::PresentSnapshot linear_snapshot;
        CHECK(gpu::present_snapshot(linear_snapshot) && linear_snapshot.rgba == reference,
              "a LINEAR scanout is published byte-for-byte");
        CHECK(unreg(handle, 0, 0, 0, 0, 0) == 0, "unregistered the LINEAR scanout");
    }

    // ---- The swizzle footprint is larger than width*height*4, and reading it must be earned -------
    // An extent with block padding: SW_64KB_R_X blocks are 256x64 texels at 32 bpp, so 256x100 spans
    // two block rows and its footprint is 256x128x4 — 28,672 bytes past the nominal end, holding real
    // texels of the second block row rather than bottom padding. Two independent facts must hold
    // before those bytes may be read: the guest's own registration stride, AND one registered guest
    // mapping spanning the whole padded range. Each covers the other's blind spot (#2026 review N2),
    // so all three combinations are asserted.
    {
        constexpr uint32_t kW = 256, kH = 100;
        const size_t linear_bytes = static_cast<size_t>(kW) * kH * 4;
        const uint32_t tiled_mode = (uint32_t)gpu::TileMode::Sw64KbRX;
        const size_t tiled_bytes = gpu::tiled_surface_bytes(kW, kH, tiled_mode, 0, 4);
        CHECK(tiled_bytes > linear_bytes, "this extent's swizzle footprint exceeds width*height*4");

        std::vector<uint8_t> reference(linear_bytes);
        for (size_t i = 0; i < reference.size(); i += 4) {
            const size_t texel = i / 4;
            reference[i + 0] = (uint8_t)(texel * 5u);
            reference[i + 1] = (uint8_t)(texel >> 4);
            reference[i + 2] = (uint8_t)(texel | 1u);
            reference[i + 3] = 0xff;
        }
        // What a truncated read can produce, derived rather than hand-picked so the expectation is
        // independent of the block geometry.
        std::vector<uint8_t> expected_partial(linear_bytes, 0);
        {
            std::vector<uint8_t> whole(tiled_bytes, 0), readable(tiled_bytes, 0);
            gpu::tile_surface(whole.data(), reference.data(), kW, kH, tiled_mode, 0, 4);
            std::copy(whole.begin(), whole.begin() + linear_bytes, readable.begin());
            gpu::detile_surface(expected_partial.data(), readable.data(), kW, kH, tiled_mode, 0, 4);
        }
        CHECK(expected_partial != reference,
              "truncating the read really does cost image content, so these arms can differ");

        uint8_t padded_attr[0x50];
        setba2((uint64_t)(uintptr_t)padded_attr, fmt, 0 /*TILE*/, kW, kH, option, 0, 0);

        // (a) Both proofs hold: two buffers one whole footprint apart, inside one guest mapping that
        // spans them. This is how a real VideoOut set is allocated.
        std::vector<uint8_t> pair(tiled_bytes * 2, 0);
        gpu::tile_surface(pair.data(), reference.data(), kW, kH, tiled_mode, 0, 4);
        host::notify_guest_mapping_added((uint64_t)(uintptr_t)pair.data(), pair.size(), true);
        VOB pair_buffers[2] = { {pair.data(), 0, {0, 0}},
                                {pair.data() + tiled_bytes, 0, {0, 0}} };
        CHECK(regb2(handle, 0, 0, (uint64_t)(uintptr_t)pair_buffers, 2,
                    (uint64_t)(uintptr_t)padded_attr) == 0 && flip(handle, 0, 0, 0, 0, 0) == 0,
              "registered and flipped a padded-footprint scanout whose allocation is proven");
        VideoOutLinearRead proven;
        CHECK(videoout_read_front_linear(proven) && proven.padded_footprint &&
                  proven.pixels == reference,
              "with both proofs the whole image de-swizzles, padded block row included");
        CHECK(unreg(handle, 0, 0, 0, 0, 0) == 0, "unregistered the proven-footprint scanout");
        host::notify_guest_mapping_removed((uint64_t)(uintptr_t)pair.data(), pair.size());

        // (b) The stride proof alone is not enough — the N2 case. Two buffers far apart, each owning
        // only width*height*4: the smallest registered distance is comfortably larger than the
        // footprint, so the stride test passes, and only the mapping test stops a silent over-read
        // into whatever follows the allocation (~240 KiB of it at 4K).
        std::vector<uint8_t> apart(tiled_bytes * 4, 0);
        gpu::tile_surface(apart.data(), reference.data(), kW, kH, tiled_mode, 0, 4);
        const uint64_t first = (uint64_t)(uintptr_t)apart.data();
        const uint64_t second = first + tiled_bytes * 2;
        host::notify_guest_mapping_added(first, linear_bytes, true);
        host::notify_guest_mapping_added(second, linear_bytes, true);
        VOB apart_buffers[2] = { {apart.data(), 0, {0, 0}},
                                 {apart.data() + tiled_bytes * 2, 0, {0, 0}} };
        CHECK(regb2(handle, 0, 0, (uint64_t)(uintptr_t)apart_buffers, 2,
                    (uint64_t)(uintptr_t)padded_attr) == 0 && flip(handle, 0, 0, 0, 0, 0) == 0,
              "registered and flipped two separately allocated buffers that land far apart");
        VideoOutLinearRead unmapped;
        CHECK(videoout_read_front_linear(unmapped) && !unmapped.padded_footprint,
              "a stride wide enough on its own does NOT license reading past the allocation");
        CHECK(unmapped.pixels == expected_partial,
              "…and the read stops exactly at the bytes the mapping vouches for");
        CHECK(unreg(handle, 0, 0, 0, 0, 0) == 0, "unregistered the far-apart scanouts");
        host::notify_guest_mapping_removed(first, linear_bytes);
        host::notify_guest_mapping_removed(second, linear_bytes);

        // (c) The mapping proof alone is not enough either: one buffer inside a generous mapping
        // still proves nothing about what follows it, because nothing says the rest of that mapping
        // is not another object. The image must still come back — de-swizzling the readable part is
        // far better than handing back swizzled bytes — but it cannot be complete, and that has to
        // be visible here rather than discovered as a corrupted corner in a title.
        std::vector<uint8_t> lone(tiled_bytes * 2, 0);
        gpu::tile_surface(lone.data(), reference.data(), kW, kH, tiled_mode, 0, 4);
        host::notify_guest_mapping_added((uint64_t)(uintptr_t)lone.data(), lone.size(), true);
        VOB lone_buffers[1] = { {lone.data(), 0, {0, 0}} };
        CHECK(regb2(handle, 0, 0, (uint64_t)(uintptr_t)lone_buffers, 1,
                    (uint64_t)(uintptr_t)padded_attr) == 0 && flip(handle, 0, 0, 0, 0, 0) == 0,
              "registered and flipped a lone padded-footprint scanout inside a large mapping");
        VideoOutLinearRead unproven;
        CHECK(videoout_read_front_linear(unproven) && !unproven.padded_footprint,
              "a single registered buffer never licenses the padded read, mapped or not");
        CHECK(unproven.pixels.size() == linear_bytes && unproven.pixels == expected_partial,
              "an unproven footprint de-swizzles precisely the bytes it is allowed to read");
        size_t matching = 0;
        for (size_t i = 0; i + 3 < reference.size(); i += 4)
            if (std::equal(reference.begin() + i, reference.begin() + i + 4,
                           unproven.pixels.begin() + i)) ++matching;
        CHECK(matching > reference.size() / 4 / 2 && matching < reference.size() / 4,
              "most of the image survives the truncation, but demonstrably not all of it");
        CHECK(unreg(handle, 0, 0, 0, 0, 0) == 0, "unregistered the lone scanout");
        host::notify_guest_mapping_removed((uint64_t)(uintptr_t)lone.data(), lone.size());
    }

    const uint64_t before_blank_flip = prosper_vo_flip_count();
    CHECK(flip(handle, (uint64_t)(int64_t)-1, 0, 0, 0, 0) == 0 &&
              prosper_vo_flip_count() == before_blank_flip + 1,
          "SubmitFlip accepts the documented -1 blank-buffer special index");

    // Closing retires exactly this handle. API and in-stream flip paths must both stop accepting it,
    // and a repeated Close reports INVALID_HANDLE instead of false success.
    const uint64_t before_close_flip_count = prosper_vo_flip_count();
    const uint64_t before_close_present_count = gpu::present_count();
    CHECK(close(handle, 0, 0, 0, 0, 0) == 0,
          "Close retires a live VideoOut handle");
    CHECK((uint32_t)close(handle, 0, 0, 0, 0, 0) == kInvalidHandle,
          "Close rejects an already-closed handle");
    CHECK((uint32_t)flip(handle, 0, 0, 0xC105ED, 0, 0) == kInvalidHandle,
          "SubmitFlip rejects a closed handle");
    prosper_vo_flip_from_gpu((uint32_t)handle, 0, 0, 0xC105ED);
    CHECK(prosper_vo_flip_count() == before_close_flip_count &&
              gpu::present_count() == before_close_present_count,
          "closed-handle flips cannot advance API or in-stream present state");
    uint8_t closed_status[0x40]; memset(closed_status, 0xEE, sizeof closed_status);
    CHECK((uint32_t)fstat(handle, (uint64_t)(uintptr_t)closed_status, 0, 0, 0, 0) ==
              kInvalidHandle,
          "status queries reject a closed handle");
    bool closed_status_untouched = true;
    for (uint8_t byte: closed_status) closed_status_untouched &= byte == 0xEE;
    CHECK(closed_status_untouched, "closed-handle status query leaves output untouched");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
