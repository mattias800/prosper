// test_videoout — guards the libSceVideoOut HLE (hle_graphics.cpp): the query/config functions must
// return real, self-consistent 1080p60 values (not zeroed stubs), and the display-buffer registration
// (swapchain scaffolding) must record the surface the game set up. Drives the functions through the
// NID registry exactly as the guest does, then asserts the reported display + recorded buffers.
#include "../src/hle/dispatch.hpp"
#include "../src/gpu/videoout_present.hpp"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <thread>

using namespace prosper;

extern "C" int      prosper_vo_buffer_count();
extern "C" uint32_t prosper_vo_display_width();
extern "C" uint32_t prosper_vo_display_height();
extern "C" uint64_t prosper_vo_display_format();
extern "C" uint64_t prosper_vo_buffer_addr(int i);

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

using Hle7Fn = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t,
                            uint64_t, uint64_t, uint64_t);

int main() {
    printf("== test_videoout ==\n");
    register_builtin_hle();

    auto res    = Hle::lookup(nid_hash("sceVideoOutGetResolutionStatus"));
    auto vbl    = Hle::lookup(nid_hash("sceVideoOutGetVblankStatus"));
    auto cap    = Hle::lookup(nid_hash("sceVideoOutGetDeviceCapabilityInfo"));
    auto issup  = Hle::lookup("Nv8c-Kb+DUM");   // IsOutputSupported
    auto setba  = Hle::lookup(nid_hash("sceVideoOutSetBufferAttribute"));
    auto regb   = Hle::lookup(nid_hash("sceVideoOutRegisterBuffers"));
    auto setba2 = Hle::lookup("PjS5uASwcV8");   // SetBufferAttribute2
    auto regb2  = Hle::lookup("rKBUtgRrtbk");   // RegisterBuffers2
    auto unreg  = Hle::lookup("N5KDtkIjjJ4");   // UnregisterBuffers
    auto cfg    = Hle::lookup("w0hLuNarQxY");   // ConfigureOutput
    auto flip   = Hle::lookup(nid_hash("sceVideoOutSubmitFlip"));
    auto fstat  = Hle::lookup(nid_hash("sceVideoOutGetFlipStatus"));
    CHECK(res && vbl && cap && issup && setba && regb && setba2 && regb2 && unreg && cfg && flip && fstat,
          "VideoOut functions registered");
    if (!(res && vbl && cap && issup && setba && regb && setba2 && regb2 && unreg && cfg && flip && fstat)) {
        printf("== FAIL ==\n"); return 1;
    }

    // #394 F3: before any completed flip, -1 is the ABI's "none yet" sentinel for both the argument
    // and current buffer. Zero is valid and used to make a frame pacer advance prematurely.
    uint8_t initial_fs[0x48]; memset(initial_fs, 0xEE, sizeof initial_fs);
    CHECK(fstat(0x1001, (uint64_t)(uintptr_t)initial_fs, 0, 0, 0, 0) == 0,
          "initial GetFlipStatus succeeds");
    CHECK(*(uint64_t*)(initial_fs + 0x00) == 0 &&
          *(int64_t*)(initial_fs + 0x18) == -1 &&
          *(int32_t*)(initial_fs + 0x38) == -1,
          "initial flip status reports count=0 and -1 argument/buffer sentinels");
    bool initial_tail_untouched = true;
    for (size_t i = 0x40; i < sizeof initial_fs; ++i)
        initial_tail_untouched &= initial_fs[i] == 0xEE;
    CHECK(initial_tail_untouched, "GetFlipStatus writes exactly its 0x40-byte ABI struct");

    // Resolution: real 1080p @ 59.94Hz (refresh enum 3), not zeroed. The ABI struct is 0x30
    // bytes: flags/reserved0 at 0x1c and reserved1[3] at 0x20 must not retain caller garbage.
    uint8_t rs[0x38]; memset(rs, 0xEE, sizeof rs);
    res(0x1001, (uint64_t)(uintptr_t)rs, 0, 0, 0, 0);
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
    vbl(0x1001, (uint64_t)(uintptr_t)vb, 0, 0, 0, 0); uint64_t c0 = *(uint64_t*)vb;
    CHECK(c0 > 0, "first vblank query reports a live non-zero counter");
    vbl(0x1001, (uint64_t)(uintptr_t)vb, 0, 0, 0, 0); uint64_t c1 = *(uint64_t*)vb;
    CHECK(c1 - c0 <= 1, "vblank counter does NOT tick per poll (immediate re-poll ~same period)");
    std::this_thread::sleep_for(std::chrono::milliseconds(40));   // > 2 vblank periods
    vbl(0x1001, (uint64_t)(uintptr_t)vb, 0, 0, 0, 0); uint64_t c2 = *(uint64_t*)vb;
    CHECK(c2 > c1, "vblank counter advances across a >1-period sleep");
    CHECK(*(uint64_t*)(vb + 0x08) > 0, "vblank processTime filled (Kyty layout @0x08)");

    // Device capability: plain SDR display (capability 0).
    uint64_t dc = 0xDEAD;
    cap(0x1001, (uint64_t)(uintptr_t)&dc, 0, 0, 0, 0);
    CHECK(dc == 0, "device capability = 0 (SDR)");

    // IsOutputSupported → supported.
    CHECK(issup(0x1001, 0xd000000a, 0, 0, 0, 0) == 1, "IsOutputSupported returns 1");

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
    CHECK((uint32_t)regb(0x1001, 15, (uint64_t)(uintptr_t)legacy_buffers, 2,
                         (uint64_t)(uintptr_t)legacy_attr, 0) == 0x80290001u,
          "legacy RegisterBuffers rejects a range past slot 15");
    uint64_t legacy_group = regb(0x1001, 2, (uint64_t)(uintptr_t)legacy_buffers, 2,
                                 (uint64_t)(uintptr_t)legacy_attr, 0);
    CHECK(legacy_group == 0, "legacy RegisterBuffers returns the lowest free attribute group");
    CHECK(prosper_vo_buffer_count() == 4 &&
          prosper_vo_buffer_addr(2) == (uint64_t)(uintptr_t)legacy_fb2 &&
          prosper_vo_buffer_addr(3) == (uint64_t)(uintptr_t)legacy_fb3,
          "legacy RegisterBuffers records the flat framebuffer array at the requested slots");
    CHECK(prosper_vo_display_width() == 1280 && prosper_vo_display_height() == 720 &&
          prosper_vo_display_format() == 0x80002200u,
          "legacy RegisterBuffers records Gen4 surface geometry and format");
    CHECK((uint32_t)regb(0x1001, 3, (uint64_t)(uintptr_t)legacy_buffers, 2,
                         (uint64_t)(uintptr_t)legacy_attr, 0) == 0x80290010u,
          "legacy RegisterBuffers rejects an overlapping slot range");
    CHECK(unreg(0x1001, legacy_group, 0, 0, 0, 0) == 0 &&
          prosper_vo_buffer_count() == 0 && prosper_vo_display_width() == 0,
          "legacy buffer group unregisters cleanly before Gen5 registration");

    // SetBufferAttribute2 fills the 0x50 attribute struct from (fmt, tiling, w, h, option).
    uint8_t attr[0x50]; memset(attr, 0xEE, sizeof attr);
    uint64_t fmt = 0x8000000000000000ull;   // the PS5 format the game requests (A8R8G8B8 sRGB)
    setba2((uint64_t)(uintptr_t)attr, fmt, 0 /*tiling*/, 1920, 1080, 0 /*option*/);
    CHECK(*(uint32_t*)(attr + 0x0c) == 1920 && *(uint32_t*)(attr + 0x10) == 1080, "attribute width/height set");
    CHECK(*(uint64_t*)(attr + 0x20) == fmt, "attribute pixel_format set");
    CHECK(*(uint32_t*)(attr + 0x04) == 0, "attribute tiling_mode set");

    // ConfigureOutput accepts the (single advertised) mode.
    CHECK(cfg(0x1001, 1, 0, 0, 0, 0) == 0, "ConfigureOutput accepted");

    // RegisterBuffers2: triple-buffered surface. buffers = array of VideoOutBuffers {data, ...}.
    struct VOB { const void* data; const void* metadata; const void* reserved[2]; };
    uint8_t fb0[16], fb1[16], fb2[16];
    VOB buffers[3] = { {fb0,0,{0,0}}, {fb1,0,{0,0}}, {fb2,0,{0,0}} };
    uint64_t rc = regb2(0x1001, 0 /*set*/, 0 /*start*/, (uint64_t)(uintptr_t)buffers, 3, (uint64_t)(uintptr_t)attr);
    CHECK(rc == 0, "RegisterBuffers2 accepted 3 buffers");

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
    rc = regb2(0x1001, 0 /*duplicate set*/, 4 /*free start*/,
               (uint64_t)(uintptr_t)candidate_buffers, 2, (uint64_t)(uintptr_t)attr);
    CHECK((uint32_t)rc == 0x8029000au,
          "RegisterBuffers2 rejects a duplicate active set identifier");

    // A different set may not claim any slot already owned by set 0. The whole range is validated
    // before mutation, so this failure cannot replace either overlapping address or reserve set 1.
    rc = regb2(0x1001, 1 /*new set*/, 1 /*occupied start*/,
               (uint64_t)(uintptr_t)candidate_buffers, 2, (uint64_t)(uintptr_t)attr);
    CHECK((uint32_t)rc == 0x80290010u,
          "RegisterBuffers2 rejects slots owned by another active set");
    CHECK(prosper_vo_buffer_count() == 3 &&
          prosper_vo_buffer_addr(0) == (uint64_t)(uintptr_t)fb0 &&
          prosper_vo_buffer_addr(1) == (uint64_t)(uintptr_t)fb1 &&
          prosper_vo_buffer_addr(2) == (uint64_t)(uintptr_t)fb2,
          "failed registrations leave all existing slots unchanged");

    uint8_t fs[0x40]; memset(fs, 0xEE, sizeof fs);
    CHECK(flip(0x1001, 2 /*buffer*/, 0 /*mode*/, 0x12345678 /*flipArg*/, 0, 0) == 0,
          "SubmitFlip accepted buffer 2");
    fstat(0x1001, (uint64_t)(uintptr_t)fs, 0, 0, 0, 0);
    CHECK(*(uint64_t*)(fs + 0x00) == 1, "flip status count increments");
    CHECK(*(int64_t*) (fs + 0x18) == 0x12345678, "flip status reports the submitted flipArg");
    CHECK(*(int32_t*) (fs + 0x38) == 2, "flip status reports the submitted currentBuffer");

    // Range validation: out-of-range buffer counts are rejected.
    CHECK(regb2(0x1001, 2, 0, (uint64_t)(uintptr_t)buffers, 99, (uint64_t)(uintptr_t)attr) != 0,
          "RegisterBuffers2 rejects an out-of-range buffer_num");

    CHECK((uint32_t)unreg(0x1001, 3 /*unregistered set*/, 0, 0, 0, 0) == 0x8029000au,
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
    uint64_t high_rc = regb2(0x1001, 1 /*set*/, 4 /*start*/,
                             (uint64_t)(uintptr_t)high_buffers, 2,
                             (uint64_t)(uintptr_t)high_attr);
    CHECK(high_rc == 0 && prosper_vo_buffer_count() == 6 &&
              prosper_vo_display_width() == 1280 && prosper_vo_display_height() == 720,
          "registry tracks two independently owned buffer sets");
    flip(0x1001, 2 /*set 0 buffer*/, 0, 0, 0, 0);
    CHECK(gpu::present_width() == 1920 && gpu::present_height() == 1080,
          "present uses set 0 geometry when an older set's buffer is flipped");
    flip(0x1001, 4 /*set 1 buffer*/, 0, 0, 0, 0);
    CHECK(gpu::present_width() == 1280 && gpu::present_height() == 720,
          "present uses set 1 geometry when its buffer is flipped");
    CHECK(unreg(0x1001, 1 /*set*/, 0, 0, 0, 0) == 0 &&
              prosper_vo_buffer_count() == 3 &&
              prosper_vo_buffer_addr(0) == (uint64_t)(uintptr_t)fb0 &&
              prosper_vo_buffer_addr(2) == (uint64_t)(uintptr_t)fb2 &&
              prosper_vo_display_width() == 1920 && prosper_vo_display_height() == 1080,
          "unregistering one set restores the remaining set's range and geometry");
    flip(0x1001, 2 /*remaining set 0 buffer*/, 0, 0, 0, 0);
    CHECK(gpu::present_width() == 1920 && gpu::present_height() == 1080,
          "present continues with the surviving set after unregister");
    CHECK(unreg(0x1001, 0 /*set*/, 0, 0, 0, 0) == 0 &&
              prosper_vo_buffer_count() == 0 && prosper_vo_buffer_addr(0) == 0 &&
              prosper_vo_display_width() == 0 && prosper_vo_display_height() == 0,
          "unregistering the remaining set clears the registry and geometry");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
