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
extern "C" uint64_t prosper_vo_flip_count();
extern "C" int prosper_vo_flip_rate();
extern "C" void prosper_vo_flip_from_gpu(uint32_t handle, int32_t bufidx,
                                           uint32_t flip_mode, int64_t flip_arg);

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

using Hle7Fn = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t,
                            uint64_t, uint64_t, uint64_t);

int main() {
    printf("== test_videoout ==\n");
    register_builtin_hle();

    auto open   = Hle::lookup(nid_hash("sceVideoOutOpen"));
    auto close  = Hle::lookup(nid_hash("sceVideoOutClose"));
    auto res    = Hle::lookup(nid_hash("sceVideoOutGetResolutionStatus"));
    auto vbl    = Hle::lookup(nid_hash("sceVideoOutGetVblankStatus"));
    auto cap    = Hle::lookup(nid_hash("sceVideoOutGetDeviceCapabilityInfo"));
    auto issup  = Hle::lookup("Nv8c-Kb+DUM");   // IsOutputSupported
    auto setba  = Hle::lookup(nid_hash("sceVideoOutSetBufferAttribute"));
    auto regb   = Hle::lookup(nid_hash("sceVideoOutRegisterBuffers"));
    auto setba2 = Hle::lookup("PjS5uASwcV8");   // SetBufferAttribute2
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
          *(int64_t*)(initial_fs + 0x18) == -1 &&
          *(int32_t*)(initial_fs + 0x38) == -1,
          "initial flip status reports count=0 and -1 argument/buffer sentinels");
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

    // SetBufferAttribute2 fills the 0x50 attribute struct from (fmt, tiling, w, h, option).
    uint8_t attr[0x50]; memset(attr, 0xEE, sizeof attr);
    uint64_t fmt = 0x8000000000000000ull;   // the PS5 format the game requests (A8R8G8B8 sRGB)
    setba2((uint64_t)(uintptr_t)attr, fmt, 0 /*tiling*/, 1920, 1080, 0 /*option*/);
    CHECK(*(uint32_t*)(attr + 0x0c) == 1920 && *(uint32_t*)(attr + 0x10) == 1080, "attribute width/height set");
    CHECK(*(uint64_t*)(attr + 0x20) == fmt, "attribute pixel_format set");
    CHECK(*(uint32_t*)(attr + 0x04) == 0, "attribute tiling_mode set");

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
    CHECK(*(int64_t*) (fs + 0x18) == 0x12345678, "flip status reports the submitted flipArg");
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
