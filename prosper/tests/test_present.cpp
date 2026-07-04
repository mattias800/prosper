// test_present — guards the libSceVideoOut flip→present path (videoout_present.cpp on top of the
// prosper_vo_* buffer registry). Registers a real triple-buffered surface whose framebuffers hold
// known pixel patterns, drives SubmitFlip through the NID registry, and asserts that the present
// layer scans out the flipped buffer and reads back exactly its pixels. This proves the present
// plumbing end-to-end headlessly — the surface the renderer will present real frames to.
#include "../src/hle/dispatch.hpp"
#include "../src/gpu/videoout_present.hpp"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

int main() {
    printf("== test_present ==\n");
    register_builtin_hle();
    gpu::present_reset();

    auto setba2 = Hle::lookup("PjS5uASwcV8");   // SetBufferAttribute2
    auto regb2  = Hle::lookup("rKBUtgRrtbk");   // RegisterBuffers2
    auto flip   = Hle::lookup(nid_hash("sceVideoOutSubmitFlip"));
    CHECK(setba2 && regb2 && flip, "VideoOut functions registered");
    if (!(setba2 && regb2 && flip)) { printf("== FAIL ==\n"); return 1; }

    // A small surface so the test framebuffers are cheap: 16x8 BGRA (512 bytes each), triple-buffered.
    const uint32_t W = 16, H = 8;
    const size_t   FB_BYTES = (size_t)W * H * 4;
    std::vector<uint8_t> fb0(FB_BYTES), fb1(FB_BYTES), fb2(FB_BYTES);
    for (size_t i = 0; i < FB_BYTES; i++) { fb0[i] = 0x11; fb1[i] = 0x22; fb2[i] = 0x33; }  // distinct fills

    // SetBufferAttribute2 with our test dimensions, then RegisterBuffers2 with the 3 framebuffers.
    uint8_t attr[0x50]; memset(attr, 0, sizeof attr);
    setba2((uint64_t)(uintptr_t)attr, 0x8000000000000000ull /*fmt*/, 0 /*tiling*/, W, H, 0 /*option*/);
    struct VOB { const void* data; const void* metadata; const void* reserved[2]; };
    VOB buffers[3] = { {fb0.data(),0,{0,0}}, {fb1.data(),0,{0,0}}, {fb2.data(),0,{0,0}} };
    uint64_t rc = regb2(0x1001, 0, 0, (uint64_t)(uintptr_t)buffers, 3, (uint64_t)(uintptr_t)attr);
    CHECK(rc == 0, "registered a 3-buffer 16x8 surface");
    CHECK(gpu::present_width() == W && gpu::present_height() == H, "present layer sees the surface dimensions");

    // Before any flip there is no front buffer.
    CHECK(gpu::present_front_index() == -1, "no front buffer before the first flip");
    std::vector<uint8_t> out(FB_BYTES, 0xEE);
    CHECK(gpu::present_readback(out.data(), out.size()) == 0, "readback yields nothing before a flip");

    // Flip buffer 1 -> it becomes the presented (scanned-out) frame.
    uint64_t base_count = gpu::present_count();
    flip(0x1001, 1 /*buffer*/, 0 /*mode*/, 0xABCD /*flipArg*/, 0, 0);
    CHECK(gpu::present_front_index() == 1, "flip selected buffer 1 as front");
    CHECK(gpu::present_count() == base_count + 1, "present count incremented");
    size_t got = gpu::present_readback(out.data(), out.size());
    CHECK(got == FB_BYTES, "readback produced a full frame");
    CHECK(out[0] == 0x22 && out[FB_BYTES - 1] == 0x22, "presented frame is buffer 1's pixels");

    // Flip buffer 2 -> scanout follows.
    flip(0x1001, 2, 0, 0, 0, 0);
    CHECK(gpu::present_front_index() == 2, "flip selected buffer 2 as front");
    gpu::present_readback(out.data(), out.size());
    CHECK(out[0] == 0x33, "presented frame now buffer 2's pixels");

    // Update buffer 2's contents (as a renderer would) and re-present: readback reflects live memory.
    for (size_t i = 0; i < FB_BYTES; i++) fb2[i] = 0x44;
    flip(0x1001, 2, 0, 0, 0, 0);
    gpu::present_readback(out.data(), out.size());
    CHECK(out[0] == 0x44 && out[100] == 0x44, "readback reflects updated framebuffer memory");

    // Out-of-range flip index leaves the front buffer unchanged (but still counts as a flip).
    flip(0x1001, 99, 0, 0, 0, 0);
    CHECK(gpu::present_front_index() == 2, "out-of-range flip index does not corrupt the front buffer");

    // Receiving side: the renderer hands a finished frame -> present_readback returns THOSE pixels
    // (the real rendered frame), not the raw guest buffer. This is shader->render->present->readback.
    CHECK(!gpu::present_has_frame(), "no rendered frame before the renderer hands one in");
    std::vector<uint8_t> rendered(FB_BYTES);
    for (size_t i = 0; i < FB_BYTES; i++) rendered[i] = (uint8_t)(i & 0xff);   // a distinctive gradient
    gpu::present_write_frame(rendered.data(), W, H);
    CHECK(gpu::present_has_frame(), "present_has_frame after write");
    std::fill(out.begin(), out.end(), 0xEE);
    size_t rb = gpu::present_readback(out.data(), out.size());
    CHECK(rb == FB_BYTES, "readback returns the rendered frame size");
    CHECK(memcmp(out.data(), rendered.data(), FB_BYTES) == 0, "readback returns the exact rendered pixels");
    // The rendered frame wins over the raw guest buffer even across flips.
    flip(0x1001, 0, 0, 0, 0, 0);
    gpu::present_readback(out.data(), out.size());
    CHECK(memcmp(out.data(), rendered.data(), FB_BYTES) == 0, "rendered frame takes precedence over the flipped guest buffer");

    // After reset, readback falls back to the guest buffer again.
    gpu::present_reset();
    CHECK(!gpu::present_has_frame(), "reset clears the rendered frame");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
