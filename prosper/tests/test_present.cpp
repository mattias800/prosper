// test_present — guards the libSceVideoOut flip→present path (videoout_present.cpp on top of the
// prosper_vo_* buffer registry). Registers a real triple-buffered surface whose framebuffers hold
// known pixel patterns, drives SubmitFlip through the NID registry, and asserts that the present
// layer scans out the flipped buffer and reads back exactly its pixels. This proves the present
// plumbing end-to-end headlessly — the surface the renderer will present real frames to.
#include "../src/hle/dispatch.hpp"
#include "../src/gpu/videoout_present.hpp"
#include "../src/host/lifecycle.hpp"
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <future>
#include <memory>
#include <thread>
#include <vector>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

using Hle8Fn = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t,
                            uint64_t, uint64_t, uint64_t, uint64_t);

int main() {
    printf("== test_present ==\n");
    register_builtin_hle();
    gpu::present_reset();

    auto open   = Hle::lookup(nid_hash("sceVideoOutOpen"));
    auto setba2 = reinterpret_cast<Hle8Fn>(
        Hle::lookup("PjS5uASwcV8"));             // SetBufferAttribute2
    auto regb2  = Hle::lookup("rKBUtgRrtbk");   // RegisterBuffers2
    auto unreg  = Hle::lookup("N5KDtkIjjJ4");   // UnregisterBuffers
    auto flip   = Hle::lookup(nid_hash("sceVideoOutSubmitFlip"));
    CHECK(open && setba2 && regb2 && unreg && flip, "VideoOut functions registered");
    if (!(open && setba2 && regb2 && unreg && flip)) { printf("== FAIL ==\n"); return 1; }
    const uint64_t handle = open(0, 0, 0, 0, 0, 0);
    CHECK((int64_t)handle > 0, "opened a live VideoOut handle");

    // A small surface so the test framebuffers are cheap: 16x8 BGRA (512 bytes each), triple-buffered.
    const uint32_t W = 16, H = 8;
    const size_t   FB_BYTES = (size_t)W * H * 4;
    std::vector<uint8_t> fb0(FB_BYTES), fb1(FB_BYTES), fb2(FB_BYTES);
    for (size_t i = 0; i < FB_BYTES; i++) { fb0[i] = 0x11; fb1[i] = 0x22; fb2[i] = 0x33; }  // distinct fills

    // SetBufferAttribute2 with our test dimensions, then RegisterBuffers2 with the 3 framebuffers.
    uint8_t attr[0x50]; memset(attr, 0, sizeof attr);
    setba2((uint64_t)(uintptr_t)attr, 0x8000000000000000ull /*fmt*/, 0 /*tiling*/, W, H,
           0 /*option*/, 0 /*dcc control*/, 0 /*dcc clear*/);
    struct VOB { const void* data; const void* metadata; const void* reserved[2]; };
    VOB buffers[3] = { {fb0.data(),0,{0,0}}, {fb1.data(),0,{0,0}}, {fb2.data(),0,{0,0}} };
    uint64_t rc = regb2(handle, 0, 0, (uint64_t)(uintptr_t)buffers, 3, (uint64_t)(uintptr_t)attr);
    CHECK(rc == 0, "registered a 3-buffer 16x8 surface");
    CHECK(gpu::present_width() == W && gpu::present_height() == H, "present layer sees the surface dimensions");

    // Before any flip there is no front buffer.
    CHECK(gpu::present_front_index() == -1, "no front buffer before the first flip");
    std::vector<uint8_t> out(FB_BYTES, 0xEE);
    CHECK(gpu::present_readback(out.data(), out.size()) == 0, "readback yields nothing before a flip");

    // Flip buffer 1 -> it becomes the presented (scanned-out) frame.
    uint64_t base_count = gpu::present_count();
    flip(handle, 1 /*buffer*/, 0 /*mode*/, 0xABCD /*flipArg*/, 0, 0);
    CHECK(gpu::present_front_index() == 1, "flip selected buffer 1 as front");
    CHECK(gpu::present_count() == base_count + 1, "present count incremented");
    size_t got = gpu::present_readback(out.data(), out.size());
    CHECK(got == FB_BYTES, "readback produced a full frame");
    CHECK(out[0] == 0x22 && out[FB_BYTES - 1] == 0x22, "presented frame is buffer 1's pixels");

    // Flip buffer 2 -> scanout follows.
    flip(handle, 2, 0, 0, 0, 0);
    CHECK(gpu::present_front_index() == 2, "flip selected buffer 2 as front");
    gpu::present_readback(out.data(), out.size());
    CHECK(out[0] == 0x33, "presented frame now buffer 2's pixels");

    // Update buffer 2's contents (as a renderer would) and re-present: readback reflects live memory.
    for (size_t i = 0; i < FB_BYTES; i++) fb2[i] = 0x44;
    flip(handle, 2, 0, 0, 0, 0);
    gpu::present_readback(out.data(), out.size());
    CHECK(out[0] == 0x44 && out[100] == 0x44, "readback reflects updated framebuffer memory");
    gpu::PresentSnapshot raw_snap;
    CHECK(gpu::present_snapshot(raw_snap), "raw present snapshot is available before rendering");
    CHECK(raw_snap.source == gpu::PresentSource::RawScanout && raw_snap.front_index == 2 &&
          raw_snap.width == W && raw_snap.height == H && raw_snap.rgba[0] == 0x44,
          "raw snapshot identifies and copies the selected guest scanout");

    // Register a second set with different dimensions. Flipping between the sets must select one
    // coherent address + geometry snapshot; otherwise the raw copy would use the latest set's size
    // with the older set's address (or vice versa).
    const uint32_t SMALL_W = 4, SMALL_H = 2;
    const size_t SMALL_BYTES = (size_t)SMALL_W * SMALL_H * 4;
    std::vector<uint8_t> fb4(SMALL_BYTES, 0x66), fb5(SMALL_BYTES, 0x77);
    uint8_t small_attr[0x50]; memset(small_attr, 0, sizeof small_attr);
    setba2((uint64_t)(uintptr_t)small_attr, 0x8000000000000000ull,
           0 /*tiling*/, SMALL_W, SMALL_H, 0 /*option*/, 0 /*dcc control*/, 0 /*dcc clear*/);
    VOB small_buffers[2] = { {fb4.data(),0,{0,0}}, {fb5.data(),0,{0,0}} };
    rc = regb2(handle, 1 /*set*/, 4 /*start*/, (uint64_t)(uintptr_t)small_buffers, 2,
               (uint64_t)(uintptr_t)small_attr);
    CHECK(rc == 0, "registered a second 2-buffer 4x2 surface");
    flip(handle, 4, 0, 0, 0, 0);
    std::vector<uint8_t> small_out(SMALL_BYTES, 0xEE);
    got = gpu::present_readback(small_out.data(), small_out.size());
    CHECK(gpu::present_width() == SMALL_W && gpu::present_height() == SMALL_H &&
          got == SMALL_BYTES && small_out.front() == 0x66 && small_out.back() == 0x66,
          "set 1 flip reads its 4x2 buffer with matching geometry");

    // Unregister invalidates the front identity under the same registry lock. Reusing the numeric
    // slot receives a new generation and must not present its new backing until another guest flip.
    CHECK(unreg(handle, 1 /*set*/, 0, 0, 0, 0) == 0 &&
          gpu::present_front_index() == -1,
          "unregistering the front set atomically clears the front identity");
    std::vector<uint8_t> fb4_reused(SMALL_BYTES, 0x88), fb5_reused(SMALL_BYTES, 0x99);
    VOB reused_buffers[2] = {
        {fb4_reused.data(),0,{0,0}}, {fb5_reused.data(),0,{0,0}}
    };
    CHECK(regb2(handle, 1 /*same set*/, 4 /*same start*/,
                (uint64_t)(uintptr_t)reused_buffers, 2,
                (uint64_t)(uintptr_t)small_attr) == 0,
          "re-registered the same slots with new backing");
    std::fill(small_out.begin(), small_out.end(), 0xEE);
    gpu::PresentSnapshot no_flip_snap;
    CHECK(gpu::present_front_index() == -1 &&
          gpu::present_readback(small_out.data(), small_out.size()) == 0 &&
          !gpu::present_snapshot(no_flip_snap),
          "re-registering a numeric slot does not present new backing without a flip");
    flip(handle, 4, 0, 0, 0, 0);
    got = gpu::present_readback(small_out.data(), small_out.size());
    CHECK(got == SMALL_BYTES && small_out.front() == 0x88 && small_out.back() == 0x88,
          "a new flip presents the re-registered generation");

    flip(handle, 1, 0, 0, 0, 0);
    std::fill(out.begin(), out.end(), 0xEE);
    got = gpu::present_readback(out.data(), out.size());
    CHECK(gpu::present_width() == W && gpu::present_height() == H &&
          got == FB_BYTES && out.front() == 0x22 && out.back() == 0x22,
          "set 0 flip restores its 16x8 buffer with matching geometry");

    // The host pause gate lives after flip publication: the UI can keep showing the new front
    // buffer while the guest caller waits, then resume it without interrupting mid-frame.
    prosper_reset_stop();
    prosper_set_paused(true);
    const uint64_t paused_count = gpu::present_count();
    auto paused_flip = std::async(std::launch::async, [&] {
        flip(handle, 1, 0, 0xCAFE, 0, 0);
        return true;
    });
    const auto publish_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (gpu::present_count() == paused_count &&
           std::chrono::steady_clock::now() < publish_deadline)
        std::this_thread::yield();
    CHECK(gpu::present_count() == paused_count + 1 && gpu::present_front_index() == 1,
          "paused flip publishes its completed boundary");
    CHECK(paused_flip.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout,
          "paused flip holds the guest caller after publication");
    prosper_set_paused(false);
    CHECK(paused_flip.wait_for(std::chrono::seconds(1)) == std::future_status::ready &&
              paused_flip.get(),
          "resume releases the guest flip caller");
    prosper_reset_stop();

    // Out-of-range flip index leaves the front buffer unchanged.
    flip(handle, 99, 0, 0, 0, 0);
    CHECK(gpu::present_front_index() == 1, "out-of-range flip index does not corrupt the front buffer");

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
    gpu::PresentSnapshot snap;
    CHECK(gpu::present_snapshot(snap), "atomic present snapshot is available");
    CHECK(snap.source == gpu::PresentSource::Rendered && snap.source_seq == gpu::present_frame_seq(),
          "snapshot identifies the rendered publication");
    CHECK(snap.width == W && snap.height == H && snap.rgba == rendered,
          "snapshot metadata and pixels describe the same frame");

    // The shared publication path must retain the renderer's exact immutable allocation. A lease
    // keeps the old frame alive even after a newer frame replaces it or the present state resets.
    auto shared_pixels = std::make_shared<const std::vector<uint8_t>>(rendered);
    const uint8_t* shared_data = shared_pixels->data();
    gpu::present_write_frame(shared_pixels, W, H);
    gpu::PresentFrameLease lease;
    CHECK(gpu::present_acquire_rendered_frame(lease) && lease.rgba &&
          lease.rgba->data() == shared_data,
          "shared rendered publication preserves allocation identity without copying");
    const uint64_t leased_seq = lease.frame_seq;
    const uint64_t leased_present_count = lease.guest_present_count;
    CHECK(leased_present_count == gpu::present_count(),
          "rendered-frame lease records the guest flip it represents");
    std::vector<uint8_t> replacement(FB_BYTES, 0x5A);
    flip(handle, 0, 0, 0, 0, 0);
    gpu::present_write_frame(replacement.data(), W, H);
    CHECK(lease.frame_seq == leased_seq && lease.rgba->data() == shared_data &&
          lease.guest_present_count == leased_present_count && *lease.rgba == rendered,
          "rendered-frame lease remains valid after a newer publication");
    gpu::PresentFrameLease replacement_lease;
    CHECK(gpu::present_acquire_rendered_frame(replacement_lease) &&
          replacement_lease.guest_present_count == gpu::present_count() &&
          replacement_lease.guest_present_count > lease.guest_present_count,
          "newer rendered publication records its newer guest flip");
    // The rendered frame wins over the raw guest buffer even across flips.
    gpu::present_readback(out.data(), out.size());
    CHECK(memcmp(out.data(), replacement.data(), FB_BYTES) == 0,
          "latest rendered frame takes precedence over the flipped guest buffer");

    // After reset, readback falls back to the guest buffer again.
    gpu::present_reset();
    CHECK(!gpu::present_has_frame(), "reset clears the rendered frame");
    CHECK(lease.rgba->data() == shared_data && *lease.rgba == rendered,
          "rendered-frame lease remains valid after present reset");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
