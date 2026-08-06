// videoout_present.hpp — headless present / swapchain over the libSceVideoOut display buffers.
//
// libSceVideoOut registers N framebuffers (the swapchain images) via RegisterBuffers2; the game
// then calls SubmitFlip(buffer_index) to present one. This module is the present side of that path:
// it tracks which buffer is currently scanned out (the "front" buffer) and lets the presented frame
// be read back (for verification now, and for a real window/offscreen Vulkan swapchain later).
//
// It reads the display surface from the prosper_vo_* buffer registry (hle_graphics.cpp), so it stays
// decoupled: sceVideoOutSubmitFlip just calls present_flip(). The framebuffer memory is guest GPU
// memory (unified, 1:1-mapped), so readback copies real pixels — nothing is faked. Until the renderer
// writes real frames into those buffers the contents are whatever the guest put there (zero), but the
// flip→present→scanout plumbing is exercised and testable end-to-end.
#pragma once
#include <cstdint>
#include <cstddef>
#include <memory>
#include <vector>

namespace prosper {

// One lock-coherent view of a registered guest scanout buffer and the attributes of its owning
// VideoOut set. The HLE registry supplies these snapshots to the present layer so an unregister or
// another set's registration cannot pair one buffer address with unrelated dimensions.
struct VideoOutBufferSnapshot {
    uint64_t address = 0;
    uint64_t pixel_format = 0;
    uint64_t generation = 0;
    int buffer_index = -1;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t tiling_mode = 0;
};

// Front-buffer selection and invalidation live under the registry mutex. A generation distinguishes
// a newly registered buffer from an older registration that occupied the same numeric slot.
bool videoout_select_buffer(int buffer_index, VideoOutBufferSnapshot& out);
bool videoout_front_snapshot(VideoOutBufferSnapshot& out);
bool videoout_display_snapshot(VideoOutBufferSnapshot& out);
int videoout_front_index();
void videoout_reset_front();

// Copy while holding the registry lease so UnregisterBuffers cannot return (and let the guest reuse
// the backing allocation) until the raw read completes. The expected-snapshot overload additionally
// rejects unregister/re-register ABA by checking the registration generation.
bool videoout_copy_buffer(const VideoOutBufferSnapshot& expected, std::vector<uint8_t>& out);
bool videoout_copy_front_buffer(std::vector<uint8_t>& out, VideoOutBufferSnapshot& metadata);
size_t videoout_copy_front_buffer(void* dst, size_t dst_cap,
                                  VideoOutBufferSnapshot* metadata = nullptr);

// The result of reading a registered scanout as an IMAGE rather than as bytes.
struct VideoOutLinearRead {
    std::vector<uint8_t> pixels;        // linear width*height*4 RGBA, de-swizzled
    VideoOutBufferSnapshot metadata{};
    // True when the buffer's contents have been observed to DIFFER from what they were at the
    // moment the guest registered it — the only evidence available in-process that the guest,
    // rather than whatever previously owned this memory, authored what is in there now. Sticky per
    // registration: once a write is seen, later frames stay authored even if one happens to match
    // the baseline. False also means "no baseline was taken" (the range was not readable at
    // registration), which is deliberately indistinguishable from "not written": both mean prosper
    // cannot say the guest wrote it, and callers must decline in either case.
    bool guest_authored = false;
    // True when the read covered the whole swizzle footprint rather than stopping at
    // width*height*4. False means the tail — real texels of the last block row — could not be
    // proven to belong to this buffer and was left zero, so the bottom-right of the image is
    // incomplete. Reported so a title showing a corrupt corner is diagnosable without a rebuild.
    bool padded_footprint = false;
};

// Read the currently flipped (front) scanout as an image: `out.pixels` is linear width*height*4
// RGBA, de-swizzled from the tiling mode the guest registered with the buffer. Use this wherever the
// result is treated as PIXELS; the raw copies above stay raw for diagnostics that want guest bytes
// exactly as they are. False means no buffer is flipped, its registration changed under the read, or
// its geometry is unusable.
bool videoout_read_front_linear(VideoOutLinearRead& out);

} // namespace prosper

namespace prosper::gpu {

// Where the frame a consumer just read actually came from. These are provenance labels, not quality
// labels, and one of them exists specifically so a verification gate can tell two cases apart:
//   Rendered     — prosper composited it from the guest's draws/dispatches.
//   GuestScanout — prosper composited NOTHING for this flip and republished the guest's own display
//                  buffer verbatim. It reaches consumers through the same present_write_frame path
//                  as a composited frame (it is what the swapchain shows), so without its own label
//                  `--require-composited-frame` would accept the exact frame it exists to reject.
//   RawScanout   — no renderer frame has ever been handed in, and the reader fell back to the guest
//                  display buffer directly (the pre-renderer boot path).
enum class PresentSource : uint8_t { None, Rendered, RawScanout, GuestScanout };

// The provenance a producer declares when it hands a frame to the present layer.
enum class PresentFrameOrigin : uint8_t { Composited, GuestScanout };

// One internally consistent readback plus the counters that identify it. The older query/readback
// calls remain for lightweight consumers, but tools that persist evidence should use this snapshot:
// querying frame_seq and pixels separately can race a renderer write and mislabel the saved image.
struct PresentSnapshot {
    PresentSource source = PresentSource::None;
    uint64_t source_seq = 0;      // rendered-frame sequence, or guest flip count for raw scanout
    uint64_t frame_seq = 0;
    uint64_t present_count = 0;
    int front_index = -1;
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<uint8_t> rgba;
};

// Lifetime-safe access to the current rendered frame without copying its pixels. The immutable
// storage remains valid after a newer frame is published or present_reset() is called.
struct PresentFrameLease {
    uint64_t frame_seq = 0;
    uint64_t guest_present_count = 0; // guest flip represented by this renderer publication
    uint32_t width = 0;
    uint32_t height = 0;
    std::shared_ptr<const std::vector<uint8_t>> rgba;
};

// Present the display buffer `buffer_index` (from sceVideoOutSubmitFlip). Records it as the front
// buffer and bumps the present counter. `flip_arg` is the guest's flip label (echoed in flip status).
void present_flip(int buffer_index, int64_t flip_arg);

// Receiving side of the present path: the back-half renderer hands its finished frame (w*h pixels,
// 4 bytes/pixel) to the present layer. present_readback then returns THIS frame — the real rendered
// pixels — instead of the raw guest display buffer, closing the loop shader → render →
// present_write_frame → present_readback. Thread-safe (renderer writes, present reads).
void present_write_frame(const void* pixels, uint32_t w, uint32_t h,
                         PresentFrameOrigin origin = PresentFrameOrigin::Composited);
void present_write_frame(std::shared_ptr<const std::vector<uint8_t>> pixels,
                         uint32_t w, uint32_t h,
                         PresentFrameOrigin origin = PresentFrameOrigin::Composited);

// True once a rendered frame has been handed in (readback returns rendered pixels, not the guest buffer).
bool present_has_frame();

int      present_front_index();   // currently-presented buffer index (-1 before the first flip)
uint64_t present_front_address(); // exact registered guest address currently selected for scanout
uint64_t present_count();         // total flips presented (guest-paced; can be far faster than render)
uint64_t present_frame_seq();     // count of rendered frames handed in (present_write_frame calls)
uint32_t present_width();
uint32_t present_height();
// Dimensions of the rendered frame handed in via present_write_frame (0 if none). Differ from
// present_width/height (the guest display dims) whenever the frame was rendered at a reduced size
// (PROSPER_RENDER_SCALE); a readback consumer must size its buffer from these when non-zero (#399).
uint32_t present_frame_width();
uint32_t present_frame_height();

// Copy the presented frame's pixels (width*height, 4 bytes/pixel) from the front buffer's guest
// memory into `dst`. Returns bytes written, or 0 if there is no surface / no flip yet / dst too small.
size_t present_readback(void* dst, size_t dst_cap);

// Acquire immutable shared ownership of the latest rendered frame. Unlike present_readback(), this
// does not copy pixels and never falls back to raw guest scanout.
bool present_acquire_rendered_frame(PresentFrameLease& out);

// Copy the best available frame and its identity as one observation. Prefers a rendered frame and
// falls back to the raw guest scanout. Returns false before either source is available.
bool present_snapshot(PresentSnapshot& out);

void present_reset();             // tests: clear front/count

} // namespace prosper::gpu
