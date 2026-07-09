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

namespace prosper::gpu {

// Present the display buffer `buffer_index` (from sceVideoOutSubmitFlip). Records it as the front
// buffer and bumps the present counter. `flip_arg` is the guest's flip label (echoed in flip status).
void present_flip(int buffer_index, int64_t flip_arg);

// Receiving side of the present path: the back-half renderer hands its finished frame (w*h pixels,
// 4 bytes/pixel) to the present layer. present_readback then returns THIS frame — the real rendered
// pixels — instead of the raw guest display buffer, closing the loop shader → render →
// present_write_frame → present_readback. Thread-safe (renderer writes, present reads).
void present_write_frame(const void* pixels, uint32_t w, uint32_t h);

// True once a rendered frame has been handed in (readback returns rendered pixels, not the guest buffer).
bool present_has_frame();

int      present_front_index();   // currently-presented buffer index (-1 before the first flip)
uint64_t present_count();         // total flips presented (guest-paced; can be far faster than render)
uint64_t present_frame_seq();     // count of rendered frames handed in (present_write_frame calls)
uint32_t present_width();
uint32_t present_height();

// Copy the presented frame's pixels (width*height, 4 bytes/pixel) from the front buffer's guest
// memory into `dst`. Returns bytes written, or 0 if there is no surface / no flip yet / dst too small.
size_t present_readback(void* dst, size_t dst_cap);

void present_reset();             // tests: clear front/count

} // namespace prosper::gpu
