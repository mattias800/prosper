// videoout_present.cpp — see videoout_present.hpp.
#include "videoout_present.hpp"
#include "gpu_timeline.hpp"
#include "host/lifecycle.hpp"
#include <atomic>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace prosper::gpu {
namespace {
std::atomic<uint64_t> g_present_count{0};
std::atomic<uint64_t> g_frame_seq{0};   // # of rendered frames handed in via present_write_frame

// The rendered frame the back-half hands us (the "scanout" image). Guarded by a mutex because the
// renderer thread writes it while guest threads / tests read it via present_readback.
std::mutex           g_frame_mx;
std::shared_ptr<const std::vector<uint8_t>> g_frame; // w*h*4 bytes when a frame is present
uint32_t             g_frame_w = 0, g_frame_h = 0;
uint64_t             g_frame_guest_present_count = 0;
std::atomic<bool>    g_have_frame{false};

bool current_scanout(VideoOutBufferSnapshot& out, int* index = nullptr) {
    if (videoout_front_snapshot(out)) {
        if (index) *index = out.buffer_index;
        return true;
    }
    if (index) *index = -1;
    return videoout_display_snapshot(out);
}
}

void present_write_frame(const void* pixels, uint32_t w, uint32_t h) {
    if (!pixels || !w || !h) return;
    size_t bytes = (size_t)w * h * 4;
    auto owned = std::make_shared<std::vector<uint8_t>>(bytes);
    std::memcpy(owned->data(), pixels, bytes);
    present_write_frame(std::move(owned), w, h);
}

void present_write_frame(std::shared_ptr<const std::vector<uint8_t>> pixels,
                         uint32_t w, uint32_t h) {
    if (!pixels || !w || !h || pixels->size() != (size_t)w * h * 4) return;
    std::lock_guard<std::mutex> lk(g_frame_mx);
    g_frame = std::move(pixels);
    g_frame_w = w; g_frame_h = h;
    g_frame_guest_present_count = g_present_count.load(std::memory_order_relaxed);
    g_have_frame.store(true, std::memory_order_release);
    g_frame_seq.fetch_add(1, std::memory_order_relaxed);
}

uint64_t present_frame_seq() { return g_frame_seq.load(std::memory_order_relaxed); }

bool present_has_frame() { return g_have_frame.load(std::memory_order_acquire); }

void present_flip(int buffer_index, int64_t flip_arg) {
    // Only accept a buffer the game actually registered; otherwise leave the front unchanged (an
    // invalid index shouldn't corrupt scanout state).
    VideoOutBufferSnapshot selected;
    if (!videoout_select_buffer(buffer_index, selected))
        current_scanout(selected);
    uint64_t n = g_present_count.fetch_add(1, std::memory_order_relaxed);
    record_gpu_timeline_present(n + 1, buffer_index, flip_arg,
                                selected.width, selected.height);
    // PROSPER_DUMP_SCANOUT=<dir>: dump the RAW guest display buffer the game just flipped (what the game
    // actually puts on screen — vs our execute_and_present composite render). First 8 flips + every 60th.
    // Written as raw w*h*4 bytes; convert offline. This reveals whether the game's real display holds the
    // title/scene even when the captured draws are a no-op composite.
    if (const char* dir = getenv("PROSPER_DUMP_SCANOUT"); dir && (n < 8 || n % 60 == 0)) {
        std::vector<uint8_t> raw;
        if (videoout_copy_buffer(selected, raw)) {
            char fn[512]; snprintf(fn, sizeof fn, "%s/scanout_%04llu_buf%d.rgba", dir,
                                   (unsigned long long)n, selected.buffer_index);
            if (FILE* f = fopen(fn, "wb")) {
                fwrite(raw.data(), 1, raw.size(), f);
                fclose(f);
            }
        }
    }

    // The desktop app requests pause from its UI thread. Wait only after the completed flip is
    // fully published and all local locks are gone: the app can keep presenting that last frame
    // and pumping SDL events, while the guest resumes from an exact display boundary.
    prosper_wait_while_paused();
}

int      present_front_index() { return videoout_front_index(); }
uint64_t present_front_address() {
    VideoOutBufferSnapshot snapshot;
    return current_scanout(snapshot) ? snapshot.address : 0;
}
uint64_t present_count()       { return g_present_count.load(std::memory_order_relaxed); }
uint32_t present_width()       { VideoOutBufferSnapshot s; return current_scanout(s) ? s.width : 0; }
uint32_t present_height()      { VideoOutBufferSnapshot s; return current_scanout(s) ? s.height : 0; }
// Dimensions of the RENDERED frame handed in via present_write_frame (0 if none present). Distinct from
// present_width/height (the guest DISPLAY dims): under PROSPER_RENDER_SCALE the rendered frame is smaller
// than the display, so a readback consumer (screenshot) must size its buffer from THESE when a rendered
// frame exists — else present_readback returns g_frame.size() != display*4 and the exact-size guard drops
// every frame silently (#399). Fall back to present_width/height only on the raw-scanout path.
uint32_t present_frame_width()  { std::lock_guard<std::mutex> lk(g_frame_mx); return g_frame_w; }
uint32_t present_frame_height() { std::lock_guard<std::mutex> lk(g_frame_mx); return g_frame_h; }

size_t present_readback(void* dst, size_t dst_cap) {
    if (!dst) return 0;
    // Prefer the real rendered frame from the back-half, if one has been handed in.
    if (g_have_frame.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lk(g_frame_mx);
        if (g_frame && !g_frame->empty() && dst_cap >= g_frame->size()) {
            std::memcpy(dst, g_frame->data(), g_frame->size());
            return g_frame->size();
        }
        return 0;
    }
    // Fallback: scan out the raw guest display buffer (contents are whatever the guest put there).
    return videoout_copy_front_buffer(dst, dst_cap);
}

bool present_acquire_rendered_frame(PresentFrameLease& out) {
    out = {};
    if (!g_have_frame.load(std::memory_order_acquire)) return false;
    std::lock_guard<std::mutex> lk(g_frame_mx);
    if (!g_frame || g_frame->empty() || !g_frame_w || !g_frame_h) return false;
    out.frame_seq = g_frame_seq.load(std::memory_order_relaxed);
    out.guest_present_count = g_frame_guest_present_count;
    out.width = g_frame_w;
    out.height = g_frame_h;
    out.rgba = g_frame;
    return true;
}

bool present_snapshot(PresentSnapshot& out) {
    out = {};
    // frame_seq is incremented while holding this mutex, after the pixels and dimensions are
    // installed, so the copied bytes and identity describe the same renderer publication.
    if (g_have_frame.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lk(g_frame_mx);
        if (!g_frame || g_frame->empty() || !g_frame_w || !g_frame_h) return false;
        out.source = PresentSource::Rendered;
        out.frame_seq = g_frame_seq.load(std::memory_order_relaxed);
        out.source_seq = out.frame_seq;
        out.present_count = g_present_count.load(std::memory_order_relaxed);
        out.front_index = videoout_front_index();
        out.width = g_frame_w;
        out.height = g_frame_h;
        out.rgba = *g_frame;
        return true;
    }

    VideoOutBufferSnapshot scanout;
    if (!videoout_copy_front_buffer(out.rgba, scanout)) return false;
    out.source = PresentSource::RawScanout;
    out.present_count = g_present_count.load(std::memory_order_relaxed);
    out.source_seq = out.present_count;
    out.frame_seq = g_frame_seq.load(std::memory_order_relaxed);
    out.front_index = scanout.buffer_index;
    out.width = scanout.width;
    out.height = scanout.height;
    return true;
}

void present_reset() {
    videoout_reset_front();
    g_present_count.store(0, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(g_frame_mx);
    g_frame.reset(); g_frame_w = g_frame_h = 0;
    g_frame_guest_present_count = 0;
    g_frame_seq.store(0, std::memory_order_relaxed);   // #399: reset the rendered-frame counter too
    g_have_frame.store(false, std::memory_order_release);
}

} // namespace prosper::gpu
