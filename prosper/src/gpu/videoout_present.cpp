// videoout_present.cpp — see videoout_present.hpp.
#include "videoout_present.hpp"
#include "gpu_execute.hpp"   // guest_readable (the flip buffer address comes from the guest)
#include "tile.hpp"          // detile_surface (registered display buffers are GPU-tiled)
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

// The display surface lives in the libSceVideoOut buffer registry (hle_graphics.cpp), exposed via
// these C hooks. Reading through them keeps the present layer decoupled from the HLE.
extern "C" int      prosper_vo_buffer_count();
extern "C" uint32_t prosper_vo_display_width();
extern "C" uint32_t prosper_vo_display_height();
extern "C" uint64_t prosper_vo_display_format();
extern "C" uint32_t prosper_vo_display_tiling();
extern "C" uint64_t prosper_vo_buffer_addr(int i);

namespace prosper::gpu {
namespace {
std::atomic<int>      g_front{-1};
std::atomic<uint64_t> g_present_count{0};

// PROSPER_SCANOUT_DUMP=<dir>: write flipped frames as 24-bit BMPs (first 8, then every 60th) so the
// scanout path is verifiable without a display. Minimal writer (rows bottom-up, BGR, 4-byte padded).
void scanout_dump_bmp(const uint8_t* px, uint32_t W, uint32_t H, uint64_t n) {
    static const char* dir = getenv("PROSPER_SCANOUT_DUMP");
    if (!dir || (n >= 8 && n % 60 != 0)) return;
    char fn[512]; snprintf(fn, sizeof fn, "%s/scanout_%04llu.bmp", dir, (unsigned long long)n);
    FILE* f = fopen(fn, "wb"); if (!f) return;
    const uint32_t rowpad = (4 - (W * 3) % 4) % 4, dataSize = (W * 3 + rowpad) * H, fileSize = 54 + dataSize;
    auto u16 = [&](uint32_t v){ uint8_t b[2] = {(uint8_t)v, (uint8_t)(v >> 8)}; fwrite(b, 1, 2, f); };
    auto u32 = [&](uint32_t v){ uint8_t b[4] = {(uint8_t)v, (uint8_t)(v>>8), (uint8_t)(v>>16), (uint8_t)(v>>24)}; fwrite(b, 1, 4, f); };
    fputc('B', f); fputc('M', f); u32(fileSize); u32(0); u32(54);
    u32(40); u32(W); u32(H); u16(1); u16(24); u32(0); u32(dataSize); u32(2835); u32(2835); u32(0); u32(0);
    for (int y = (int)H - 1; y >= 0; y--) {
        for (uint32_t x = 0; x < W; x++) {
            const uint8_t* p = &px[((size_t)y * W + x) * 4];
            fputc(p[2], f); fputc(p[1], f); fputc(p[0], f);
        }
        for (uint32_t k = 0; k < rowpad; k++) fputc(0, f);
    }
    fclose(f);
    fprintf(stderr, "[scanout] flip %llu -> %s (%ux%u)\n", (unsigned long long)n, fn, W, H);
}

// The rendered frame the back-half hands us (the "scanout" image). Guarded by a mutex because the
// renderer thread writes it while guest threads / tests read it via present_readback.
std::mutex           g_frame_mx;
std::vector<uint8_t> g_frame;      // w*h*4 bytes when a frame is present
uint32_t             g_frame_w = 0, g_frame_h = 0;
std::atomic<bool>    g_have_frame{false};
// True when the most recent present event was a FLIP (scanout = the guest buffer, read lazily);
// false when a renderer frame was handed in afterwards (scanout = g_frame). Most-recent-wins is the
// honest model: the flip follows the frame's draws in the stream, so it supersedes them, and a later
// rendered frame supersedes the flip.
std::atomic<bool>    g_scanout_is_flip{false};
}

void present_write_frame(const void* pixels, uint32_t w, uint32_t h) {
    if (!pixels || !w || !h) return;
    size_t bytes = (size_t)w * h * 4;
    std::lock_guard<std::mutex> lk(g_frame_mx);
    g_frame.resize(bytes);
    std::memcpy(g_frame.data(), pixels, bytes);
    g_frame_w = w; g_frame_h = h;
    g_have_frame.store(true, std::memory_order_release);
    g_scanout_is_flip.store(false, std::memory_order_release);   // renderer frame is now newest
}

bool present_has_frame() { return g_have_frame.load(std::memory_order_acquire); }

// Read the current front guest buffer into `out` (w*h*4 RGBA), de-swizzling per the registered
// tiling attribute. This is the REAL scanout: the frame a flip presents is the guest buffer's
// contents — the game composes into it (The Messenger's title art is CPU-written into the flip
// buffer; its GPU composite draw carries CB_TARGET_MASK=0 at the draw and writes nothing). Reading
// LAZILY (at readback, not at flip) mirrors hardware scanout continuously fetching the buffer, so
// post-flip memory updates are visible. CONFIDENCE: MED on tiling_mode==1 meaning linear (Kyty
// VideoOutBufferAttribute); the observed tiled display surface is SW_4KB_S.
static bool read_guest_scanout(std::vector<uint8_t>& out, uint32_t* out_w, uint32_t* out_h) {
    int front = g_front.load(std::memory_order_relaxed);
    if (front < 0) return false;
    uint64_t addr = prosper_vo_buffer_addr(front);
    uint32_t w = prosper_vo_display_width(), h = prosper_vo_display_height();
    if (!addr || !w || !h || w > 8192 || h > 8192) return false;
    const uint32_t tmode = prosper_vo_display_tiling() == 1
                             ? (uint32_t)TileMode::Linear : (uint32_t)TileMode::Sw4KbS;
    const size_t nb          = (size_t)w * h * 4;
    const size_t tiled_bytes = tiled_surface_bytes(w, h, tmode);
    if (!guest_readable(addr, (uint32_t)nb)) return false;
    std::vector<uint8_t> tiled(tiled_bytes, 0);
    // Padded tail (h rounded to whole tile rows) may be unmapped — degrade to the validated w*h
    // bytes rather than dropping the frame (bottom tile-row only).
    std::memcpy(tiled.data(), (const void*)(uintptr_t)addr,
                guest_readable(addr, (uint32_t)tiled_bytes) ? tiled_bytes : nb);
    out.resize(nb);
    detile_surface(out.data(), tiled.data(), w, h, tmode);
    if (out_w) *out_w = w;
    if (out_h) *out_h = h;
    return true;
}

void present_flip(int buffer_index, int64_t /*flip_arg*/) {
    // Only accept a buffer the game actually registered; otherwise leave the front unchanged (an
    // invalid index shouldn't corrupt scanout state).
    if (buffer_index >= 0 && buffer_index < prosper_vo_buffer_count()) {
        g_front.store(buffer_index, std::memory_order_relaxed);
        // A flip makes the GUEST BUFFER the scanout source (until a renderer frame arrives later).
        g_scanout_is_flip.store(true, std::memory_order_release);
        if (getenv("PROSPER_SCANOUT_DUMP")) {
            std::vector<uint8_t> px; uint32_t w = 0, h = 0;
            if (read_guest_scanout(px, &w, &h))
                scanout_dump_bmp(px.data(), w, h, g_present_count.load(std::memory_order_relaxed));
        }
    }
    g_present_count.fetch_add(1, std::memory_order_relaxed);
}

int      present_front_index() { return g_front.load(std::memory_order_relaxed); }
uint64_t present_count()       { return g_present_count.load(std::memory_order_relaxed); }
uint32_t present_width()       { return prosper_vo_display_width(); }
uint32_t present_height()      { return prosper_vo_display_height(); }

size_t present_readback(void* dst, size_t dst_cap) {
    if (!dst) return 0;
    // Scanout source = whichever present event is newest: a flip (guest buffer, read lazily so
    // post-flip memory updates are visible, detiled per the registered tiling), else the rendered
    // frame the back-half handed in.
    if (!g_scanout_is_flip.load(std::memory_order_acquire) &&
        g_have_frame.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lk(g_frame_mx);
        if (!g_frame.empty() && dst_cap >= g_frame.size()) {
            std::memcpy(dst, g_frame.data(), g_frame.size());
            return g_frame.size();
        }
        return 0;
    }
    std::vector<uint8_t> px; uint32_t w = 0, h = 0;
    if (read_guest_scanout(px, &w, &h) && dst_cap >= px.size()) {
        std::memcpy(dst, px.data(), px.size());
        return px.size();
    }
    // No flip yet / unreadable buffer: fall back to the rendered frame if any.
    if (g_have_frame.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lk(g_frame_mx);
        if (!g_frame.empty() && dst_cap >= g_frame.size()) {
            std::memcpy(dst, g_frame.data(), g_frame.size());
            return g_frame.size();
        }
    }
    return 0;
}

void present_reset() {
    g_front.store(-1, std::memory_order_relaxed);
    g_present_count.store(0, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(g_frame_mx);
    g_frame.clear(); g_frame_w = g_frame_h = 0;
    g_have_frame.store(false, std::memory_order_release);
    g_scanout_is_flip.store(false, std::memory_order_release);
}

} // namespace prosper::gpu
