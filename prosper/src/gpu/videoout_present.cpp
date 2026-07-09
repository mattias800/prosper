// videoout_present.cpp — see videoout_present.hpp.
#include "videoout_present.hpp"
#include <atomic>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

// The display surface lives in the libSceVideoOut buffer registry (hle_graphics.cpp), exposed via
// these C hooks. Reading through them keeps the present layer decoupled from the HLE.
extern "C" int      prosper_vo_buffer_count();
extern "C" uint32_t prosper_vo_display_width();
extern "C" uint32_t prosper_vo_display_height();
extern "C" uint64_t prosper_vo_display_format();
extern "C" uint64_t prosper_vo_buffer_addr(int i);

namespace prosper::gpu {
namespace {
std::atomic<int>      g_front{-1};
std::atomic<uint64_t> g_present_count{0};
std::atomic<uint64_t> g_frame_seq{0};   // # of rendered frames handed in via present_write_frame

// The rendered frame the back-half hands us (the "scanout" image). Guarded by a mutex because the
// renderer thread writes it while guest threads / tests read it via present_readback.
std::mutex           g_frame_mx;
std::vector<uint8_t> g_frame;      // w*h*4 bytes when a frame is present
uint32_t             g_frame_w = 0, g_frame_h = 0;
std::atomic<bool>    g_have_frame{false};
}

void present_write_frame(const void* pixels, uint32_t w, uint32_t h) {
    if (!pixels || !w || !h) return;
    size_t bytes = (size_t)w * h * 4;
    std::lock_guard<std::mutex> lk(g_frame_mx);
    g_frame.resize(bytes);
    std::memcpy(g_frame.data(), pixels, bytes);
    g_frame_w = w; g_frame_h = h;
    g_have_frame.store(true, std::memory_order_release);
    g_frame_seq.fetch_add(1, std::memory_order_relaxed);
}

uint64_t present_frame_seq() { return g_frame_seq.load(std::memory_order_relaxed); }

bool present_has_frame() { return g_have_frame.load(std::memory_order_acquire); }

void present_flip(int buffer_index, int64_t /*flip_arg*/) {
    // Only accept a buffer the game actually registered; otherwise leave the front unchanged (an
    // invalid index shouldn't corrupt scanout state).
    if (buffer_index >= 0 && buffer_index < prosper_vo_buffer_count())
        g_front.store(buffer_index, std::memory_order_relaxed);
    uint64_t n = g_present_count.fetch_add(1, std::memory_order_relaxed);
    // PROSPER_DUMP_SCANOUT=<dir>: dump the RAW guest display buffer the game just flipped (what the game
    // actually puts on screen — vs our execute_and_present composite render). First 8 flips + every 60th.
    // Written as raw w*h*4 bytes; convert offline. This reveals whether the game's real display holds the
    // title/scene even when the captured draws are a no-op composite.
    if (const char* dir = getenv("PROSPER_DUMP_SCANOUT"); dir && (n < 8 || n % 60 == 0)) {
        int idx = (buffer_index >= 0 && buffer_index < prosper_vo_buffer_count()) ? buffer_index : g_front.load();
        uint64_t addr = prosper_vo_buffer_addr(idx);
        uint32_t w = prosper_vo_display_width(), h = prosper_vo_display_height();
        if (addr && w && h) {
            char fn[512]; snprintf(fn, sizeof fn, "%s/scanout_%04llu_buf%d.rgba", dir, (unsigned long long)n, idx);
            if (FILE* f = fopen(fn, "wb")) { fwrite((const void*)(uintptr_t)addr, 1, (size_t)w * h * 4, f); fclose(f); }
        }
    }
}

int      present_front_index() { return g_front.load(std::memory_order_relaxed); }
uint64_t present_count()       { return g_present_count.load(std::memory_order_relaxed); }
uint32_t present_width()       { return prosper_vo_display_width(); }
uint32_t present_height()      { return prosper_vo_display_height(); }

size_t present_readback(void* dst, size_t dst_cap) {
    if (!dst) return 0;
    // Prefer the real rendered frame from the back-half, if one has been handed in.
    if (g_have_frame.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lk(g_frame_mx);
        if (!g_frame.empty() && dst_cap >= g_frame.size()) {
            std::memcpy(dst, g_frame.data(), g_frame.size());
            return g_frame.size();
        }
        return 0;
    }
    // Fallback: scan out the raw guest display buffer (contents are whatever the guest put there).
    int front = g_front.load(std::memory_order_relaxed);
    if (front < 0) return 0;
    uint64_t addr = prosper_vo_buffer_addr(front);
    uint32_t w = prosper_vo_display_width(), h = prosper_vo_display_height();
    if (!addr || !w || !h) return 0;
    size_t bytes = (size_t)w * h * 4;   // 32-bit color (BGRA/RGBA per the registered pixel format)
    if (dst_cap < bytes) return 0;
    std::memcpy(dst, (const void*)(uintptr_t)addr, bytes);
    return bytes;
}

void present_reset() {
    g_front.store(-1, std::memory_order_relaxed);
    g_present_count.store(0, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(g_frame_mx);
    g_frame.clear(); g_frame_w = g_frame_h = 0;
    g_have_frame.store(false, std::memory_order_release);
}

} // namespace prosper::gpu
