// videoout_present.cpp — see videoout_present.hpp.
#include "videoout_present.hpp"
#include <atomic>
#include <cstring>

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
}

void present_flip(int buffer_index, int64_t /*flip_arg*/) {
    // Only accept a buffer the game actually registered; otherwise leave the front unchanged (an
    // invalid index shouldn't corrupt scanout state).
    if (buffer_index >= 0 && buffer_index < prosper_vo_buffer_count())
        g_front.store(buffer_index, std::memory_order_relaxed);
    g_present_count.fetch_add(1, std::memory_order_relaxed);
}

int      present_front_index() { return g_front.load(std::memory_order_relaxed); }
uint64_t present_count()       { return g_present_count.load(std::memory_order_relaxed); }
uint32_t present_width()       { return prosper_vo_display_width(); }
uint32_t present_height()      { return prosper_vo_display_height(); }

size_t present_readback(void* dst, size_t dst_cap) {
    int front = g_front.load(std::memory_order_relaxed);
    if (front < 0 || !dst) return 0;
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
}

} // namespace prosper::gpu
