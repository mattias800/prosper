// guest_texture_layout.cpp — see guest_texture_layout.hpp.
#include "guest_texture_layout.hpp"

#include <map>
#include <mutex>

namespace prosper::gpu {
namespace {

struct LinearTextureLayout {
    uint64_t end = 0;
    uint32_t row_pitch_bytes = 0;
};

std::mutex g_linear_texture_layout_mx;
std::map<uint64_t, LinearTextureLayout> g_linear_texture_layouts;

} // namespace

void register_guest_linear_texture_layout(uint64_t base, size_t bytes,
                                          uint32_t row_pitch_bytes) {
    if (!base || !bytes || !row_pitch_bytes || bytes > UINT64_MAX - base) return;
    std::lock_guard<std::mutex> lock(g_linear_texture_layout_mx);
    g_linear_texture_layouts[base] = {base + bytes, row_pitch_bytes};
}

void unregister_guest_linear_texture_layout(uint64_t base) {
    if (!base) return;
    std::lock_guard<std::mutex> lock(g_linear_texture_layout_mx);
    g_linear_texture_layouts.erase(base);
}

uint32_t guest_linear_texture_row_pitch(uint64_t address, uint32_t visible_row_bytes) {
    if (!address || !visible_row_bytes) return 0;
    std::lock_guard<std::mutex> lock(g_linear_texture_layout_mx);
    auto it = g_linear_texture_layouts.upper_bound(address);
    if (it == g_linear_texture_layouts.begin()) return 0;
    --it;
    return address < it->second.end && visible_row_bytes <= it->second.row_pitch_bytes
        ? it->second.row_pitch_bytes : 0;
}

} // namespace prosper::gpu
