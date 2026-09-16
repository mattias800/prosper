#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace prosper::frontend {

// Renderer faces are proved by their generation; each remaining face needs its own exact
// guest prefix. Prefix lengths cannot be collapsed to a total: two short faces can exchange
// readable bytes without changing their concatenation.
struct DepthCubeSourceSnapshot {
    std::array<size_t, 6> prefixes{};
    uint32_t renderer_mask = 0;
    bool active = false;
};

struct DepthCubeSourceLayout {
    uint64_t base = 0;
    uint64_t stride = 0;
    size_t face_bytes = 0;

    bool fits(uint64_t watched_base, size_t watched_bytes) const {
        if (!base || !face_bytes || stride < face_bytes || base < watched_base ||
            watched_bytes > UINT64_MAX - watched_base || stride > (UINT64_MAX - base) / 5)
            return false;
        const uint64_t last = base + 5 * stride;
        return last <= watched_base + watched_bytes &&
            face_bytes <= watched_base + watched_bytes - last;
    }
};

template<class Copy>
size_t capture_depth_cube_source(const DepthCubeSourceLayout& layout, uint32_t renderer_mask,
                                bool eager_control, std::vector<uint8_t>& scratch,
                                DepthCubeSourceSnapshot& snapshot, Copy&& copy,
                                size_t& read_bytes) {
    snapshot = {};
    snapshot.active = true;
    snapshot.renderer_mask = renderer_mask;
    size_t guest_faces = 0;
    for (unsigned face = 0; face < 6; ++face)
        guest_faces += !(renderer_mask & (1u << face));
    scratch.resize(layout.face_bytes * (guest_faces ? guest_faces : (eager_control ? 1u : 0u)));
    // Diagnostic control reads the renderer-owned faces too, before overwriting those discarded
    // bytes with the guest-only snapshot. Both policies retain exactly the same cache contents.
    if (eager_control)
        for (unsigned face = 0; face < 6; ++face)
            if (renderer_mask & (1u << face))
                read_bytes += copy(scratch.data(), layout.base + face * layout.stride,
                                   layout.face_bytes);
    size_t total = 0;
    for (unsigned face = 0; face < 6; ++face) {
        if (renderer_mask & (1u << face)) continue;
        const size_t got = copy(scratch.data() + total, layout.base + face * layout.stride,
                                layout.face_bytes);
        snapshot.prefixes[face] = got;
        total += got;
        read_bytes += got;
    }
    return total;
}

template<class Span, class Equal>
bool equal_depth_cube_source(const DepthCubeSourceLayout& layout,
                             const DepthCubeSourceSnapshot& snapshot,
                             const std::vector<uint8_t>& bytes,
                             Span&& span, Equal&& equal, size_t& compared_bytes) {
    size_t offset = 0;
    for (unsigned face = 0; face < 6; ++face) {
        if (snapshot.renderer_mask & (1u << face)) continue;
        const uint64_t addr = layout.base + face * layout.stride;
        const size_t prefix = snapshot.prefixes[face];
        if (prefix > layout.face_bytes || offset > bytes.size() ||
            prefix > bytes.size() - offset || span(addr, layout.face_bytes) != prefix)
            return false;
        size_t compared = 0;
        const bool same = !prefix || equal(bytes.data() + offset, addr, prefix, compared);
        compared_bytes += compared;
        if (!same || compared != prefix) return false;
        offset += prefix;
    }
    return offset == bytes.size();
}

} // namespace prosper::frontend
