#include "gpu/recompiler/fragment_quad_composition.hpp"
#include <algorithm>
#include <set>
#include <tuple>

namespace prosper::gpu {
namespace {
int64_t region(int32_t coordinate, uint32_t pixels) {
    if (!pixels) return 0;
    const int64_t value = coordinate;
    return value < 0 ? -((-value + pixels - 1) / pixels) : value / pixels;
}
auto scope(const FragmentQuadCompositionInput& quad) {
    return std::tuple{quad.view, quad.layer, quad.sample};
}
auto location(const FragmentQuadCompositionInput& quad) {
    return std::tuple{quad.view, quad.layer, quad.sample, quad.y, quad.x};
}
auto region_key(const FragmentQuadCompositionInput& quad, uint32_t pixels) {
    return std::tuple{scope(quad), region(quad.y, pixels), region(quad.x, pixels)};
}
auto key(const FragmentQuadCompositionInput& quad, uint32_t pixels) {
    return std::tuple{region_key(quad, pixels), quad.primitive, quad.y, quad.x};
}
} // namespace
FragmentQuadComposition
fragment_quad_composition_reference(std::span<const FragmentQuadCompositionInput> quads,
                                    FragmentQuadCompositionProfile profile) {
    const auto refuse = [](const char* reason) { return FragmentQuadComposition{{}, reason}; };
    if (profile.wave_break > 3) return refuse("fragment-quad-wave-break-encoding-unimplemented");
    if (quads.size() > UINT32_MAX) return refuse("fragment-quad-record-index-overflow");
    const uint32_t pixels = profile.wave_break ? (4u << profile.wave_break) : 0;
    std::vector<FragmentQuadCompositionInput> ordered(quads.begin(), quads.end());
    std::set<decltype(key(FragmentQuadCompositionInput{}, pixels))> keys;
    std::set<uint32_t> records;
    for (const auto& quad : ordered) {
        if (quad.record == UINT32_MAX || !records.insert(quad.record).second)
            return refuse("fragment-quad-record-identity-invalid");
        if ((quad.x % 2) || (quad.y % 2)) return refuse("fragment-quad-origin-invalid");
        if (quad.backing != 15 || (quad.live & ~quad.backing) ||
            (quad.export_candidates & ~quad.backing) || !quad.live)
            return refuse("fragment-quad-live-or-backing-unavailable");
        if (!keys.insert(key(quad, pixels)).second)
            return refuse("fragment-quad-normalized-origin-duplicate");
    }
    std::sort(ordered.begin(), ordered.end(),
              [&](const auto& a, const auto& b) { return key(a, pixels) < key(b, pixels); });
    FragmentQuadComposition result;
    std::vector<FragmentQuadCompositionInput> current;
    const auto flush = [&] {
        if (current.empty()) return;
        FragmentQuadComposedWave wave;
        wave.records.fill(UINT32_MAX);
        wave.primitives.fill(UINT32_MAX);
        wave.quad_count = static_cast<uint32_t>(current.size());
        for (uint32_t index = 0; index < wave.quad_count; ++index) {
            const auto& quad = current[index];
            wave.records[index] = quad.record;
            wave.primitives[index] = quad.primitive;
            const uint32_t shift = index * 4;
            wave.backing |= uint64_t(quad.backing) << shift;
            wave.live |= uint64_t(quad.live) << shift;
            wave.export_candidates |= uint64_t(quad.export_candidates) << shift;
            if (index && quad.primitive != current[index - 1].primitive)
                wave.primitive_transitions |= uint16_t(1u << (index - 1));
        }
        result.waves.push_back(wave);
        current.clear();
    };
    for (const auto& quad : ordered) {
        const bool changed_region =
            !current.empty() && region_key(quad, pixels) != region_key(current.front(), pixels);
        const bool collision = std::any_of(current.begin(), current.end(), [&](const auto& peer) {
            return location(quad) == location(peer);
        });
        if (current.size() == 16 || changed_region || collision) flush();
        current.push_back(quad);
    }
    flush();
    return result;
}
std::optional<uint32_t> fragment_quad_system_word(const FragmentQuadComposedWave& wave,
                                                  uint16_t parameter_offset,
                                                  std::optional<bool> bc_optimize) {
    if (!bc_optimize || !wave.quad_count || wave.quad_count > 16 ||
        (wave.primitive_transitions & 0x8000u))
        return {};
    const uint32_t occupied_transitions = (1u << (wave.quad_count - 1)) - 1;
    if (wave.primitive_transitions & ~occupied_transitions) return {};
    return (uint32_t(*bc_optimize) << 31) | (uint32_t(wave.primitive_transitions) << 16) |
           parameter_offset;
}
} // namespace prosper::gpu
