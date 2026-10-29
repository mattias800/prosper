// Scan only known metadata planes intersected by this write, under the existing queue lock.
#include "shared/live/submit_renderer/dcc_write_content.hpp"
#include <cstdint>
#include <cstddef>
#include <span>
#include <optional>
#include <vector>
#include "shared/live/live_renderer_internal.hpp"
#include <algorithm>
#include <mutex>

namespace prosper::frontend::submit_renderer {
std::vector<DccWriteFact> capture_dcc_write_facts(std::span<const DccWritePlane> planes,
                                                  uint64_t address, uint64_t bytes,
                                                  std::span<const uint8_t> content) {
    std::vector<DccWriteFact> facts;
    if (!address || !bytes) return facts;
    for (const auto& plane : planes) {
        if (!plane.address || !plane.bytes ||
            (address <= plane.address ? plane.address - address >= bytes
                                      : address - plane.address >= plane.bytes))
            continue;
        DccWriteFact fact{plane, std::nullopt};
        if (address <= plane.address) {
            const auto offset = plane.address - address;
            if (offset < content.size() && plane.bytes <= content.size() - offset &&
                offset <= bytes && plane.bytes <= bytes - offset) {
                const auto data =
                    content.subspan(static_cast<size_t>(offset), static_cast<size_t>(plane.bytes));
                if (std::all_of(data.begin(), data.end(),
                                [&](uint8_t byte) { return byte == data.front(); }))
                    fact.uniform_byte = data.front();
            }
        }
        facts.push_back(fact);
    }
    return facts;
}
void refresh_dcc_write_planes(std::vector<DccWritePlane>& planes, const RttCache& cache) {
    planes.clear();
    for (const auto& [target, surface] : cache)
        if (surface.dcc_metadata_addr && surface.dcc_metadata_bytes)
            planes.push_back({target, surface.dcc_metadata_addr, surface.dcc_metadata_bytes});
}
void apply_dcc_write_facts(RttCache& cache, std::span<const DccWriteFact> facts) {
    for (const auto& fact : facts) {
        const auto found = cache.find(fact.plane.target);
        if (found != cache.end() && found->second.dcc_metadata_dirty &&
            found->second.dcc_metadata_addr == fact.plane.address &&
            found->second.dcc_metadata_bytes == fact.plane.bytes)
            found->second.dcc_uniform_write_byte = fact.uniform_byte;
    }
}
void track_rtt_dcc_write_plane(uint64_t target, const RttSurf& surface) {
    auto& pending = pending_guest_gpu_writes();
    std::lock_guard<std::mutex> lock(pending.mutex);
    auto& planes = pending.dcc_planes;
    std::erase_if(planes, [=](const auto& plane) { return plane.target == target; });
    if (surface.dcc_metadata_addr && surface.dcc_metadata_bytes)
        planes.push_back({target, surface.dcc_metadata_addr, surface.dcc_metadata_bytes});
}
} // namespace prosper::frontend::submit_renderer
