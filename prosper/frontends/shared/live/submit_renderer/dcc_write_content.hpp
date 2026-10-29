// Owned facts from complete metadata-plane writes, never borrowed output pointers.
#pragma once
#include <cstdint>
#include <optional>
#include <span>
#include <vector>
#include <unordered_map>

namespace prosper::frontend {
struct RttSurf;
namespace submit_renderer {
struct DccWritePlane {
    uint64_t target = 0, address = 0, bytes = 0;
};
struct DccWriteFact {
    DccWritePlane plane;
    std::optional<uint8_t> uniform_byte;
};
std::vector<DccWriteFact> capture_dcc_write_facts(std::span<const DccWritePlane>,
    uint64_t address, uint64_t bytes, std::span<const uint8_t> content);
void refresh_dcc_write_planes(std::vector<DccWritePlane>&,
                              const std::unordered_map<uint64_t, RttSurf>&);
void apply_dcc_write_facts(std::unordered_map<uint64_t, RttSurf>&,
                           std::span<const DccWriteFact>);
void track_rtt_dcc_write_plane(uint64_t target, const RttSurf&);
} // namespace submit_renderer
} // namespace prosper::frontend
