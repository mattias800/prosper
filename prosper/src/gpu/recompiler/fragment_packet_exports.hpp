#pragma once
#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace prosper::gpu {
// Immutable code-generation/consumer policy. LegacyRaw observes enabled sources even EXEC-off;
// Architectural observes only guest-EXEC lanes. Neither policy establishes raster/commit authority.
enum class FragmentPacketExportObservation : uint8_t { LegacyRaw, Architectural };
inline constexpr uint32_t kFragmentPacketArchitecturalExportWords = 14;
struct FragmentPacketExportSite {
    uint32_t pc = 0, target = 0, en = 0, compr = 0, done = 0, vm = 0;
    bool operator==(const FragmentPacketExportSite&) const = default;
};
// Physical source words, not converted shader-output formats. COMPR uses two packed words;
// unobserved words are absent, not genuine zeros. PC order is original-program forward order.
struct FragmentPacketExportEvent {
    FragmentPacketExportSite site;
    bool exec = false, export_enabled = false;
    std::array<std::optional<uint32_t>, 4> source_words;
};
struct FragmentPacketExportComponent {
    uint32_t bits = 0, width = 32;
    bool operator==(const FragmentPacketExportComponent&) const = default;
};
struct FragmentPacketArchitecturalLane {
    std::vector<FragmentPacketExportEvent> events;
    std::array<std::array<std::optional<FragmentPacketExportComponent>, 4>, 8> colors;
    std::array<std::optional<FragmentPacketExportComponent>, 4> depth;
    bool valid_mask_available = false, valid_mask = false, terminal_done = false;
    bool export_enabled = false, commit_eligible = false;
};
struct FragmentPacketArchitecturalResult {
    std::vector<FragmentPacketArchitecturalLane> lanes;   // EMPTY on any malformed lane/site
    std::string rejection;
    uint32_t lane = UINT32_MAX, pc = UINT32_MAX;
};
uint32_t fragment_packet_export_source_mask(uint32_t en, bool compr);
const char* fragment_packet_architectural_export_gap(const FragmentPacketExportSite&);
std::string fragment_packet_export_schema_marker(std::span<const FragmentPacketExportSite>);
FragmentPacketArchitecturalResult
decode_fragment_packet_architectural_exports(std::span<const FragmentPacketExportSite>,
                                             std::span<const uint32_t> records);
}   // namespace prosper::gpu
