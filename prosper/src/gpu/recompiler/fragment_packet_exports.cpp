#include "gpu/recompiler/fragment_packet_exports.hpp"
#include <utility>

namespace prosper::gpu {
uint32_t fragment_packet_export_source_mask(uint32_t en, bool compr) {
    // AMD RDNA2 Table 56: compressed EN0/EN2 select VSRC0(R,G)/VSRC1(B,A).
    return compr ? ((en & 1u ? 1u : 0u) | (en & 4u ? 2u : 0u)) : en;
}
const char* fragment_packet_architectural_export_gap(const FragmentPacketExportSite& site) {
    if (site.en > 15 || site.compr > 1 || site.done > 1 || site.vm > 1)
        return "packet-export-fields-invalid";
    if (site.compr && site.en != 0 && site.en != 3 && site.en != 12 && site.en != 15)
        return "packet-compressed-export-channel-mask-invalid";
    if (site.target < 8) return nullptr;
    if (site.target == 8) {
        if (site.compr) return "packet-compressed-depth-export-unimplemented";
        if (site.en && !(site.en & ~5u)) return nullptr;
    }
    if (site.target == 9 && !site.en && !site.compr) return nullptr;
    return "packet-export-target-unimplemented";
}
std::string fragment_packet_export_schema_marker(std::span<const FragmentPacketExportSite> sites) {
    std::string marker = "Prosper.GuestFragmentPacket.ExportObservation=EXP14;";
    for (const auto& site : sites)
        marker += std::to_string(site.pc) + "," + std::to_string(site.target) + "," +
                  std::to_string(site.en) + "," + std::to_string(site.compr) + "," +
                  std::to_string(site.done) + "," + std::to_string(site.vm) + ";";
    return marker;
}
FragmentPacketArchitecturalResult
decode_fragment_packet_architectural_exports(std::span<const FragmentPacketExportSite> sites,
                                             std::span<const uint32_t> records) {
    FragmentPacketArchitecturalResult result;
    const auto reject = [&](const char* reason) {
        result.lanes.clear();
        result.rejection = reason;
        return result;
    };
    if (sites.empty() || sites.size() > 64 ||
        records.size() != 64 * sites.size() * kFragmentPacketArchitecturalExportWords)
        return reject("packet-architectural-export-abi-invalid");
    for (size_t index = 0; index < sites.size(); ++index) {
        if ((index && sites[index].pc <= sites[index - 1].pc) ||
            fragment_packet_architectural_export_gap(sites[index]))
            return reject("packet-architectural-export-schema-invalid");
    }
    std::vector<FragmentPacketArchitecturalLane> staged(64);
    for (uint32_t lane = 0; lane < 64; ++lane) {
        result.lane = lane;
        auto& output = staged[lane];
        for (size_t index = 0; index < sites.size(); ++index) {
            const auto& site = sites[index];
            result.pc = site.pc;
            const auto record = records.subspan((lane * sites.size() + index) *
                                                    kFragmentPacketArchitecturalExportWords,
                                                kFragmentPacketArchitecturalExportWords);
            if (!record[0]) {
                for (uint32_t field : record)
                    if (field) return reject("packet-unreached-export-invalid");
                continue;
            }
            if (record[0] != 1 || record[1] > 1 || record[2] > 1 || record[3] != site.target ||
                record[4] != site.en || record[5] != site.compr || record[6] != site.done ||
                record[7] != site.vm || record[12] != site.pc)
                return reject("packet-architectural-export-record-invalid");
            if (output.terminal_done) return reject("packet-export-after-done-unimplemented");
            if (!output.events.empty() && output.export_enabled != bool(record[2]))
                return reject("packet-export-eligibility-record-invalid");
            const uint32_t observed =
                record[1] ? fragment_packet_export_source_mask(site.en, site.compr) : 0;
            if (record[13] != observed) return reject("packet-export-observation-mask-invalid");
            FragmentPacketExportEvent event{site, bool(record[1]), bool(record[2]), {}};
            for (uint32_t word = 0; word < 4; ++word) {
                if (observed & (1u << word))
                    event.source_words[word] = record[8 + word];
                else if (record[8 + word])
                    return reject("packet-unobserved-export-payload-invalid");
            }
            // EXEC applies to payload independently of VM and of host raster eligibility. Enabled
            // channels accumulate across sites; a later null/EXEC-off export cannot erase data.
            for (uint32_t channel = 0; channel < 4; ++channel) {
                if (!record[1] || !(site.en & (1u << channel))) continue;
                const uint32_t word = site.compr ? channel / 2 : channel;
                const auto bits = site.compr ? (record[8 + word] >> ((channel & 1u) * 16)) & 0xffffu
                                             : record[8 + word];
                const FragmentPacketExportComponent component{bits, site.compr ? 16u : 32u};
                if (site.target < 8)
                    output.colors[site.target][channel] = component;
                else if (site.target == 8)
                    output.depth[channel] = component;
            }
            output.export_enabled = bool(record[2]);
            if (site.vm) {
                output.valid_mask_available = true;
                output.valid_mask = bool(record[1]);   // final executed VM wins, NOT final EXEC
            }
            output.terminal_done = bool(site.done);
            output.events.push_back(std::move(event));
        }
        // Primary PS contract: at least one VM and final executed EXP has DONE. Refuse an
        // incomplete bounded event stream; do not invent coverage or framebuffer conversion.
        if (!output.terminal_done || !output.valid_mask_available)
            return reject("packet-export-terminal-or-valid-mask-unproved");
        output.commit_eligible = output.valid_mask && output.export_enabled;
    }
    result.lane = result.pc = UINT32_MAX;
    result.lanes = std::move(staged);
    return result;
}
}   // namespace prosper::gpu
