#include "gpu/recompiler/fragment_draw_capacity.hpp"
#include "gpu/recompiler/fragment_packet_mask_requirements.hpp"
#include <algorithm>
#include <bitset>
#include <cstring>
#include <string_view>

namespace prosper::gpu {
bool fragment_draw_architectural_exports_match(const FragmentPacketKernel& kernel) {
    const auto& packet = kernel.program.packet;
    if (packet.export_observation != FragmentPacketExportObservation::Architectural ||
        packet.export_sites.empty() || packet.export_sites.size() > 64 ||
        packet.exports_per_lane != packet.export_sites.size() ||
        kernel.program.status_offset !=
            64 * packet.export_sites.size() * kFragmentPacketArchitecturalExportWords ||
        packet.spirv.size() < 5 || packet.spirv[0] != 0x07230203u || kernel.guest_code.empty() ||
        kernel.guest_code.size() > 4096)
        return false;
    std::vector<Rdna2Inst> original;
    rdna2_walk(kernel.guest_code.data(), kernel.guest_code.size(), original);
    std::vector<FragmentPacketExportSite> original_sites, retained_sites;
    const auto inventory = [](const std::vector<Rdna2Inst>& instructions,
                              std::vector<FragmentPacketExportSite>& sites) {
        for (const auto& in : instructions)
            if (in.fmt == Rdna2Format::EXP)
                sites.push_back({in.pc, in.exp_target, in.exp_en, in.exp_compr,
                                 (in.words[0] >> 11) & 1u, (in.words[0] >> 12) & 1u});
    };
    inventory(original, original_sites);
    inventory(kernel.instructions, retained_sites);
    if (original_sites != packet.export_sites || retained_sites != original_sites) return false;
    for (size_t index = 0; index < original_sites.size(); ++index)
        if ((index && original_sites[index].pc <= original_sites[index - 1].pc) ||
            fragment_packet_architectural_export_gap(original_sites[index]))
            return false;
    const auto expected = fragment_packet_export_schema_marker(original_sites);
    bool marked = false;
    for (size_t pc = 5; pc < packet.spirv.size();) {
        const auto count = packet.spirv[pc] >> 16;
        if (!count || count > packet.spirv.size() - pc) return false;
        if ((packet.spirv[pc] & 0xffffu) == 330 && count > 1) { // OpModuleProcessed
            const auto* text = reinterpret_cast<const char*>(packet.spirv.data() + pc + 1);
            const auto* end =
                static_cast<const char*>(std::memchr(text, 0, (count - 1) * sizeof(uint32_t)));
            if (!end) return false;
            const std::string_view marker(text, static_cast<size_t>(end - text));
            if (marker.starts_with("Prosper.GuestFragmentPacket.ExportObservation=")) {
                if (marked || marker != expected) return false;
                marked = true;
            }
        }
        pc += count;
    }
    return marked;
}
bool FragmentDrawCapacity::matches_collector(const RasterQuadCollector& other) const {
    if (collector_.lane_words != other.lane_words ||
        collector_.record_words != other.record_words || collector_.max_quads != other.max_quads ||
        collector_.rejection != other.rejection || collector_.fields.size() != other.fields.size())
        return false;
    for (size_t index = 0; index < collector_.fields.size(); ++index) {
        const auto& a = collector_.fields[index];
        const auto& b = other.fields[index];
        if (a.kind != b.kind || a.index != b.index || a.selector != b.selector ||
            a.words != b.words || a.guest_initialization_proved != b.guest_initialization_proved)
            return false;
    }
    return true;
}
std::shared_ptr<const FragmentDrawCapacity>
fragment_draw_capacity(std::shared_ptr<const FragmentPacketKernel> kernel,
                       const RasterQuadCollector& collector, std::string& rejection,
                       FragmentDrawEntryRecipe entry_recipe,
                       std::vector<FragmentDrawRasterInput> raster_inputs) {
    rejection.clear();
    const auto refuse = [&](const char* reason) -> std::shared_ptr<const FragmentDrawCapacity> {
        rejection = reason;
        return {};
    };
    if (entry_recipe != FragmentDrawEntryRecipe::OwnedUserPrefixAndShaderDefinedMasks &&
        entry_recipe != FragmentDrawEntryRecipe::DrawBoundRasterSystemAndQuadMasks)
        return refuse("fragment-draw-entry-recipe-invalid");
    if (entry_recipe == FragmentDrawEntryRecipe::OwnedUserPrefixAndShaderDefinedMasks &&
        !raster_inputs.empty())
        return refuse("fragment-draw-raster-entry-schema-invalid");
    if (!kernel || kernel->program.packet.spirv.empty() ||
        !kernel->program.packet.rejection.empty() || !kernel->layout.gpu_capacity)
        return refuse("fragment-draw-capacity-kernel-unavailable");
    if (kernel->program.packet.export_observation != FragmentPacketExportObservation::LegacyRaw &&
        !fragment_draw_architectural_exports_match(*kernel))
        return refuse("fragment-draw-architectural-export-schema-mismatch");
    if (!collector.rejection.empty() || !collector.max_quads || collector.max_quads > 4096 ||
        collector.lane_words < kRasterQuadLaneFixedWords || collector.lane_words > 512 ||
        collector.record_words != collector.lane_words * 4)
        return refuse("fragment-draw-capacity-collector-invalid");
    uint64_t lane_words = kRasterQuadLaneFixedWords;
    if (collector.fields.size() > 128) return refuse("fragment-draw-capacity-collector-invalid");
    for (const auto& field : collector.fields) {
        if (!field.words || field.words > 4)
            return refuse("fragment-draw-capacity-collector-invalid");
        lane_words += field.words;
    }
    if (lane_words != collector.lane_words)
        return refuse("fragment-draw-capacity-collector-invalid");
    const auto& layout = kernel->layout;
    if (!layout.input_words || layout.output_words != kernel->program.packet.output_words.size())
        return refuse("fragment-draw-capacity-layout-invalid");
    if (entry_recipe == FragmentDrawEntryRecipe::DrawBoundRasterSystemAndQuadMasks) {
        const auto& packet = kernel->program.packet;
        if (raster_inputs.size() > 2 ||
            packet.export_observation != FragmentPacketExportObservation::Architectural ||
            packet.initial_mask_availability != kPacketInitialExec ||
            packet.input_stride != 2 * layout.vgprs.size() + 4 ||
            kernel->topology != FragmentPacketQuadTopology::ConsecutiveLogicalQuads)
            return refuse("fragment-draw-raster-entry-schema-invalid");
        std::bitset<256> registers;
        for (const auto& row : raster_inputs) {
            if (row.reg >= 256 || row.collector_word < 5 || row.collector_word > 6 ||
                registers.test(row.reg) ||
                std::find(layout.vgprs.begin(), layout.vgprs.end(), row.reg) == layout.vgprs.end())
                return refuse("fragment-draw-raster-entry-schema-invalid");
            registers.set(row.reg);
        }
    }
    const uint64_t waves = (uint64_t(collector.max_quads) + 15) / 16;
    const uint64_t input_span = uint64_t(layout.input_words) + 2;
    const uint64_t output_span = uint64_t(layout.output_words) + kPacketWaveOutputPrefix;
    const uint64_t input_words = kFragmentDrawHeaderWords + waves * input_span;
    const uint64_t output_words = waves * output_span;
    const uint64_t collector_words =
        kRasterQuadBufferHeaderWords + uint64_t(collector.max_quads) * collector.record_words;
    // Check the complete products before narrowing or allocating. Contiguous fixed strides prove
    // every capacity region disjoint; a mutable GPU route table cannot redirect a worker.
    constexpr uint64_t max_words = 16 * 1024 * 1024;
    if (input_words > max_words || output_words > max_words || collector_words > (8u << 18))
        return refuse("fragment-draw-capacity-storage-budget");
    uint32_t pixel_slots = 1;
    while (pixel_slots < collector.max_quads * 8) pixel_slots *= 2;
    std::array<uint32_t, kFragmentDrawAuthorityWords> words{kFragmentDrawAuthorityMagic,
                                                            static_cast<uint32_t>(waves),
                                                            static_cast<uint32_t>(input_words),
                                                            static_cast<uint32_t>(output_words),
                                                            static_cast<uint32_t>(input_span),
                                                            static_cast<uint32_t>(output_span),
                                                            collector.max_quads,
                                                            static_cast<uint32_t>(collector_words),
                                                            pixel_slots,
                                                            kFragmentDrawCommitHeaderWords +
                                                                pixel_slots};
    return std::shared_ptr<const FragmentDrawCapacity>(new FragmentDrawCapacity(
        std::move(kernel), words, collector, entry_recipe, std::move(raster_inputs)));
}
}   // namespace prosper::gpu
