#include "gpu/recompiler/fragment_draw_capacity.hpp"

namespace prosper::gpu {
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
                       const RasterQuadCollector& collector, std::string& rejection) {
    rejection.clear();
    const auto refuse = [&](const char* reason) -> std::shared_ptr<const FragmentDrawCapacity> {
        rejection = reason;
        return {};
    };
    if (!kernel || kernel->program.packet.spirv.empty() ||
        !kernel->program.packet.rejection.empty() || !kernel->layout.gpu_capacity)
        return refuse("fragment-draw-capacity-kernel-unavailable");
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
    return std::shared_ptr<const FragmentDrawCapacity>(
        new FragmentDrawCapacity(std::move(kernel), words, collector));
}
}   // namespace prosper::gpu
