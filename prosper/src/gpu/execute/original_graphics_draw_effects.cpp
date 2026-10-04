#include "gpu/recompiler/original_graphics_draw_effects.hpp"
#include "gpu/execute/native_graphics_source_lineage.hpp"
#include "gpu/execute/scalar_bank_read_point.hpp"
#include "gpu/recompiler/original_fragment_producer.hpp"

namespace prosper::gpu {
bool OriginalGraphicsDrawEffects::matches_draw(uint64_t submit, uint64_t order,
                                               const SharedShaderWords& vertex,
                                               const SharedShaderWords& fragment,
                                               std::span<const uint32_t> geometry,
                                               std::span<const uint32_t> fragment_words) const {
    if (!submit || submit != submit_ || order != order_ || !geometry.empty() || !vertex_ ||
        !vertex_->matches(vertex_->source(), vertex))
        return false;
    if (original_fragment_)
        return !fragment_ && fragment_words.empty() &&
               original_fragment_->matches_modules(vertex, geometry, fragment_words);
    return fragment_ && !fragment_words.empty() &&
           fragment_->matches(fragment_->source(), fragment);
}

std::shared_ptr<const OriginalGraphicsDrawEffects> seal_original_graphics_draw_effects(
    const OrderedScalarBankReadPoint& point, const GpuState& state, uint64_t order,
    std::shared_ptr<const NativeGraphicsStageCompilation> vertex,
    std::shared_ptr<const NativeGraphicsStageCompilation> fragment,
    std::shared_ptr<const OriginalFragmentDrawProducer> original_fragment) {
    const auto render = extract_render_state(state);
    if (!point.belongs_to_draw(state, point.source_submit(), order, render.ps_addr) || !vertex ||
        vertex->stage() != ShaderProgramStage::Vertex ||
        vertex->read_point_identity() != point.identity())
        return {};
    const auto associated = [&](const NativeGraphicsStageCompilation& selected, uint64_t address) {
        const auto expected = point.original_source(address);
        const auto& actual = selected.source().words;
        return expected && actual && expected.get() == actual.get() &&
               !expected.owner_before(actual) && !actual.owner_before(expected) &&
               selected.matches(selected.source(), selected.module());
    };
    if (!associated(*vertex, render.es_addr)) return {};
    if (original_fragment) {
        if (fragment || original_fragment->read_point_identity() != point.identity()) return {};
    } else if (!fragment || fragment->stage() != ShaderProgramStage::Fragment ||
               fragment->read_point_identity() != point.identity() ||
               !associated(*fragment, render.ps_addr))
        return {};
    if (!point.belongs_to_draw(state, point.source_submit(), order, render.ps_addr)) return {};
    return std::shared_ptr<const OriginalGraphicsDrawEffects>(new OriginalGraphicsDrawEffects(
        std::move(vertex), std::move(fragment), std::move(original_fragment), point.source_submit(),
        order));
}
} // namespace prosper::gpu
