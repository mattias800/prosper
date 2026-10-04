#include "gpu/execute/original_fragment_draw_producer.hpp"
#include "gpu/execute/scalar_bank_read_point.hpp"

namespace prosper::gpu {
OriginalFragmentDrawProducer::OriginalFragmentDrawProducer(
    std::shared_ptr<const NativeGraphicsStageCompilation> vertex,
    std::shared_ptr<const FragmentScalarBank> bank, const RenderState& state)
    : vertex_(std::move(vertex)), bank_(std::move(bank)), launch_(state.ps_raster_launch),
      float_mode_(state.ps_float_mode), float_flags_(state.ps_float_flags),
      launch_rsrc1_(state.ps_launch_rsrc1) {}
uint64_t OriginalFragmentDrawProducer::read_point_identity() const {
    return bank_->read_point_identity();
}
bool OriginalFragmentDrawProducer::matches_modules(const SharedShaderWords& vertex,
                                                   std::span<const uint32_t> geometry,
                                                   std::span<const uint32_t> fragment) const {
    return vertex_ && geometry.empty() && fragment.empty() &&
           vertex_->matches(vertex_->source(), vertex);
}
bool OriginalFragmentDrawProducer::matches(const RasterQuadInputs& in) const {
    return vertex_ && bank_ && in.scalar_bank == bank_ && !in.owned_wave_pending && in.source_fs &&
           in.source_fs->empty() && in.source_gs && in.source_gs->empty() &&
           vertex_->stage() == ShaderProgramStage::Vertex &&
           vertex_->matches(vertex_->source(), in.source_vs) &&
           bank_->matches(in.raw_code, in.vgpr_requirements, in.entry) && in.launch == launch_ &&
           in.float_mode == float_mode_ && in.float_flags == float_flags_ &&
           in.launch_rsrc1 == launch_rsrc1_;
}
std::shared_ptr<const OriginalFragmentDrawProducer>
seal_original_fragment_draw_producer(const OrderedScalarBankReadPoint& point, const GpuState& state,
                                     uint64_t order,
                                     std::shared_ptr<const NativeGraphicsStageCompilation> vertex,
                                     std::shared_ptr<const FragmentScalarBank> bank) {
    const auto render = extract_render_state(state);
    if (!point.belongs_to_draw(state, point.source_submit(), order, render.ps_addr) || !vertex ||
        vertex->stage() != ShaderProgramStage::Vertex || !bank ||
        vertex->read_point_identity() != point.identity() ||
        bank->read_point_identity() != point.identity() ||
        !bank->matches(point.packet_source(render.ps_addr),
                       point.packet_requirements(render.ps_addr), point.entry()))
        return {};
    const auto expected = point.original_source(render.es_addr);
    const auto& observed = vertex->source().words;
    if (!expected || !observed || expected.get() != observed.get() ||
        expected.owner_before(observed) || observed.owner_before(expected) ||
        !vertex->matches(vertex->source(), vertex->module()))
        return {};
    return std::shared_ptr<const OriginalFragmentDrawProducer>(
        new OriginalFragmentDrawProducer(std::move(vertex), std::move(bank), render));
}
} // namespace prosper::gpu
