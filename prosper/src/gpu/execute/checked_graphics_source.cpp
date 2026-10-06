#include "gpu/execute/checked_graphics_source.hpp"
#include "gpu/execute/registered_graphics_source_internal.hpp"
#include "gpu/execute/shader_cache_internal.hpp"

namespace prosper::gpu {
SharedShaderAnalysis checked_graphics_source_analysis(const CheckedGraphicsSource* source) {
    return source ? source->analysis() : SharedShaderAnalysis{};
}
bool checked_graphics_source_current(const CheckedGraphicsSource* source) {
    return source && source->current();
}
bool checked_graphics_source_requires_owned_waves(const CheckedGraphicsSource* source,
                                                  bool fragment_launch_wave64) {
    // Immutable classification only; this cannot mint permission or revive an expired point.
    return source && source->source().decoded &&
           source->source().decoded->requires_owned_waves(fragment_launch_wave64);
}
GraphicsReadSource checked_graphics_source_observation(const CheckedGraphicsSource* source) {
    return source && source->current() ? source->source() : GraphicsReadSource{};
}
const SharedShaderAnalysis& CheckedGraphicsSource::analysis() const {
    return source_.native_analysis->analysis();
}
bool CheckedGraphicsSource::current() const {
    return state_ && point_ &&
           point_->belongs_to_draw(*state_, point_->source_submit(), order_,
                                   point_->fragment_address());
}
bool CheckedGraphicsSource::belongs_to(const GpuState& state, uint64_t address, uint64_t order,
                                       ShaderProgramStage stage) const {
    return state_ == &state && address_ == address && order_ == order && stage_ == stage &&
           current();
}
std::shared_ptr<const CheckedGraphicsSource>
checked_graphics_source(std::shared_ptr<const OrderedScalarBankReadPoint> point,
                        const GpuState& state, uint64_t address, uint64_t order,
                        ShaderProgramStage stage) {
    if (!point || (stage != ShaderProgramStage::Vertex && stage != ShaderProgramStage::Fragment) ||
        !point->belongs_to_draw(state, point->source_submit(), order, point->fragment_address()))
        return {};
    const auto original = point->original_source(address);
    const auto decoded = point->decoded_source(address);
    const auto native = point->native_analysis(address);
    const auto* header = point->registered_header(address);
    const auto snapshot = point->header_snapshot(address);
    const uint32_t expected_type = stage == ShaderProgramStage::Fragment ? 1u : 2u;
    if (!original || !decoded || !native || !native->belongs_to(decoded) || !native->analysis() ||
        !header || !snapshot || snapshot->type != expected_type ||
        reinterpret_cast<uint64_t>(snapshot->code) != address || original.get() != &decoded->code ||
        original.owner_before(decoded) || decoded.owner_before(original))
        return {};
    GraphicsReadSource source;
    source.words = original;
    source.chains = std::shared_ptr<const std::vector<RawNestedWideChain>>(
        decoded, &decoded->owned_nested_wide_chains);
    source.packet_requirements = std::shared_ptr<const FragmentPacketVgprRequirements>(
        decoded, &decoded->packet_requirements);
    source.vertex_effects =
        std::shared_ptr<const OriginalGraphicsStageEffects>(decoded, &decoded->original_effects[0]);
    source.fragment_effects =
        std::shared_ptr<const OriginalGraphicsStageEffects>(decoded, &decoded->original_effects[1]);
    source.decoded = decoded;
    source.native_analysis = native;
    source.registered_header = header;
    source.header_snapshot = snapshot;
    return std::shared_ptr<const CheckedGraphicsSource>(new CheckedGraphicsSource(
        std::move(point), state, address, order, stage, std::move(source)));
}
}   // namespace prosper::gpu
