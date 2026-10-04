#include "gpu/execute/checked_graphics_source.hpp"
#include "gpu/execute/registered_graphics_source_internal.hpp"
#include "gpu/execute/shader_cache_internal.hpp"

namespace prosper::gpu {
bool CheckedGraphicsSource::current() const {
    return state_ && point_ && point_->belongs_to_draw(*state_, point_->source_submit(), order_,
                                                      point_->fragment_address());
}
std::shared_ptr<const CheckedGraphicsSource> checked_graphics_source(
    std::shared_ptr<const OrderedScalarBankReadPoint> point, const GpuState& state,
    uint64_t address, uint64_t order, ShaderProgramStage stage) {
    if (!point || (stage != ShaderProgramStage::Vertex && stage != ShaderProgramStage::Fragment) ||
        !point->belongs_to_draw(state, point->source_submit(), order, point->fragment_address()))
        return {};
    const auto original = point->original_source(address);
    const auto decoded = point->decoded_source(address);
    const auto native = point->native_analysis(address);
    const auto* header = point->registered_header(address);
    const uint32_t expected_type = stage == ShaderProgramStage::Fragment ? 1u : 2u;
    if (!original || !decoded || !native || !native->belongs_to(decoded) || !native->analysis() ||
        !header || header->type != expected_type ||
        reinterpret_cast<uint64_t>(header->code) != address ||
        original.get() != &decoded->code || original.owner_before(decoded) ||
        decoded.owner_before(original))
        return {};
    return std::shared_ptr<const CheckedGraphicsSource>(new CheckedGraphicsSource(
        std::move(point), state, address, order, stage, original, native->analysis()));
}
}   // namespace prosper::gpu
