#pragma once
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/execute/scalar_bank_read_point.hpp"

namespace prosper::gpu {
// A checked compile-input handoff, not a code-only cache lookup or resource admission token.
// Minting requires the actual live ordered draw association; normal native compilation consumes
// this exact snapshot analysis instead of re-observing the guest code address.
class CheckedGraphicsSource {
    friend std::shared_ptr<const CheckedGraphicsSource>
    checked_graphics_source(std::shared_ptr<const OrderedScalarBankReadPoint>, const GpuState&,
                            uint64_t, uint64_t, ShaderProgramStage);
    std::shared_ptr<const OrderedScalarBankReadPoint> point_;
    const GpuState* state_ = nullptr;
    uint64_t address_ = 0, order_ = 0;
    ShaderProgramStage stage_;
    GraphicsReadSource source_;
    CheckedGraphicsSource(std::shared_ptr<const OrderedScalarBankReadPoint> point,
                          const GpuState& state, uint64_t address, uint64_t order,
                          ShaderProgramStage stage, GraphicsReadSource source)
        : point_(std::move(point)), state_(&state), address_(address), order_(order), stage_(stage),
          source_(std::move(source)) {}

public:
    uint64_t address() const { return address_; }
    ShaderProgramStage stage() const { return stage_; }
    uint64_t read_point_identity() const { return point_->identity(); }
    const SharedShaderAnalysis& analysis() const;
    const SharedShaderWords& original_words() const { return source_.words; }
    const GraphicsReadSource& source() const { return source_; }
    bool current() const;
    bool belongs_to(const GpuState&, uint64_t address, uint64_t order, ShaderProgramStage) const;
};
std::shared_ptr<const CheckedGraphicsSource>
checked_graphics_source(std::shared_ptr<const OrderedScalarBankReadPoint>, const GpuState&,
                        uint64_t address, uint64_t command_order, ShaderProgramStage);
}   // namespace prosper::gpu
